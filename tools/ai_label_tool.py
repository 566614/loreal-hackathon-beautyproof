# -*- coding: utf-8 -*-
"""
可见 AI 标识取证工具（第九件证据）—— 从图上「平台自己承认是 AI 画的」字样取证。

为什么需要它（2026-10-02 实测触发）：
    生产模型 AIGC v4 在**全新风格的 AI 图**上会掉到 75%（12 张零样本，3 张直接判 0.0 = 真实）。
    漏检集中在「伪真实感 / 生活流 / 直播截图 / 场景静物」这类风格 —— 它们看起来太像照片。
    但实测发现：这批图右下角都带平台自动打的**「AI生成」标识**（OCR 置信度 0.95~0.99）。
    也就是说：**模型有盲区，但系统不必有盲区** —— 图上留着硬证据。
    本工具只做一件事：把这类可见标识认出来，作为独立、可审计、几乎零误伤的硬证据。

铁律（必须写在报告里，不能只在代码里）：
    1. 这是**补偿手段，不是万能解** —— 平台打标识的图能被抓住；洗掉标识再发的图本工具发现不了，
       那类图仍然只能靠 AIGC 模型 + 频域 + TruFor。
    2. 标识也可能是**被伪造/贴上去的**（有人 P 一行「AI生成」到真实图上），
       所以本证据只能证明「图上写着它是 AI 生成的」，不能证明「它确实是 AI 生成的」；
       遇到与图片内容明显矛盾的情况，仍需人工复核。
    3. OCR 置信度只说明「抄得像不像」，不说明内容真假 —— 沿用 OCR 工具的定位。

用法：
    python tools/ai_label_tool.py <图片路径>

来源：能力来自第三方 PaddleOCR（Apache-2.0），封装与判定规则为 BeautyProof 团队原创。
"""
import json
import re
import sys
from pathlib import Path

# 强模式：命中即视为「图上明确标注了 AI 生成」——需要 AI 语义 + 生成动作，两者同现
CN_STRONG = [
    r"AI\s*生成", r"AI\s*绘制", r"AI\s*创作", r"AI\s*合成", r"AI\s*出图",
    r"人工智能\s*生成", r"人工智能\s*绘制", r"由\s*AI\s*生成", r"本图.{0,4}AI",
    r"AI\s*辅助\s*生成", r"AIGC\s*生成",
]
EN_STRONG = [
    r"generated\s*by\s*ai", r"ai[\s\-_]?generated", r"made\s*with\s*ai",
    r"created\s*by\s*ai", r"ai[\s\-_]?artificial", r"synthetic\s*image",
]
# 辅助信息：平台名/生成器名，单独出现不足以定级，只用来给标识溯源
GENERATOR_HINTS = [
    "workbuddy", "即梦", "jimeng", "midjourney", "stable diffusion", "stablediffusion",
    "dall·e", "dalle", "dall-e", "flux", "leonardo", "liblib", "可画", "midjouney",
]

MIN_CONF = 0.85  # 低于这个置信度不认（抄得不像就别硬判）

_OCR_SINGLESTON = {}


def _ocr(image_path):
    """PaddleOCR 实例缓存 —— 每次新建要 10s+，同一进程里复用同一个。"""
    if "obj" not in _OCR_SINGLESTON:
        from paddleocr import PaddleOCR
        _OCR_SINGLESTON["obj"] = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
        )
    return _OCR_SINGLESTON["obj"]


def read_lines(image_path, reuse_ocr_evidence=True):
    """拿图上所有文字行。

    ⚠️ 编排上的关键设计：**优先复用 ocr_tool 已经落盘的证据**（outputs/ocr_<stem>.json）。
    PaddleOCR 首次加载要 60~90 秒，如果本工具自己再跑一遍 OCR，每次鉴定要多花一分半钟，
    而它要的其实只是 ocr 结果里的 text/bbox/confidence —— 同一份输入，第二次推理没有任何增量。
    这也是「工具编排有价值」的一个具体例子：第九件工具是第五件工具的下游二次分析，
    边际成本约 1 秒，不是 90 秒。
    没有 ocr 证据时（例如单独对本工具跑）才自己调 PaddleOCR。
    """
    if reuse_ocr_evidence:
        p = Path(__file__).resolve().parent.parent / "outputs" / f"ocr_{Path(image_path).stem}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                lines = data.get("evidence") or []
                if lines:
                    return lines
            except Exception:
                pass  # 证据坏了就退回自己跑，不因为缓存问题丢人
    return _ocr_lines(image_path)


def _ocr_lines(image_path):
    ocr = _ocr(image_path)
    lines = []
    for res in ocr.predict(str(image_path)):
        for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            lines.append({
                "text": str(text),
                "bbox": [round(min(xs), 1), round(min(ys), 1),
                         round(max(xs), 1), round(max(ys), 1)],
                "confidence": round(float(score), 4),
            })
    return lines


def detect(lines):
    """纯函数：从 OCR 文本行里判定「图上是否带可见 AI 标识」。返回命中明细（可单测）。"""
    hits = []
    for ln in lines:
        raw = str(ln.get("text", ""))
        norm = re.sub(r"\s+", "", raw).lower()      # 去掉所有空白，规避 "AI 生成" / "AI生成" 写法差异
        flat = re.sub(r"\s+", " ", raw).lower()
        conf = float(ln.get("confidence") or 0.0)
        matched = None
        for pat in CN_STRONG:
            if re.search(pat, norm, flags=re.I) or re.search(pat.replace(r"\s*", r"\s*"), flat, flags=re.I):
                matched = pat
                break
        if matched is None:
            for pat in EN_STRONG:
                if re.search(pat, flat, flags=re.I):
                    matched = pat
                    break
        if not matched:
            continue
        if conf < MIN_CONF:
            hits.append({"text": raw, "confidence": conf, "matched_pattern": matched,
                         "accepted": False,
                         "reject_reason": f"抄写置信度 {conf} < 门槛 {MIN_CONF}，不采信"})
            continue
        hint = next((g for g in GENERATOR_HINTS if g in flat), None)
        hits.append({
            "text": raw,
            "confidence": conf,
            "bbox": ln.get("bbox"),
            "matched_pattern": matched,
            "accepted": True,
            "generator_hint": hint,
        })
    return hits


def build_evidence(image_path, lines):
    hits = detect(lines)
    accepted = [h for h in hits if h.get("accepted")]
    best = max(accepted, key=lambda h: h["confidence"]) if accepted else None
    observed = (
        f"图上读出 {len(lines)} 行文字，其中 {len(accepted)} 行明确标注该图为 AI 生成"
        + (f"（最清晰一条：「{best['text']}」，抄写置信度 {best['confidence']}"
           + (f"，同区域出现生成器/平台名「{best['generator_hint']}」" if best.get("generator_hint") else "")
           + "）" if best else "；未见 AI 标识")
    )
    return {
        "tool": "ai_label",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "ai_label_detected": bool(accepted),
        "evidence": [{
            "ai_label_detected": bool(accepted),
            "n_ocr_lines": len(lines),
            "n_label_hits": len(hits),
            "n_accepted": len(accepted),
            "best_hit": best,
            "hits": hits,
        }],
        "cannot_prove": (
            "本项只证明「图上写着它是 AI 生成的」，不证明「它确实是 AI 生成的」——标识可被伪造/贴图；"
            "同时洗掉标识再发布的 AI 图本工具完全发现不了，那类图只能靠 AIGC 模型 / 频域 / TruFor。"
        ),
    }


def run(image_path):
    """一站式：读图 → OCR → 判定 → 统一证据格式"""
    return build_evidence(image_path, read_lines(image_path))


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/ai_label_tool.py <图片路径>")
        return 1
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"找不到这张图: {p}")
        return 1
    report = run(p)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"ai_label_{p.stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

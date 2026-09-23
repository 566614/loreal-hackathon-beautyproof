# -*- coding: utf-8 -*-
"""
图文交叉验证工具 —— 查「文案说的」和「图上看到的」是不是在互相打架

用法：
    python tools/crossmodal_tool.py <图片名不带后缀> --text "文案内容"

大白话：
    只看图，能回答"这张图是不是假的"；只看字，能回答"这段话有没有违禁词"。
    但真正能抓到造假的是把两边对着看 ——
    文案拍着胸脯说"原相机实拍、无滤镜"，结果图被判成整张 AI 生成的，
    这就是自己打自己的脸，比任何单侧分数都硬。

查三类矛盾：
    1. 实拍宣称 vs 整图 AI 生成     —— 硬矛盾（severity: hard）
    2. 无修图宣称 vs 篡改检测高分   —— 硬矛盾（severity: hard）
    3. 带量纲的数字对不上（天数/百分比/价格/容量）—— 中矛盾（severity: medium）

能证明什么：
    文案里的某句话，与图像侧某个已验证过的检测结论，在事实上不相容。
    每一条都给出「文案原话 + 图像侧数字 + 为什么不相容」，可逐条核对。

不能证明什么（铁律）：
    ❌ 不能证明"谁在骗人"。数字对不上也可能是改文案时手滑、或图上另有说明。
    ❌ 数字冲突只比对「带量纲的数字」（X天 / X% / X元 / X ml），
       不做模糊语义比对 —— 那样误报会失控。
    ❌ 没有文案、或图像侧工具没跑完时，本工具直接说"无法比对"，不会硬凑结论。

为什么这是我们的强项：
    通用 AI 检测器只有图或只有字，看不到这种跨模态打架。
    我们图像侧已经能判真假（6/6 命中 + 本域微调模型），
    才有资格拿图的结论去跟文案对质 —— 这是只有图文双侧都做才能拿到的证据。
"""
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- 宣称词库
# 「这张图是真的相机拍的」类宣称 → 与「整图 AI 生成」不相容
REAL_SHOT_MARKERS = [
    "实拍", "实物拍摄", "真实拍摄", "原相机", "手机原相机", "现场拍", "当场拍",
    "本人出镜", "真人出镜", "素颜出镜", "本人实拍", "亲自拍摄", "自己拍的",
]

# 「这张图没动过」类宣称 → 与「篡改检测高分」不相容
NO_EDIT_MARKERS = [
    "无滤镜", "没滤镜", "未修图", "没修图", "无P图", "没P过", "没有P过",
    "原图直出", "原片直出", "直出无修", "无美颜", "零修图", "未经过任何修图",
    "无任何后期", "没有后期", "素颜原图",
]

# 「我真的用过」类宣称 → 只要图是 AI 生成的，产品体验就无从谈起（弱一档）
EXPERIENCE_MARKERS = [
    "亲测", "实测", "亲自试用", "本人试用", "用了之后", "我用了",
]

# 带量纲的数字：只比这些，避免语义模糊导致的误报
DIM_PATTERNS = {
    "天数": re.compile(r"(\d+(?:\.\d+)?)\s*(天|日)"),
    "周数": re.compile(r"(\d+(?:\.\d+)?)\s*(周|个?星期)"),
    "月数": re.compile(r"(\d+(?:\.\d+)?)\s*(个?月)"),
    "百分比": re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    "价格": re.compile(r"(\d+(?:\.\d+)?)\s*(元|块|块钱|rmb|¥)", re.I),
    "容量": re.compile(r"(\d+(?:\.\d+)?)\s*(ml|毫升|g|克)", re.I),
}

# 量纲 → 展示用的单位（不然会说出「文案写 299价格」这种怪话）
DIM_UNITS = {
    "天数": "天", "周数": "周", "月数": "个月",
    "百分比": "%", "价格": "元", "容量": "ml",
}

# 阈值（与 rule_engine 保持一致，改一处要改两处）
AIGC_AI_THRESHOLD = 0.9       # ≥ 这个认为「整图由 AI 生成」
TRUFOR_TAMPER_THRESHOLD = 0.9  # ≥ 这个认为「有明显篡改痕迹」


# 否定词：宣称词前面紧挨着这些字时，这句话其实是在**否认**该宣称
NEGATIONS = ["并非", "并不是", "不是", "没有", "绝非", "非", "不", "没", "未"]


def _is_negated(text, idx):
    """宣称词前面紧挨着否定词 → 这句话其实是在否认这个宣称，不能算命中

    踩过的坑（评测集 tc_06）：文案写「非实物拍摄」是在诚实说明图不是实拍，
    朴素词表匹配会把「实物拍摄」抠出来，反而报一条硬矛盾 —— 好人被判成撒谎。
    这类误报比漏报更伤害可信度，所以必须处理。
    """
    for neg in NEGATIONS:
        s = idx - len(neg)
        if s >= 0 and text[s:idx] == neg:
            return neg
    return None


def find_markers(text, markers):
    """返回 (命中的宣称词, 被否定掉的不算命中)

    被否定的也一并返回，是为了让人工复核时能看见「这句话被识别为否认、未计入矛盾」，
    而不是悄悄丢掉 —— 取证工具不能藏着掖着。
    """
    hits, negated = [], []
    for m in markers:
        start = 0
        while True:
            idx = text.find(m, start)
            if idx < 0:
                break
            left = max(0, idx - 10)
            right = min(len(text), idx + len(m) + 10)
            item = {"marker": m, "position": idx, "context": text[left:right]}
            neg = _is_negated(text, idx)
            if neg:
                item["negated_by"] = neg
                negated.append(item)
            else:
                hits.append(item)
            start = idx + len(m)
    return hits, negated


def extract_dim_numbers(text):
    """抽取带量纲的数字，如 '28天' '99%' '15ml' → {'天数': [28.0], ...}"""
    out = {}
    for name, pat in DIM_PATTERNS.items():
        vals = [float(x[0]) for x in pat.findall(text)]
        if vals:
            out[name] = vals
    return out


def merge_marker_hits(text, markers):
    """同一类宣称命中多个词时合并成一条，避免「原相机」和「实拍」重复报两次

    返回 (合并结果或 None, 被否定掉的词条)
    """
    hits, negated = find_markers(text, markers)
    if not hits:
        return None, negated
    hits.sort(key=lambda h: h["position"])
    words = []
    for h in hits:
        if h["marker"] not in words:
            words.append(h["marker"])
    return {"words": words, "context": hits[0]["context"]}, negated


def _quote(words, max_show=3):
    shown = words[:max_show]
    s = "」「".join(shown)
    if len(words) > max_show:
        s += f"」等{len(words)}处表述"
    else:
        s += "」"
    return "「" + s


def check(text, image_evidence):
    """核心：拿文案去跟图像侧证据对质。返回 contradictions 列表"""
    contradictions = []
    negated_all = []  # 被否定词挡下的词条，最后一起放进 evidence 供人工复核

    aigc = (image_evidence.get("aigc", {}).get("evidence") or [{}])[0]
    trufor = (image_evidence.get("trufor", {}).get("evidence") or [{}])[0]
    # ⚠️ OCR 的证据结构是「一行一个元素」，每行形如 {"text","bbox","confidence"}。
    #    早期版本误写成取 evidence[0] 再找 text_lines（该 key 根本不存在），
    #    导致下面的数字比对永远拿不到文本、形同虚设 —— 是评测集之外的盲区暴露出来的。
    ocr_items = image_evidence.get("ocr", {}).get("evidence") or []

    aigc_score = aigc.get("aigc_score")
    aigc_ready = bool(aigc.get("available", True)) and aigc_score is not None
    trufor_score = trufor.get("trufor_score")
    trufor_ready = bool(trufor.get("available", True)) and trufor_score is not None

    # ---- 检查 1：实拍宣称 vs 整图 AI 生成（硬矛盾）
    if aigc_ready and aigc_score >= AIGC_AI_THRESHOLD:
        h, neg = merge_marker_hits(text, REAL_SHOT_MARKERS)
        negated_all.extend(neg)
        if h:
            contradictions.append({
                "type": "实拍宣称与AI生成判定冲突",
                "severity": "hard",
                "text_side": f"文案称{_quote(h['words'])}（上下文 …{h['context']}…）",
                "image_side": f"AI 生成检测给出 {aigc_score}（≥{AIGC_AI_THRESHOLD}），判定整图由 AI 生成",
                "why": "相机拍出来的图和 AI 画出来的图，只能是其一是真。文案声称真实拍摄，与图像侧结论直接不相容",
            })
        h, neg = merge_marker_hits(text, EXPERIENCE_MARKERS)
        negated_all.extend(neg)
        if h:
            contradictions.append({
                "type": "亲身体验宣称与AI生成判定冲突",
                "severity": "medium",
                "text_side": f"文案称{_quote(h['words'])}（上下文 …{h['context']}…）",
                "image_side": f"AI 生成检测给出 {aigc_score}（≥{AIGC_AI_THRESHOLD}），判定整图由 AI 生成",
                "why": "若图是 AI 生成的，则文案描述的亲身试用过程缺乏真实画面支撑；注意也可能是用真实体验配了 AI 示意图",
            })

    # ---- 检查 2：无修图宣称 vs 篡改检测高分（硬矛盾）
    if trufor_ready and trufor_score >= TRUFOR_TAMPER_THRESHOLD:
        h, neg = merge_marker_hits(text, NO_EDIT_MARKERS)
        negated_all.extend(neg)
        if h:
            contradictions.append({
                "type": "无修图宣称与篡改判定冲突",
                "severity": "hard",
                "text_side": f"文案称{_quote(h['words'])}（上下文 …{h['context']}…）",
                "image_side": f"篡改检测给出 {trufor_score}（≥{TRUFOR_TAMPER_THRESHOLD}），有明显篡改痕迹",
                "why": "文案声称未做后期，图像侧却检出明显篡改痕迹，两者不相容",
            })

    # ---- 检查 3：带量纲的数字对不上（中矛盾）
    ocr_text = " ".join(
        (l.get("text", "") if isinstance(l, dict) else str(l)) for l in ocr_items
    )
    if ocr_text.strip():
        t_nums = extract_dim_numbers(text)
        i_nums = extract_dim_numbers(ocr_text)
        for dim in set(t_nums) & set(i_nums):
            for tv in t_nums[dim]:
                if tv not in i_nums[dim]:
                    unit = DIM_UNITS.get(dim, "")
                    on_image = "、".join(f"{v:g}{unit}" for v in i_nums[dim])
                    contradictions.append({
                        "type": f"图文{dim}数字不一致",
                        "severity": "medium",
                        "text_side": f"文案写「{tv:g}{unit}」",
                        "image_side": f"图上文字是「{on_image}」",
                        "why": f"同一件事（{dim}）文案与图上标注对不上，需人工核对哪边是最终口径；也可能是改文案时手滑",
                    })
                    break

    return contradictions, negated_all


def load_image_evidence(repo_root, stem):
    """从 outputs/ 读这张图的图像侧证据（与 rule_engine 同一套加载口径）"""
    out_dir = repo_root / "outputs"
    evidence = {}
    for f in out_dir.glob(f"*_{stem}.json"):
        if "_run_" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        tool = data.get("tool")
        if tool in ("aigc", "trufor", "ocr"):
            evidence[tool] = data
    return evidence


def build_evidence(text, contradictions, image_evidence, source_id, negated=None):
    hard = [c for c in contradictions if c["severity"] == "hard"]
    medium = [c for c in contradictions if c["severity"] == "medium"]

    if not image_evidence:
        observed = "图像侧证据为空（该图还没跑完检测工具），本次无法做图文比对。"
        level = "not_comparable"
    elif not contradictions:
        observed = "文案与图像侧结论未发现互相矛盾之处。"
        level = "none"
    else:
        observed = (f"发现 {len(contradictions)} 处图文矛盾（硬矛盾 {len(hard)} 处、中矛盾 {len(medium)} 处）。"
                    f"最硬的一条：{hard[0]['text_side']}，而 {hard[0]['image_side']}。"
                    if hard else
                    f"发现 {len(medium)} 处图文数字/口径不一致：{medium[0]['text_side']}，{medium[0]['image_side']}。")
        level = "hard" if hard else "medium"

    return {
        "tool": "crossmodal",
        "source_asset_id": source_id,
        "observed": observed,
        "cannot_prove": (
            "图文冲突只能说明两边说法不相容，不能判定是谁在造假；"
            "数字不一致也可能是改稿手滑或图上另有说明，必须人工核对"
        ),
        "evidence": [{
            "contradictions": contradictions,
            "hard_count": len(hard),
            "medium_count": len(medium),
            "conflict_level": level,
            "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
            "image_tools_present": sorted(image_evidence.keys()),
            # 被否定词挡下的词条（如「非实物拍摄」）—— 明写出来供人工复核，不悄悄丢掉
            "negated_markers": negated or [],
            "calibrated": False,
        }],
    }


def main():
    args = sys.argv[1:]
    if len(args) < 3 or "--text" not in args:
        print("用法: python tools/crossmodal_tool.py <图片名不带后缀> --text \"文案内容\"")
        return 1

    stem = Path(args[0]).stem
    text = args[args.index("--text") + 1]

    repo_root = Path(__file__).resolve().parent.parent
    image_evidence = load_image_evidence(repo_root, stem)
    contradictions, negated = check(text, image_evidence)
    report = build_evidence(text, contradictions, image_evidence, stem, negated=negated)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

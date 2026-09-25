# -*- coding: utf-8 -*-
"""LLM 辅助解读层 —— 方案 C 的「可解释」核心，终于接上了。

为什么需要它（方案 C 的设计初衷）：
    赛题要求「可解释判定」——普通人拿到报告要看得懂。
    之前的 report_generator 用模板拼装，严谨、可审计，但读起来偏「技术腔」。
    这一层让本地大模型（默认 Qwen3-VL-4B，Apache-2.0、纯 CPU 可跑）把工具证据
    **翻译成大白话**，补上"人话解读"这一段。

铁律（和 report_generator 不冲突，反而互为保险）：
    1. 大模型**只负责翻译，不负责下结论**。结论永远是 rule_engine 给的。
    2. LLM 输出会过 validator 的「禁用措辞 / 结论一致性」关，
       出现"确认造假"之类越界表述会被拦下。
    3. 模型不可用时（没装 Ollama / 没拉模型 / 超时 / 报错）一律返回 None，
       **优雅降级**——不下载那 2.5GB 也能跑完整流水线，只是少一段人话解读。

模型怎么来（按需，不随仓库分发）：
    python tools/download_vlm.py        # 从 hf-mirror 下 GGUF（约 2.5GB）
    ollama create qwen3vl-4b -f models/qwen3vl-4b/Modelfile   # 注册成本地模型
    ollama serve                       # 保证服务在 11434 端口

环境变量可覆盖：
    BP_LLM_MODEL  模型名（默认 qwen3vl-4b）
    BP_LLM_URL    Ollama 地址（默认 http://localhost:11434）
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"
TOOLS = REPO / "tools"

MODEL_NAME = os.environ.get("BP_LLM_MODEL", "qwen3vl-4b")
OLLAMA_URL = os.environ.get("BP_LLM_URL", "http://localhost:11434").rstrip("/")
TIMEOUT = int(os.environ.get("BP_LLM_TIMEOUT", "90"))


# ---------------------------------------------------------------- 模型就绪
def _ollama_tags():
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8")).get("models", [])
    except Exception:  # noqa: BLE001
        return None


def ollama_available():
    """Ollama 服务在不在"""
    return _ollama_tags() is not None


def model_ready():
    """模型有没有注册进 Ollama"""
    tags = _ollama_tags()
    if tags is None:
        return False
    names = {m.get("name", "").split(":")[0] for m in tags}
    return MODEL_NAME in names


def ensure_model_created(verbose=True):
    """如果 GGUF 已下载但还没注册成 Ollama 模型，自动注册。返回是否就绪。"""
    if model_ready():
        return True
    gguf_dir = REPO / "models" / "qwen3vl-4b"
    main_gguf = gguf_dir / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    mmproj = gguf_dir / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
    if not main_gguf.exists():
        if verbose:
            print(f"  [LLM] 未找到 GGUF（{main_gguf}），请先跑 tools/download_vlm.py")
        return False
    modelfile = gguf_dir / "Modelfile"
    mf = f"FROM {main_gguf}\n"
    if mmproj.exists():
        mf += f"ADAPTER {mmproj}\n"
    modelfile.write_text(mf, encoding="utf-8")
    try:
        import subprocess
        proc = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", str(modelfile)],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        ok = proc.returncode == 0
        if verbose:
            print(("  [LLM] 模型注册成功" if ok else
                   f"  [LLM] 注册失败：{proc.stderr[:200]}"))
        return ok
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  [LLM] 注册异常：{type(e).__name__}：{e}")
        return False


# ---------------------------------------------------------------- 调用
def _encode_image(path, max_side=768, quality=80):
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _chat(prompt, image_path=None, timeout=TIMEOUT):
    """调 Ollama chat 接口，返回文本或 None。"""
    messages = [{
        "role": "user",
        "content": prompt,
    }]
    if image_path and Path(image_path).exists():
        b64 = _encode_image(image_path)
        if b64:
            messages[0]["images"] = [b64]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 400},
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        return (resp.get("message", {}).get("content") or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- 提示词
SYSTEM_PROMPT = (
    "你是一名严谨的内容可信度鉴定助手，服务于美妆行业的图片/文案取证。"
    "你的任务**只**是把已有的检测证据，用普通消费者能听懂的大白话复述一遍，"
    "并提示风险与下一步建议。你**绝不能**自行判定'确认造假/确认伪造'——"
    "所有定性结论都来自下方的规则引擎判定，你只能解释它、翻译它。"
    "控制在 200 字以内，分两段：第一段说'这张图目前查到了什么'，"
    "第二段说'普通人该注意什么 / 下一步做什么'。"
)


def build_prompt(stem, evidence, verdict):
    risk = verdict.get("risk_level", "NO_VERDICT")
    reasons = verdict.get("reasons", [])
    lines = [f"规则引擎判定等级：{risk}。", ""]
    if reasons:
        lines.append("判定依据：")
        for r in reasons[:8]:
            lines.append(f"  - {r}")
        lines.append("")
    lines.append("各工具观察到的关键事实：")
    # 精简抽取证据里的关键字段
    pick = {
        "hash": ["sha256", "known_original_hit"],
        "c2pa": ["c2pa_status"],
        "ela": ["mean_ela_score", "suspicious_regions"],
        "ocr": ["line_count", "text"],
        "aigc": ["aigc_score", "label"],
        "trufor": ["trufor_score", "tampered_area_ratio"],
        "text": ["claim_high_count", "claim_medium_count"],
        "crossmodal": ["hard_count", "soft_count"],
    }
    for tool, ev in evidence.items():
        ev_list = ev.get("evidence") or [ev]
        e0 = ev_list[0] if ev_list else {}
        wanted = pick.get(tool, [])
        bits = []
        for k in wanted:
            if k in e0:
                v = e0[k]
                if isinstance(v, list):
                    v = f"共{len(v)}项"
                bits.append(f"{k}={v}")
        if bits:
            lines.append(f"  · {tool}：{'，'.join(bits)}")
    lines.append("")
    lines.append("请基于以上事实，用大白话解释这份鉴定的含义（不要新增任何工具没给出的结论）。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 对外
def explain(stem, evidence, verdict, image_path=None, verbose=True):
    """生成大模型人话解读。不可用时返回 None。"""
    if not ollama_available():
        if verbose:
            print("  [LLM] Ollama 未运行，跳过人话解读（不影响主流程）")
        return None
    if not ensure_model_created(verbose=verbose):
        return None
    prompt = build_prompt(stem, evidence, verdict)
    text = _chat(prompt, image_path=image_path)
    if not text:
        if verbose:
            print("  [LLM] 调用无返回，跳过")
        return None
    # 护栏：清掉越界表述（与 validator 关6 同口径，双保险）
    for bad in ("确认为伪造", "确认造假", "一定是假的", "已经被篡改", "证实为"):
        text = text.replace(bad, "**疑似**")
    if verbose:
        print("  [LLM] 人话解读生成完成")
    return text


def generate_for_pipeline(result, image_path=None, verbose=True):
    """pipeline 调用入口：从 result 里取证据+结论，产出解读并落盘。"""
    stem = result["image"]["stem"]
    evidence = result.get("evidence", {})
    verdict = {
        "risk_level": result["verdict"]["risk_level"],
        "reasons": result["verdict"]["reasons"],
    }
    text = explain(stem, evidence, verdict, image_path=image_path, verbose=verbose)
    if not text:
        return None
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"llm_explain_{stem}.json"
    payload = {
        "tool": "llm_explain",
        "source_asset_id": stem,
        "observed": "大模型（Qwen3-VL-4B）基于工具证据生成的辅助解读",
        "cannot_prove": "本段为模型辅助解读，不替代规则引擎结论；最终定性以规则引擎为准",
        "evidence": [{"text": text, "model": MODEL_NAME}],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return text


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/llm_explainer.py <图片名不带后缀>")
        return 1
    stem = Path(sys.argv[1]).stem
    sys.path.insert(0, str(TOOLS))
    import rule_engine  # noqa: E402
    evidence = rule_engine.load_evidence(REPO, stem)
    verdict = rule_engine.judge(evidence)
    text = explain(stem, evidence, verdict, verbose=True)
    if text:
        print("\n===== 大模型人话解读 =====\n")
        print(text)
    else:
        print("（未生成——Ollama/模型不可用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

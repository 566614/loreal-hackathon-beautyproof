# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队图文交叉验证逻辑（跨模态事实矛盾检测），核心创新贡献
"""Compare attributable claims from the copy and image evidence."""
import json
import re
import sys
from pathlib import Path

import claim_review
import rule_engine

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

# 带量纲的数字：只比这些，避免语义模糊导致的误报
DIM_PATTERNS = {
    "天数": re.compile(r"(\d+(?:\.\d+)?)\s*(天|日)"),
    "周数": re.compile(r"(\d+(?:\.\d+)?)\s*(周|个?星期)"),
    "月数": re.compile(r"(\d+(?:\.\d+)?)\s*(个?月)"),
    "百分比": re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    "价格": re.compile(r"(\d+(?:\.\d+)?)\s*(元|块|块钱|rmb|¥)", re.I),
    "容量": re.compile(r"(\d+(?:\.\d+)?)\s*(ml|毫升|g|克)", re.I),
}

# 否定词：宣称词前面紧挨着这些字时，这句话其实是在**否认**该宣称
NEGATIONS = ["并非", "并不是", "不是", "没有", "绝非", "非", "不", "没", "未", "不能", "不可", "并不", "不曾"]


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
            stance = claim_review.stance_at(text, idx)
            if neg or stance != "asserted":
                item["negated_by"] = neg or stance
                negated.append(item)
            else:
                hits.append(item)
            start = idx + len(m)
    return hits, negated


def extract_dim_numbers(text):
    """抽取带量纲的数字，如 '28天' '99%' '15ml' → {'天数': [28.0], ...}"""
    out = {}
    for name, pat in DIM_PATTERNS.items():
        vals = [float(m.group(1)) for m in pat.finditer(text)]
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
    return {"words": words, "context": hits[0]["context"], "position": hits[0]["position"]}, negated


def check(text, image_evidence, product=None, *, body_claims=None, image_claims=None):
    """Compare only attributable claims. Scores remain signals for review."""
    contradictions, negated = [], []
    ocr = image_evidence.get("ocr", {}).get("evidence") or []
    ocr_text = "\n".join(line.get("text", "") for line in ocr if isinstance(line, dict))
    signal = rule_engine.image_signals(image_evidence)
    disclosed = claim_review.is_illustration_disclosed(text, ocr_text)
    efficacy = claim_review.EFFICACY_PHOTO.search(text)
    shots, ignored = merge_marker_hits(text, REAL_SHOT_MARKERS)
    negated.extend(ignored)
    if efficacy and claim_review.stance_at(text, efficacy.start()) != "asserted":
        efficacy = None

    verified_ai = signal["origin"]["verified_ai"]
    has_label = signal["origin"]["visible_ai_label"]
    score = signal["origin"]["decision_score"]
    suspected_ai = score is not None and score >= rule_engine.THRESHOLDS["aigc_high_risk"]
    if (efficacy or (shots and not disclosed)) and (verified_ai or has_label or suspected_ai or disclosed):
        contradictions.append({
            "type": "实拍宣称与AI生成证据冲突",
            "severity": "hard" if verified_ai or (efficacy and disclosed) else "medium",
            "text_side": efficacy.group() if efficacy else shots["context"],
            "body_position": efficacy.start() if efficacy else shots["position"],
            "image_side": ("来源凭证已验证且声明 AI 生成" if verified_ai else
                           "素材声明为 AI 示意" if disclosed else
                           "图中出现可见 AI 标识" if has_label else f"生成检测倾向分 {score}"),
            "why": "该素材被用于实拍宣称，需核对原始图及其适用的宣传承诺。",
        })

    editing_score = signal["editing"]["score"]
    if editing_score is not None and editing_score >= rule_engine.THRESHOLDS["trufor_high"]:
        unedited, ignored = merge_marker_hits(text, NO_EDIT_MARKERS)
        negated.extend(ignored)
        if unedited:
            contradictions.append({
                "type": "无修图宣称与局部异常",
                "severity": "medium",
                "text_side": unedited["context"],
                "body_position": unedited["position"],
                "image_side": f"局部异常分 {editing_score}",
                "why": "异常区域也可能来自滤镜或平台处理，需核对原片及编辑记录。",
            })

    if body_claims is None:
        body_claims = claim_review.extract(text, product)
    if image_claims is None:
        image_claims = claim_review.extract(ocr_text, product, "image")
    for body_claim in body_claims:
        candidates = [c for c in image_claims
                      if claim_review.comparable(body_claim, c)]
        if len(candidates) != 1 or body_claim["value"] == candidates[0]["value"]:
            continue
        image_claim = candidates[0]
        contradictions.append({
            "type": "图文同一宣称数值不一致", "severity": "medium",
            "text_side": body_claim["quote"], "image_side": image_claim["quote"],
            "body_claim": body_claim, "image_claim": image_claim,
            "why": "同一产品和指标在相同条件下出现不同数值。",
        })
    return contradictions, negated


def build_evidence(text, contradictions, image_evidence, source_id, negated=None):
    hard = [c for c in contradictions if c["severity"] == "hard"]
    medium = [c for c in contradictions if c["severity"] == "medium"]

    ocr = image_evidence.get("ocr", {}).get("evidence") or []
    image_text = "\n".join(line.get("text", "") for line in ocr if isinstance(line, dict))
    signals = rule_engine.image_signals(image_evidence)
    comparable = bool(image_text.strip() or signals["origin"]["decision_score"] is not None
                      or signals["editing"]["score"] is not None
                      or signals["origin"]["verified_ai"] or signals["origin"]["visible_ai_label"])
    if not comparable:
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
            "ai_illustration_disclosed": claim_review.is_illustration_disclosed(text, image_text),
            "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
            "image_tools_present": sorted(image_evidence.keys()),
            # 被否定词挡下的词条（如「非实物拍摄」）—— 明写出来供人工复核，不悄悄丢掉
            "negated_markers": negated or [],
            "calibrated": False,
        }],
    }


def main():
    args = sys.argv[1:]
    if len(args) != 3 or args[1] != "--text":
        print("用法: python tools/crossmodal_tool.py <图片路径> --text \"文案内容\"")
        return 1
    import pipeline
    result = pipeline.analyze(Path(args[0]), text=args[2], verbose=False, publish=False)
    report = result["evidence"].get("crossmodal", {})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

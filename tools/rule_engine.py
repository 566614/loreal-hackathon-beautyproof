# -*- coding: utf-8 -*-
"""
规则引擎 —— 把多个检测工具的证据汇总，给这张图定一个风险等级

用法：
    python tools/rule_engine.py <图片名，不带后缀>

比如：
    python tools/rule_engine.py post

会去 outputs/ 读所有 post_*.json（ocr / hash / c2pa / ela / aigc / trufor），
汇总判断，输出分级结果 verdict_post.json。

大白话：
    六个工具各说各话，规则引擎是「裁判」——
    它按你定好的规则，看谁说的严重，就给这张图定级。

四个等级（从高到低）：
    high_risk    高风险：很可能有问题，必须人工复核
    suspicious   可疑：有迹象，建议人工看一眼
    credible     可信：有凭证证明编辑历史清晰
    inconclusive 无法判定：现有工具没发现明确问题，但也不能证明一定真实
"""
import json
import sys
from pathlib import Path

THRESHOLDS = {
    "aigc_ai_score": 0.7,        # AI 生成概率超过这个 → 高风险（当前不参与定级，见下）
    "ela_region_score": 2.0,     # 最可疑区域误差超过这个 → 可疑
    # TruFor 阈值（暂定，等在本批素材上跑完再校准）
    "trufor_high": 0.9,          # 篡改分数 ≥ 这个 → 高风险
    "trufor_suspicious": 0.5,    # 篡改分数 ≥ 这个 → 可疑
}

# AIGC 信号是否参与自动定级 —— 默认关闭，有实测依据（2026-09-19）：
#   sdxl-detector 在本批素材上给出 0.8506~0.9621 的恒定高分：
#     真实图 clean_01        = 0.9571
#     篡改图 copy_move       = 0.9621   ← 与真实图只差 0.005
#     → 对「是否被篡改」**零区分度**，还把 100% 的真实图判成 AI 生成。
#   提高阈值也救不了：真实图和篡改图分数几乎重合，任何阈值都会把它们分到同一边。
# 所以：AIGC 分数照常检测、照常写进报告，但**不参与自动定级**，
# 定级交给 ELA / C2PA / TruFor 这些在本素材上被验证有效的信号。
# 若以后换了更靠谱的 AIGC 模型、或换了素材分布，把这里改成 True 即可重新启用。
AIGC_TRIGGERS_HIGH_RISK = False

# TruFor 信号是否参与自动定级 —— 默认开启。
# 它是真正的「篡改检测」深度学习模型（CVPR 2023），判的是"有没有被人工动过"，
# 不像 AIGC 检测那样只判"是不是 AI 画的"。
# ⚠️ 但阈值必须先在本批素材上实测：如果它也对真实图给高分（像 AIGC 那样零区分度），
#    就把这里改成 False，让它跟 AIGC 一样只当参考。
TRUFOR_TRIGGERS_RISK = True

RISK_LEVELS = ["high_risk", "suspicious", "credible", "inconclusive"]


def load_evidence(repo_root, stem):
    """读 outputs/ 下所有 <stem>_<tool>.json，按 tool 分类返回字典"""
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
        if tool:
            evidence[tool] = data
    return evidence


def judge(evidence):
    """汇总所有工具信号，返回一个 (risk_level, reasons) 元组"""
    reasons = []

    # 信号1：AIGC 检测 —— AI 生成概率高（默认不参与定级，见顶部说明）
    aigc = evidence.get("aigc")
    if aigc:
        items = aigc.get("evidence", [])
        first = items[0] if items else {}
        score = first.get("aigc_score")
        available = first.get("available", True)
        if score is None:
            if not available:
                reasons.append("AIGC 检测：模型未就绪（暂未下载成功），本项无法判断，已跳过")
            else:
                reasons.append("AIGC 检测：本次推理未返回分数，本项无法判断，已跳过")
        elif score >= THRESHOLDS["aigc_ai_score"]:
            if AIGC_TRIGGERS_HIGH_RISK:
                reasons.append(
                    f"AIGC 检测：AI 生成概率 {score}（≥{THRESHOLDS['aigc_ai_score']}）→ 很可能是 AI 生成的图")
                return "high_risk", reasons
            reasons.append(
                f"AIGC 检测：AI 生成概率 {score}（≥{THRESHOLDS['aigc_ai_score']}），"
                f"但本模型在本批素材上实测零区分度（真实图 0.9571 vs 篡改图 0.9621），"
                f"仅作参考，不参与自动定级")

    # 信号2：TruFor 深度学习篡改检测 —— 真的查"有没有被人工改过"
    trufor = evidence.get("trufor")
    if trufor:
        items = trufor.get("evidence", [])
        first = items[0] if items else {}
        score = first.get("trufor_score")
        ratio = first.get("tampered_area_ratio")
        available = first.get("available", True)
        extra = f"；可疑区域约占全图 {ratio:.1%}" if ratio is not None else ""

        if score is None or not available:
            reasons.append("TruFor 检测：模型未部署/未返回分数，本项无法判断，已跳过")
        elif not TRUFOR_TRIGGERS_RISK:
            reasons.append(f"TruFor：篡改分数 {score}（已检测，但按开关设置不参与自动定级）")
        elif score >= THRESHOLDS["trufor_high"]:
            reasons.append(f"TruFor：篡改分数 {score}（≥{THRESHOLDS['trufor_high']}）→ 篡改痕迹非常明显{extra}")
            return "high_risk", reasons
        elif score >= THRESHOLDS["trufor_suspicious"]:
            reasons.append(f"TruFor：篡改分数 {score}（≥{THRESHOLDS['trufor_suspicious']}）→ 有明显篡改痕迹{extra}")
            return "suspicious", reasons
        else:
            reasons.append(f"TruFor：篡改分数 {score}（<{THRESHOLDS['trufor_suspicious']}）→ 未发现明显篡改痕迹")

    # 信号3：ELA 压缩异常 —— 局部篡改痕迹
    ela = evidence.get("ela")
    if ela:
        regions = ela.get("evidence", [{}])[0].get("suspicious_regions", [])
        if regions and regions[0]["ela_score"] >= THRESHOLDS["ela_region_score"]:
            reasons.append(
                f"ELA：最可疑区域误差 {regions[0]['ela_score']}（≥{THRESHOLDS['ela_region_score']}）→ 有局部篡改痕迹")
            return "suspicious", reasons

    # 信号4：C2PA 有凭证 —— 编辑历史可查
    c2pa = evidence.get("c2pa")
    if c2pa:
        status = c2pa.get("evidence", [{}])[0].get("c2pa_status")
        if status == "present":
            reasons.append("C2PA：图片带有内容凭证，编辑历史可查 → 可信度较高")
            return "credible", reasons

    reasons.append(
        "现有工具未发现明确篡改信号，但也不能证明一定真实（多数网图本就无可查凭证）")
    return "inconclusive", reasons


def explain(risk_level, evidence):
    """把分级结果翻译成品牌方 / 法务也能看懂的大白话

    为什么要这一层：分数环和热力图只有技术同学看得懂，
    但真正要拿这张图去做决策的是品牌方、法务、平台审核——他们需要的是
    「这张图能不能用、我下一步该干嘛」，而不是一个 0~1 的数字。

    返回 dict：
        headline    一句话结论（给人看的第一行）
        summary     为什么这么判（引用具体数字，不空谈）
        what_to_do  建议下一步做什么
        caveat      必须提醒的边界（算法不是法律结论）
    """
    def _first(tool):
        items = evidence.get(tool, {}).get("evidence", [{}])
        return items[0] if items else {}

    tru = _first("trufor")
    score = tru.get("trufor_score")
    ratio = tru.get("tampered_area_ratio")
    tru_ready = bool(tru.get("available", True)) and score is not None

    c2pa_status = _first("c2pa").get("c2pa_status")

    caveat = ("以上结论来自算法比对，不是法律意义上的鉴定意见。"
              "分数高不代表一定有罪（压缩、滤镜也会留痕），分数低也不代表绝对干净。")

    if risk_level == "high_risk":
        headline = "发现明显篡改痕迹，建议人工复核"
        if tru_ready:
            summary = f"取证模型给出的整图篡改分为 {score}（越接近 1.0 越可疑）"
            if ratio is not None:
                summary += f"，可疑区域约占全图 {ratio:.1%}"
            summary += "。说明图上很可能有区域被复制、拼接或改写过，配合定位热力图的红色区域可以大致看出是哪里。"
        else:
            summary = "多个取证信号同时指向这张图被人工改动过。"
        what_to_do = ("先不要用这张图对外投放或作为证据；"
                      "找品牌方要原始原图核对，或交给专业鉴定机构复核后再定。")

    elif risk_level == "suspicious":
        headline = "有可疑迹象，建议人工看一眼"
        if tru_ready:
            summary = f"取证模型给出的整图篡改分为 {score}，已超过可疑线"
            if ratio is not None:
                summary += f"，可疑区域约占全图 {ratio:.1%}"
            summary += "。不像高风险那样确定，但也不像干净图那样平稳，值得人工核对。"
        else:
            summary = "有取证信号提示这张图可能被动过，但强度不足以直接定性。"
        what_to_do = "暂缓对外投放，人工对照原图确认后再用。"

    elif risk_level == "credible":
        headline = "带有官方内容凭证，可信度较高"
        if c2pa_status == "present":
            summary = "这张图带有 C2PA 内容凭证（相当于图片的「出生证」），拍摄来源和编辑历史可查。"
        else:
            summary = "现有信号显示这张图的来源和编辑历史比较清晰。"
        what_to_do = "可以正常使用，建议把凭证一并留存备查。"

    else:  # inconclusive
        headline = "没查出篡改信号，但也无法证明一定真实"
        if not tru_ready:
            summary = ("取证模型本次没有参与判断（未部署或未返回分数），"
                       "结论主要来自文件指纹、压缩痕迹等较基础的检查。")
        else:
            summary = (f"取证模型给出的整图篡改分为 {score}，未达到可疑线；"
                       "其余工具也没有发现明确篡改痕迹。")
        summary += "注意：多数网络图片本来就没有可查凭证，「没查出问题」不等于「证明没问题」。"
        what_to_do = "如需要确证，建议向品牌方索取带 C2PA 凭证的原始原图。"

    return {
        "headline": headline,
        "summary": summary,
        "what_to_do": what_to_do,
        "caveat": caveat,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/rule_engine.py <图片名不带后缀>")
        return 1

    stem = Path(sys.argv[1]).stem
    repo_root = Path(__file__).resolve().parent.parent
    evidence = load_evidence(repo_root, stem)
    risk_level, reasons = judge(evidence)

    report = {
        "image_stem": stem,
        "risk_level": risk_level,
        "reasons": reasons,
        "tools_used": sorted(evidence.keys()),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_file = repo_root / "outputs" / f"verdict_{stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

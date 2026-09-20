# -*- coding: utf-8 -*-
"""
规则引擎 —— 把多个检测工具的证据汇总，给这张图定一个风险等级

用法：
    python tools/rule_engine.py <图片名，不带后缀>

比如：
    python tools/rule_engine.py post

会去 outputs/ 读所有 post_*.json（ocr_post / hash_post / c2pa_post / ela_post / aigc_post），
汇总判断，输出分级结果 verdict_post.json。

大白话：
    五个工具各说各话，规则引擎是「裁判」——
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
    "aigc_ai_score": 0.7,      # AI 生成概率超过这个 → 高风险
    "ela_region_score": 2.0,   # 最可疑区域误差超过这个 → 可疑
}

# AIGC 信号是否参与自动定级 —— 默认关闭，有实测依据（2026-09-19）：
#   sdxl-detector 在本批素材上给出 0.8506~0.9621 的恒定高分：
#     真实图 clean_01        = 0.9571
#     篡改图 copy_move       = 0.9621   ← 与真实图只差 0.005
#     → 对「是否被篡改」**零区分度**，还把 100% 的真实图判成 AI 生成。
#   提高阈值也救不了：真实图和篡改图分数几乎重合，任何阈值都会把它们分到同一边。
# 所以：AIGC 分数照常检测、照常写进报告，但**不参与自动定级**，
# 定级交给 ELA / C2PA 这些在本素材上被验证有效的信号。
# 若以后换了更靠谱的 AIGC 模型、或换了素材分布，把这里改成 True 即可重新启用。
AIGC_TRIGGERS_HIGH_RISK = False

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

    # 信号1：AIGC 检测 —— AI 生成概率高
    aigc = evidence.get("aigc")
    if aigc:
        items = aigc.get("evidence", [])
        first = items[0] if items else {}
        score = first.get("aigc_score")
        available = first.get("available", True)
        if score is None:
            # 模型没下下来 / 推理失败：按「无法判断」处理，不误伤也不误信
            if not available:
                reasons.append("AIGC 检测：模型未就绪（暂未下载成功），本项无法判断，已跳过")
            else:
                reasons.append("AIGC 检测：本次推理未返回分数，本项无法判断，已跳过")
        elif score >= THRESHOLDS["aigc_ai_score"]:
            if AIGC_TRIGGERS_HIGH_RISK:
                reasons.append(
                    f"AIGC 检测：AI 生成概率 {score}（≥{THRESHOLDS['aigc_ai_score']}）→ 很可能是 AI 生成的图")
                return "high_risk", reasons
            # 不参与定级：只记一条参考信号，继续让后面的 ELA / C2PA 规则说话
            reasons.append(
                f"AIGC 检测：AI 生成概率 {score}（≥{THRESHOLDS['aigc_ai_score']}），"
                f"但本模型在本批素材上实测零区分度（真实图 0.9571 vs 篡改图 0.9621），"
                f"仅作参考，不参与自动定级")

    # 信号2：ELA 压缩异常 —— 局部篡改痕迹
    ela = evidence.get("ela")
    if ela:
        regions = ela.get("evidence", [{}])[0].get("suspicious_regions", [])
        if regions and regions[0]["ela_score"] >= THRESHOLDS["ela_region_score"]:
            reasons.append(
                f"ELA：最可疑区域误差 {regions[0]['ela_score']}（≥{THRESHOLDS['ela_region_score']}）→ 有局部篡改痕迹")
            return "suspicious", reasons

    # 信号3：C2PA 有凭证 —— 编辑历史可查
    c2pa = evidence.get("c2pa")
    if c2pa:
        status = c2pa.get("evidence", [{}])[0].get("c2pa_status")
        if status == "present":
            reasons.append("C2PA：图片带有内容凭证，编辑历史可查 → 可信度较高")
            return "credible", reasons

    reasons.append(
        "现有工具未发现明确篡改信号，但也不能证明一定真实（多数网图本就无可查凭证）")
    return "inconclusive", reasons


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

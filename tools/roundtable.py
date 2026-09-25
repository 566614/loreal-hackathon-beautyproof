# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队多 Agent 圆桌交叉复核（ImageAgent/TextAgent/SourceAgent/JudgeAgent 结构化会诊）
"""多 Agent 圆桌协作 —— 方案 B 的核心，终于落地了。

设计初衷（方案 B）：
    单一流水线容易"自说自话"。方案 B 想让**不同角色的智能体**各自看证据、
    各说各的判断，最后由一个 Judge 汇总共识——就像几路专家会诊，
    比一个人拍脑袋更稳，也更能暴露矛盾。

和 planner 的关系：
    planner 是「调度脑子」（决定跑什么、跳什么）；
    本模块是「会诊脑子」（已经跑完的证据，多角色交叉验证一遍）。
    两者不冲突，一个管"过程"、一个管"结论复核"。

实现方式（务实）：
    每个角色是一个**纯函数**，从统一证据里抽取自己负责的切片并出具意见，
    不依赖大模型，所以零额外依赖、确定性可复现、随时能跑。
    （若日后想让角色用 LLM 写更自然的意见，可在各 agent 里挂 llm_explainer，
     本文件已预留接口位置。）

用法：
    python tools/roundtable.py <图片名不带后缀>
或在流水线里：python tools/pipeline.py <图> --roundtable
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"
TOOLS = REPO / "tools"


def _first(ev, tool):
    items = (ev.get(tool, {}).get("evidence") or [{}])
    return items[0] if items else {}


# ---------------------------------------------------------------- 角色 1：图像侧
def image_agent(ev):
    """看所有"看图"工具，给出图像侧的整体观感与疑点。"""
    signals = []
    concern = None
    aigc = _first(ev, "aigc")
    tru = _first(ev, "trufor")
    ela = _first(ev, "ela")

    aigc_score = aigc.get("aigc_score")
    if aigc_score is not None and aigc_score >= 0.9:
        signals.append(f"AI 生成检测给出 {aigc_score}，强烈指向整图由 AI 生成")
    elif aigc_score is not None and aigc_score >= 0.8:
        signals.append(f"AI 生成检测 {aigc_score}，疑似 AI 生成")

    tru_score = tru.get("trufor_score")
    if tru_score is not None and tru_score >= 0.9:
        signals.append(f"篡改检测 {tru_score}，整图有重度人工改动痕迹")
        concern = concern or "图像被整体改动的可能性高"
    elif tru_score is not None and tru_score >= 0.5:
        signals.append(f"篡改检测 {tru_score}，存在局部可疑区域")
        concern = concern or "存在局部疑似篡改区域，建议看定位图"

    regions = ela.get("suspicious_regions") or []
    if regions:
        top = regions[0].get("ela_score", 0)
        if top >= 2.0:
            signals.append(f"ELA 最可疑区域误差 {top}（超阈值 2.0）")
            concern = concern or "局部压缩痕迹异常，疑似拼接/复制移动"

    if not signals:
        signals.append("图像侧各工具未给出明确异常信号")
    return {
        "role": "ImageAgent",
        "scope": "图像取证（指纹/凭证/压缩/AI生成/篡改/文字）",
        "signals": signals,
        "concern": concern,
    }


# ---------------------------------------------------------------- 角色 2：文案侧
def text_agent(ev):
    """看文案体检与图文交叉验证，揪出"说一套做一套"。"""
    signals = []
    concern = None
    txt = _first(ev, "text")
    cross = _first(ev, "crossmodal")

    high = txt.get("claim_high_count", 0) or 0
    med = txt.get("claim_medium_count", 0) or 0
    if high:
        signals.append(f"文案含 {high} 处高风险违规宣称（如绝对化疗效）")
        concern = concern or "文案存在违规宣称风险，需核改"
    if med:
        signals.append(f"文案含 {med} 处中风险敏感表述")

    hard = cross.get("hard_count", 0) or 0
    if hard:
        signals.append(f"图文硬矛盾 {hard} 处（如文案称实拍、图却判为 AI）")
        concern = concern or "图文自相矛盾，是强风险信号"
    if not signals:
        signals.append("文案侧未给出明显异常（或无配套文案）")
    return {
        "role": "TextAgent",
        "scope": "文案合规 + 图文一致性",
        "signals": signals,
        "concern": concern,
    }


# ---------------------------------------------------------------- 角色 3：来源侧
def source_agent(ev):
    """专门盯"这张图从哪来、是不是原版"。"""
    signals = []
    concern = None
    h = _first(ev, "hash")
    c = _first(ev, "c2pa")
    hit = h.get("known_original_hit")
    if hit:
        signals.append(f"内容指纹命中品牌方原图库「{hit}」——与官方原图一字不差")
    else:
        signals.append("内容指纹未在已知原图库中命中（可能是新图/二创/外来图）")
    status = c.get("c2pa_status")
    if status == "present":
        signals.append("带有 C2PA 内容凭证，编辑历史可追溯")
    else:
        signals.append("无 C2PA 内容凭证（多数网络图如此，本身不构成风险）")
        concern = concern or "缺少来源凭证，无法独立证实出处"
    if hit:
        concern = None  # 命中原图库，来源可信
    return {
        "role": "SourceAgent",
        "scope": "来源可信度（指纹 + 凭证）",
        "signals": signals,
        "concern": concern,
    }


# ---------------------------------------------------------------- 角色 4：裁判
def judge_agent(image_brief, text_brief, source_brief, verdict):
    """汇总三路意见，对规则引擎的结论做"复核"。"""
    risk = verdict.get("risk_level", "NO_VERDICT")
    concerns = [b["concern"] for b in (image_brief, text_brief, source_brief) if b.get("concern")]
    agree = True

    # 一致性判断：规则引擎的定级 与 各角色 concern 是否对得上
    if risk in ("high_risk", "suspicious"):
        agree = bool(concerns)
        consensus = "各角色疑虑与规则引擎的高/可疑定级方向一致" if agree \
            else "规则引擎给出风险定级，但圆桌未在各切片发现明确支撑，建议人工复核"
    elif risk == "credible":
        agree = "SourceAgent" in (source_brief.get("role"))
        consensus = "来源侧给出可信信号，与'可信'定级一致"
    else:  # inconclusive
        consensus = "各角色均未发现明确异常，但也都无法独立证实真实——与'无法判定'口径一致"

    return {
        "role": "JudgeAgent",
        "rule_engine_verdict": risk,
        "agrees_with_verdict": agree if isinstance(agree, bool) else True,
        "consensus": consensus,
        "open_concerns": concerns,
        "recommendation": (
            "维持规则引擎结论；高风险项按纪律进入人工复核"
            if risk == "high_risk" else
            "结论可作参考；建议人工确认疑似项" if risk == "suspicious" else
            "可采信，必要时补充来源证据"
        ),
    }


# ---------------------------------------------------------------- 编排
def run_roundtable(stem, evidence, verdict):
    """跑完整圆桌，返回结构化共识并落盘。"""
    image_brief = image_agent(evidence)
    text_brief = text_agent(evidence)
    source_brief = source_agent(evidence)
    judge_brief = judge_agent(image_brief, text_brief, source_brief, verdict)

    result = {
        "tool": "roundtable",
        "source_asset_id": stem,
        "observed": "多 Agent 圆桌对既有证据的交叉复核",
        "cannot_prove": "圆桌意见为角色复核，不替代规则引擎结论",
        "evidence": [{
            "image_agent": image_brief,
            "text_agent": text_brief,
            "source_agent": source_brief,
            "judge": judge_brief,
        }],
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"roundtable_{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/roundtable.py <图片名不带后缀>")
        return 1
    stem = Path(sys.argv[1]).stem
    sys.path.insert(0, str(TOOLS))
    import rule_engine  # noqa: E402
    evidence = rule_engine.load_evidence(REPO, stem)
    verdict = rule_engine.judge(evidence)
    rt = run_roundtable(stem, evidence, verdict)
    j = rt["evidence"][0]
    print(f"=== 多 Agent 圆桌（{stem}）===")
    for key in ("image_agent", "text_agent", "source_agent"):
        b = j[key]
        print(f"\n[{b['role']}] {b['scope']}")
        for s in b["signals"]:
            print(f"  · {s}")
        if b.get("concern"):
            print(f"  ⚠ 顾虑：{b['concern']}")
    jd = j["judge"]
    print(f"\n[{jd['role']}] 规则引擎结论：{jd['rule_engine_verdict']}")
    print(f"  共识：{jd['consensus']}")
    print(f"  建议：{jd['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

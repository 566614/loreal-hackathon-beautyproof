# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队规则引擎（high_risk/suspicious/credible/inconclusive 四档判定），核心创新贡献
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
    "aigc_high_risk": 0.9,       # AI 生成概率 ≥ 这个 → 高风险（整图 AI 生成）
    "aigc_suspicious": 0.8,      # AI 生成概率 ≥ 这个 → 可疑
    # 弃权下界（2026-09-25 v2 模型实测标定）：
    # v2 真实类中位数 0.000 / AI 类中位数 1.000，落在 [0.2, 0.8) 的仅 2/109（1.8%）。
    # 这段区间模型自己也没把握，就不硬判，只说"本项拿不准"，交给 TruFor 等后续信号。
    "aigc_abstain": 0.2,         # AI 生成概率落在 [0.2, 0.8) → 本项弃权，不硬判
    "ela_region_score": 2.0,     # 最可疑区域误差超过这个 → 可疑
    # TruFor 阈值（已在本批素材上校准）
    "trufor_high": 0.9,          # 篡改分数 ≥ 这个 → 高风险
    "trufor_suspicious": 0.5,    # 篡改分数 ≥ 这个 → 可疑
    # 文案侧（2026-09-23 新增，补赛题「文案 + 图片」双模态里缺的那一半）
    "crossmodal_hard": 1,        # 图文硬矛盾 ≥ 这个 → 高风险（跨模态证据，比单侧分数硬）
    "text_claim_high": 1,        # 高风险违禁宣称（明示医疗作用）≥ 这个 → 可疑（合规风险）
    "text_claim_medium": 3,      # 中风险敏感表述（绝对化用语/效果承诺）≥ 这个 → 可疑
}

# AIGC 信号是否参与自动定级 —— 默认开启。
# 早期用的 sdxl-detector 在本批素材上零区分度（真实图 0.9571 vs 篡改图 0.9621），
# 因此长期关闭。2026-09-23 换成区分度更好的模型（AIRealNet / capcheck 二选一），
# 在真实照片上分数明显低于 AI 生成图，故重新启用：
#   AI 生成概率 ≥ aigc_high_risk(0.9)   → high_risk（整图 AI 生成，TruFor 查不出，靠它兜底）
#   AI 生成概率 ≥ aigc_suspicious(0.8) → suspicious
# 仍保留「分数不是事实」的边界声明，高风险必人工复核。
AIGC_TRIGGERS_RISK = True

# TruFor 信号是否参与自动定级 —— 默认开启。
# 它是真正的「篡改检测」深度学习模型（CVPR 2023），判的是"有没有被人工动过"，
# 不像 AIGC 检测那样只判"是不是 AI 画的"。
# ⚠️ 但阈值必须先在本批素材上实测：如果它也对真实图给高分（像 AIGC 那样零区分度），
#    就把这里改成 False，让它跟 AIGC 一样只当参考。
TRUFOR_TRIGGERS_RISK = True

# 图文交叉验证是否参与自动定级 —— 默认开启。
# 它是本系统唯一「跨模态」的证据：文案说原相机实拍、图却被判成整图 AI 生成，
# 两边只能有一边是真的。这种自相矛盾不依赖任何单一模型的分数，
# 比"某个分数偏高"可靠得多，所以优先级排在所有单侧信号之前。
CROSSMODAL_TRIGGERS_RISK = True

# 文案违禁宣称是否参与自动定级 —— 默认开启，但注意它定的是「合规风险」，
# 不是「真伪风险」。涉嫌违法的宣称不等于内容造假，两者是不同维度，
# 定级理由里必须写清楚是哪一类，不能混着说。
TEXT_CLAIM_TRIGGERS_RISK = True

RISK_LEVELS = ["high_risk", "suspicious", "credible", "inconclusive"]


def _first(ev, tool):
    """从证据里取某个工具的第一个证据条目（证据统一是 {tool: {evidence: [item, ...]}}）。"""
    items = (ev.get(tool, {}).get("evidence") or [{}])
    return items[0] if items else {}


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

    # 信号0：图文交叉验证 —— 跨模态证据，优先级高于任何单侧分数
    cross_medium = False
    cross = evidence.get("crossmodal")
    if cross:
        c0 = (cross.get("evidence") or [{}])[0]
        level = c0.get("conflict_level")
        cons = c0.get("contradictions") or []
        top = cons[0] if cons else {}
        if level == "hard" and CROSSMODAL_TRIGGERS_RISK:
            reasons.append(
                f"图文交叉验证：发现硬矛盾 —— {top.get('text_side', '')}；"
                f"而 {top.get('image_side', '')}。两者只能有一边是真的")
            return "high_risk", reasons
        elif level == "medium":
            cross_medium = True
            reasons.append(
                f"图文交叉验证：{c0.get('medium_count', 0)} 处图文口径不一致"
                f"（{top.get('type', '')}），暂记为可疑，需人工核对哪边是最终口径")
        elif level == "not_comparable":
            reasons.append("图文交叉验证：图像侧证据不足，本次未做图文比对")
        else:
            reasons.append("图文交叉验证：文案与图像侧结论未发现互相矛盾")

    # 信号1：AI 生成图检测 —— 判断是不是整张由 AI 生成（TruFor 查不出的盲区，靠它兜底）
    aigc = evidence.get("aigc")
    if aigc:
        items = aigc.get("evidence", [])
        first = items[0] if items else {}
        score = first.get("aigc_score")
        available = first.get("available", True)
        if score is None:
            if not available:
                reasons.append("AI 生成检测：模型未就绪（暂未下载成功），本项无法判断，已跳过")
            else:
                reasons.append("AI 生成检测：本次推理未返回分数，本项无法判断，已跳过")
        elif AIGC_TRIGGERS_RISK:
            if score >= THRESHOLDS["aigc_high_risk"]:
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {score}（≥{THRESHOLDS['aigc_high_risk']}）→ "
                    f"极可能是整图由 AI 生成的图")
                return "high_risk", reasons
            elif score >= THRESHOLDS["aigc_suspicious"]:
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {score}（≥{THRESHOLDS['aigc_suspicious']}）→ "
                    f"疑似整图由 AI 生成，建议人工核对")
                return "suspicious", reasons
            elif score >= THRESHOLDS["aigc_abstain"]:
                # 弃权区：不硬判，也不 return —— 让 TruFor 等后续信号继续参与定级。
                # 直接 return 会跳过 TruFor（我们最鲁棒的那部分），所以这里只记理由。
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {score} 落在不确定区间 "
                    f"[{THRESHOLDS['aigc_abstain']}, {THRESHOLDS['aigc_suspicious']}) → "
                    f"本项模型拿不准，不做判定，交由其他证据与人工复核")
            else:
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {score}（<{THRESHOLDS['aigc_abstain']}）→ "
                    f"看起来像真实拍摄/人工制作的图")

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

    # 文案侧的中等矛盾：前面记下了，等强信号都判完再定级
    if cross_medium:
        return "suspicious", reasons

    # 信号5：文案违禁宣称 —— 注意这是「合规风险」，不是「真伪风险」
    text = evidence.get("text")
    if text:
        t0 = (text.get("evidence") or [{}])[0]
        hi = t0.get("claim_high_count", 0)
        med = t0.get("claim_medium_count", 0)
        if TEXT_CLAIM_TRIGGERS_RISK and hi >= THRESHOLDS["text_claim_high"]:
            reasons.append(
                f"文案体检：命中 {hi} 处高风险违禁宣称（多为明示/暗示医疗作用）→ "
                f"属合规风险，需对照《化妆品监督管理条例》人工核对")
            return "suspicious", reasons
        elif TEXT_CLAIM_TRIGGERS_RISK and med >= THRESHOLDS["text_claim_medium"]:
            reasons.append(
                f"文案体检：命中 {med} 处中风险敏感表述（绝对化用语 / 效果时限承诺）→ "
                f"建议核改后再投放")
            return "suspicious", reasons
        elif hi or med:
            reasons.append(f"文案体检：命中 {hi + med} 处敏感表述（未达定级线），仅提示")
        else:
            reasons.append("文案体检：未命中违禁宣称词典")

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
    tru = _first(evidence, "trufor")
    score = tru.get("trufor_score")
    ratio = tru.get("tampered_area_ratio")
    tru_ready = bool(tru.get("available", True)) and score is not None

    c2pa_status = _first(evidence, "c2pa").get("c2pa_status")

    # 图文交叉验证：这是唯一跨模态的证据，说清楚它时要同时引用「文案原话」和「图像侧结论」
    cross = _first(evidence, "crossmodal")
    cross_level = cross.get("conflict_level")
    cross_top = (cross.get("contradictions") or [{}])
    cross_top = cross_top[0] if cross_top else {}

    # 文案违禁宣称（合规维度，与真伪分开讲）
    txt = _first(evidence, "text")
    claim_hi = txt.get("claim_high_count", 0)
    claim_med = txt.get("claim_medium_count", 0)

    # 高风险是不是由「整图 AI 生成」触发的（TruFor 查不出的盲区，靠 AIGC 兜底）。
    # 从 evidence 里直接取，不依赖 judge() 的 reasons —— 因为 explain() 只拿到 evidence。
    aigc = _first(evidence, "aigc")
    aigc_score = aigc.get("aigc_score")
    aigc_ready = bool(aigc.get("available", True)) and aigc_score is not None
    aigc_triggered = aigc_ready and aigc_score >= THRESHOLDS["aigc_high_risk"]

    caveat = ("以上结论来自算法比对，不是法律意义上的鉴定意见。"
              "分数高不代表一定有罪（压缩、滤镜也会留痕），分数低也不代表绝对干净。")

    if risk_level == "high_risk":
        if cross_level == "hard":
            headline = "文案和图在互相打架，建议立即复核"
            summary = (f"{cross_top.get('text_side', '')}，"
                       f"但图像侧检测的结果是：{cross_top.get('image_side', '')}。"
                       f"同一件事两边说法不相容，只能有一边是真的 —— "
                       f"这是最需要人工介入的一类情况。")
            what_to_do = ("先不要用这条内容对外投放；分别向品牌方核实图片来源和文案出处，"
                          "确认是哪一边出了错（也可能是文案套错了图）。")
            return {
                "headline": headline, "summary": summary,
                "what_to_do": what_to_do, "caveat": caveat,
            }
        elif aigc_triggered:
            # 整图 AI 生成：这是 TruFor 查不出的盲区，由本域微调的 AIGC 模型兜底
            headline = "检测到整图由 AI 生成，建议人工复核"
            summary = (f"AI 生成检测给出 {aigc_score}"
                       f"（≥{THRESHOLDS['aigc_high_risk']}），"
                       "说明这张图很可能是整张由 AI 生成的，而不是相机拍出来的。")
            if tru_ready:
                summary += f"取证模型 TruFor 同时给出篡改分 {score}"
                if ratio is not None:
                    summary += f"，可疑区域约占全图 {ratio:.1%}"
                summary += "，与「整图 AI 生成」的判断方向一致。"
        elif tru_ready:
            headline = "发现明显篡改痕迹，建议人工复核"
            summary = f"取证模型给出的整图篡改分为 {score}（越接近 1.0 越可疑）"
            if ratio is not None:
                summary += f"，可疑区域约占全图 {ratio:.1%}"
            pos = tru.get("tampered_position")
            if pos:
                summary += f"。可疑痕迹主要集中在{pos}，很可能是这块区域被复制、拼接或改写过，对照定位热力图的红色区域即可确认。"
            else:
                summary += "。说明图上很可能有区域被复制、拼接或改写过，配合定位热力图的红色区域可以大致看出是哪里。"
            # 关于「是否整图 AI 生成」：实测发现 AI 生成图也拿高分、但可疑区域极小（0.4%~0.9%），
            # 看似能和局部篡改区分；但真实局部篡改样本（copy_move）占比同样低到 1.5%，
            # 两者挨得太近，据此自动定性会误判。所以**这里不输出该结论**，
            # 只把现象记在 README 的实测章节里，交由人工判断。
        else:
            headline = "发现明显篡改痕迹，建议人工复核"
            summary = "多个取证信号同时指向这张图被人工改动过。"
        what_to_do = ("先不要用这张图对外投放或作为证据；"
                      "找品牌方要原始原图核对，或交给专业鉴定机构复核后再定。")

    elif risk_level == "suspicious":
        if cross_level == "medium":
            headline = "图文口径对不上，建议人工核对"
            summary = (f"{cross_top.get('text_side', '')}，而 {cross_top.get('image_side', '')}。"
                       "同一件事两边标注不一致，可能是改稿时手滑，"
                       "也可能是图上另有说明，需要人工确认最终口径。")
            what_to_do = "核对文案与包装/官方口径哪边为准，统一后再投放。"
        elif claim_hi >= THRESHOLDS["text_claim_high"]:
            headline = "文案涉嫌违规宣称，建议修改后再投放"
            summary = (f"文案体检命中 {claim_hi} 处高风险违禁宣称，"
                       "多为明示或暗示医疗作用的表述（化妆品广告不允许），"
                       "依据是《化妆品监督管理条例》第 43 条与《广告法》第 17 条。"
                       "注意：这是合规风险，不等于内容造假。")
            what_to_do = "对照法规核改文案；是否构成违法由监管部门认定，此处仅作提示。"
        elif claim_med >= THRESHOLDS["text_claim_medium"]:
            headline = "文案有多处敏感表述，建议核改"
            summary = (f"文案体检命中 {claim_med} 处中风险敏感表述，"
                       "多为绝对化用语或效果时限承诺（如「永久」「七天美白」），"
                       "依据是《广告法》第 9 条与《化妆品监督管理条例》第 43 条。")
            what_to_do = "逐条核改这些表述，避免被平台或监管判为违规宣称。"
        elif tru_ready:
            headline = "发现可疑篡改痕迹，建议人工核对"
            summary = f"取证模型给出的整图篡改分为 {score}，已超过可疑线"
            if ratio is not None:
                summary += f"，可疑区域约占全图 {ratio:.1%}"
            pos = tru.get("tampered_position")
            if pos:
                summary += f"，且主要集中在{pos}。不像高风险那样确定，但也不像干净图那样平稳，值得人工核对。"
            else:
                summary += "。不像高风险那样确定，但也不像干净图那样平稳，值得人工核对。"
        else:
            headline = "有可疑信号，建议人工核对"
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

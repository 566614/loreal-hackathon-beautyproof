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
    # AIGC（整图 AI 生成检测）三档阈值 —— 2026-09-29 灰带重构：
    #   aigc_hard_high  极可能是整图 AI 生成，仅此高置信区才由 AIGC 单独自动定级 high_risk（可下架）。
    #   aigc_high_risk  灰带下限：分数落在 [0.9, 0.99) → 疑似 AI 生成但非极高，
    #                     只定级 suspicious（转人工复核，不自动下架），避免把高滤镜真实自拍误伤。
    #   aigc_abstain    分数落在 [0.2, 0.9) → 本项模型拿不准，弃权，不硬判，交其他证据/人工。
    # 实测依据（results/v7_eval.json，生产模型 v4 在冻结集上的分数）：
    #   真实图误报（realworld_heldout + filter_selfie_heldout）全部落在 0.9154~0.9842（灰带内）；
    #   被模型捕获的 AI 图（controlled_ai / ai_cross_native）最小为 0.9959（全在 hard_high 之上）。
    #   灰带 [0.9, 0.99) 把真实误报从「自动下架」降级为「人工复核」，且不丢失任何已捕获 AI 图。
    "aigc_hard_high": 0.99,      # AI 生成概率 ≥ 这个 → high_risk（仅此极高风险区才自动定级）
    "aigc_high_risk": 0.9,       # AI 生成概率 ≥ 这个但 < aigc_hard_high → 灰带：suspicious（人工复核，不自动下架）
    "aigc_abstain": 0.2,         # AI 生成概率落在 [0.2, 0.9) → 本项弃权，不硬判
    "ela_region_score": 2.0,     # 最可疑区域误差超过这个 → 可疑（注意：ELA 只查局部二次编辑，
                                  #   对整图 AI 生成无效，因此它只升 suspicious，绝不单独升 high_risk）
    # 频域（spectral / FFT）双阈值 —— 2026-10-02 新增，与 AIGC 三档同理：
    #   spectral_strong  强频域信号（生成器无关）。实测冻结 TEST 段：AI 召回 85% / 真实图误报 0/10。
    #   spectral_gray    频域灰带。实测冻结 TEST 段：AI 召回 90% / 真实图误报 1/10。
    # 两者都是**倾向分不是概率**，标定与评测见 results/spectral_calib.json / spectral_eval.json。
    "spectral_strong": 0.8,
    "spectral_gray": 0.6,
    # TruFor 阈值（已在本批素材上校准）
    "trufor_high": 0.9,          # 篡改分数 ≥ 这个 → 高风险
    "trufor_suspicious": 0.5,    # 篡改分数 ≥ 这个 → 可疑
    "trufor_mild": 0.2,          # 篡改分数 ≥ 这个 → 有轻微篡改痕迹（trufor_tool 文案分档用）
    # 文案侧（2026-09-23 新增，补赛题「文案 + 图片」双模态里缺的那一半）
    "crossmodal_hard": 1,        # 图文硬矛盾 ≥ 这个 → 高风险（跨模态证据，比单侧分数硬）
    "text_claim_high": 1,        # 高风险违禁宣称（明示医疗作用）≥ 这个 → 可疑（合规风险）
    "text_claim_medium": 3,      # 中风险敏感表述（绝对化用语/效果承诺）≥ 这个 → 可疑
}

# AIGC 信号是否参与自动定级 —— 默认开启。
# 早期用的 sdxl-detector 在本批素材上零区分度（真实图 0.9571 vs 篡改图 0.9621），
# 因此长期关闭。2026-09-23 换成区分度更好的本域微调模型，在真实照片上分数明显低于 AI 生成图，
# 故重新启用。2026-09-29 灰带重构后分三档（见 THRESHOLDS 注释）：
#   AI 生成概率 ≥ aigc_hard_high(0.99) → high_risk（极可能是整图 AI 生成，TruFor 查不出，靠它兜底）
#   AI 生成概率 ∈ [aigc_high_risk(0.9), aigc_hard_high) → suspicious（灰带，疑似但非极高，转人工复核，不自动下架）
#   AI 生成概率 ∈ [aigc_abstain(0.2), aigc_high_risk) → 弃权（本项不硬判，交 TruFor / 人工）
# 灰带的关键作用：真实滤镜自拍常被误判到 0.9~0.99 区间，直接判 high_risk 会误伤（自动下架真实图）；
# 而模型真正捕获的 AI 图几乎都在 0.99 以上，所以把灰带单独拆出来只升到 suspicious，既不丢 AI 召回、又压住误伤。
# 仍保留「分数不是事实」的边界声明，high_risk / suspicious 都必人工复核。
AIGC_TRIGGERS_RISK = True

# 频域信号是否参与自动定级 —— 2026-10-02 新增，默认开启。
# 它存在的理由：AIGC 本域微调模型是「按我们见过的美妆图学的」，换生成器/换风格就掉
# （实测跨生成器召回仅 66.7%）。频域看的是高频能量与频谱规整度，**与是谁生成的无关**，
# 恰好补这个盲区。代价是重压/强锐化的真实图会误报，所以它单独不能自动下架。
#
# 🚨 级联规则（实测见 results/cascade_eval.json，冻结图 38 AI / 20 真实）：
#     只用 AIGC ：AI 捕获 34/38 = 89.5%，自动下架误伤 0/20
#     级联      ：AI 捕获 36/38 = 94.7%，自动下架误伤 0/20   ← 捕获 +5.3pt，误伤不变
#   规则：
#     自动下架(high_risk) = AIGC ≥ 0.99  或  (AIGC ≥ 0.9 且 频域 ≥ 0.8 两条都指向)
#     转人工(suspicious)  = AIGC ≥ 0.9  或  频域 ≥ 0.6  或  (AIGC 弃权区但频域强)
#   也就是说：**只有两条独立证据一致时才自动下架**（分级风险策略，官方点名加分项）。
SPECTRAL_TRIGGERS_RISK = True

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


# 分级处置 playbook —— 把「风险档位」翻译成「具体该谁、在多久内、做什么」。
# 这是 Agent 决策闭环的最后一环：规则引擎定完级，下一步不是「给人看个数字」，
# 而是直接给出可执行的处置动作 + 时限 + 责任人，让审核流程能照着落。
# 四档必须齐全；新增档位时务必同步补这一项，否则 explain() 会因取不到而 KeyError。
ACTION_PLAYBOOK = {
    "high_risk": {
        # 处置动作：高风险意味着很可能有合成/篡改/伪造痕迹
        "disposition": "下架 / 冻结并转人工复核",
        # 建议时限：复核前不得对外发布，避免虚假内容扩散
        "time_limit": "2 小时内完成人工复核，复核结论出来前不得对外发布",
        # 责任人角色：谁该拍板
        "owner_role": "平台审核员（必要时联动品牌方与法务）",
        # 是否强制人工：高风险必须过人工，不能自动放行
        "must_human_review": True,
    },
    "suspicious": {
        "disposition": "暂停发布，标记待人工确认",
        "time_limit": "24 小时内由人工确认或解除标记",
        "owner_role": "平台审核员 / 品牌方内容运营",
        "must_human_review": True,
    },
    "credible": {
        "disposition": "放行，留存凭证备查",
        "time_limit": "正常流转，凭证随内容一并归档留存",
        "owner_role": "创作者自查 / 品牌方内容运营",
        "must_human_review": False,
    },
    "inconclusive": {
        "disposition": "谨慎放行（标注「未确证」）/ 索取更强凭证",
        "time_limit": "如用于对外投放，建议 48 小时内向品牌方索取带 C2PA 凭证的原始原图",
        "owner_role": "品牌方内容运营 / 创作者自查",
        "must_human_review": False,
    },
}


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

    # 信号1：生成类证据级联 —— AIGC 本域微调模型 × 频域工具
    # 为什么要两条：两者失效模式不同。AIGC 弱在跨生成器，频域弱在重压/强锐化真实图。
    # 单用一条都有系统性盲区，级联用一条的长处补另一条的短处。
    aigc = evidence.get("aigc")
    aigc_gray_flag = False   # 最终最多升 suspicious 的标记（转人工，不自动下架）
    aigc_score = None
    if aigc:
        items = aigc.get("evidence", [])
        first = items[0] if items else {}
        aigc_score = first.get("aigc_score")
        available = first.get("available", True)
        if aigc_score is None:
            if not available:
                reasons.append("AI 生成检测：模型未就绪（暂未下载成功），本项无法判断，已跳过")
            else:
                reasons.append("AI 生成检测：本次推理未返回分数，本项无法判断，已跳过")
        elif AIGC_TRIGGERS_RISK:
            if aigc_score >= THRESHOLDS["aigc_hard_high"]:
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {aigc_score}（≥{THRESHOLDS['aigc_hard_high']}）→ "
                    f"极可能是整图由 AI 生成的图")
                return "high_risk", reasons
            elif aigc_score >= THRESHOLDS["aigc_high_risk"]:
                aigc_gray_flag = True
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {aigc_score} 落在高分区但未到极高 "
                    f"[{THRESHOLDS['aigc_high_risk']}, {THRESHOLDS['aigc_hard_high']}) → "
                    f"疑似整图 AI 生成，需另一条独立证据佐证或人工复核")
            elif aigc_score >= THRESHOLDS["aigc_abstain"]:
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {aigc_score} 落在不确定区间 "
                    f"[{THRESHOLDS['aigc_abstain']}, {THRESHOLDS['aigc_high_risk']}) → "
                    f"本项模型拿不准，不做判定，交由其他证据与人工复核")
            else:
                reasons.append(
                    f"AI 生成检测：AI 生成概率 {aigc_score}（<{THRESHOLDS['aigc_abstain']}）→ "
                    f"看起来像真实拍摄/人工制作的图")

    # 信号1b：频域取证 —— 与压缩历史无关的独立证据线，专治 AIGC 的跨生成器盲区
    spec = evidence.get("spectral")
    spec_strong = False
    spec_gray = False
    spec_score = None
    if spec:
        s0 = (spec.get("evidence") or [{}])[0]
        spec_score = s0.get("spectral_score")
        if spec_score is None:
            reasons.append("频域取证：未返回倾向分，本项跳过（多半是 results/spectral_calib.json 还没标定）")
        elif not SPECTRAL_TRIGGERS_RISK:
            reasons.append(f"频域取证：倾向分 {spec_score}（已检测，但按开关设置不参与自动定级）")
        elif spec_score >= THRESHOLDS["spectral_strong"]:
            spec_strong = True
            reasons.append(
                f"频域取证：倾向分 {spec_score}（≥{THRESHOLDS['spectral_strong']}）→ "
                f"高频能量/频谱规整度明显不像相机直出照片（生成器无关的独立信号）")
        elif spec_score >= THRESHOLDS["spectral_gray"]:
            spec_gray = True
            reasons.append(
                f"频域取证：倾向分 {spec_score} 落在频域灰带 "
                f"[{THRESHOLDS['spectral_gray']}, {THRESHOLDS['spectral_strong']}) → 有偏离但不极端")
        else:
            reasons.append(f"频域取证：倾向分 {spec_score}（<{THRESHOLDS['spectral_gray']}）→ 频谱形态接近常规照片")

    # 级联判定：两条独立证据一致才自动下架（实测：捕获率 +5.3pt，自动下架误伤保持 0/20）
    if aigc_gray_flag and spec_strong:
        reasons.append(
            f"双证据一致：AIGC 高分区（{aigc_score}）+ 频域强信号（{spec_score}）→ "
            f"两条互相独立的证据都指向 AI 生成，自动定级高风险")
        return "high_risk", reasons
    if aigc_score is not None and THRESHOLDS["aigc_abstain"] <= aigc_score < THRESHOLDS["aigc_high_risk"] \
            and spec_strong:
        # AIGC 弃权但频域强 —— 这正是级联要捞回来的那批跨生成器漏检
        aigc_gray_flag = True
        reasons.append(
            f"级联捞回：AIGC 落在弃权区（{aigc_score}）本不判定，"
            f"但频域给出强信号（{spec_score}）→ 合并升为可疑，转人工复核")
    if spec_gray:
        aigc_gray_flag = True

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

    # ELA 压缩异常 —— 局部篡改痕迹
    # ⚠️ 定位已下调：ELA 依赖"原图 vs 二次编辑"的压缩历史差异，而 AI 从零生成的图
    #    没有编辑区、压缩场均匀，ELA 往往看起来很干净；反过来截图/平台重压会全图误差。
    #    所以 ELA 只能提示"哪块值得人工看"，绝不单独升 high_risk。
    ela = evidence.get("ela")
    if ela:
        regions = ela.get("evidence", [{}])[0].get("suspicious_regions", [])
        if regions and regions[0]["ela_score"] >= THRESHOLDS["ela_region_score"]:
            reasons.append(
                f"ELA：最可疑区域误差 {regions[0]['ela_score']}（≥{THRESHOLDS['ela_region_score']}）→ "
                f"有局部二次编辑痕迹（注意：ELA 只查局部编辑，**不能**用它判断整图是否 AI 生成，"
                f"截图与平台重压也会造成同样现象）")
            return "suspicious", reasons

    # 信号4：C2PA / TC260 凭证 —— 来源可追溯，但必须「验证通过」才提升可信度
    # （reviewer P0：旧逻辑把 present 直接升 credible，是「非空字符串即可信」的误判；
    #  现在只有 verified 且非 AI 声明的凭证才升 credible；AI 声明反而要标可疑。）
    c2pa = evidence.get("c2pa")
    if c2pa:
        e0 = c2pa.get("evidence", [{}])[0]
        status = e0.get("c2pa_status")
        verified = e0.get("c2pa_verified", False)
        declares_ai = e0.get("c2pa_declares_ai_generated", False)

        if declares_ai:
            # 生成器自声明 AI 生成：这是「是 AI 生成」的强证据，绝不是「来源可信」
            reasons.append(
                "C2PA/TC260：凭证或标识显示该内容由 AI 生成（生成器自声明），"
                "属 AI 生成证据，不提升来源可信度，转交人工确认")
            return "suspicious", reasons

        if status in ("present", "present_verified", "present_unverified") and verified:
            reasons.append("C2PA：凭证存在且通过验证，编辑历史可查 → 来源可信度较高")
            return "credible", reasons

        if status == "present_unverified":
            reasons.append(
                "C2PA：图片带有内容凭证，但本环境无法验证签名/声明，"
                "保留为待核验（不自动判可信）")
        # status == "missing" / "error"：无凭证，不处理，继续走 inconclusive

    # 生成类信号收口：AIGC 灰带 / 频域灰带 / 级联捞回，都只转人工复核，不自动下架。
    # 而非 high_risk（自动下架）—— 直接压住「高滤镜真实自拍」被误判为自动下架的误伤；
    # 同时不丢失任何已被模型捕获的 AI 图（它们均 ≥0.995，走上面的 hard_high 分支已是 high_risk）。
    if aigc_gray_flag:
        reasons.append(
            "综合其他工具无强信号，生成类证据只到灰带 → 转为「可疑 / 待人工复核」，不自动下架")
        return "suspicious", reasons

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
    # high_risk 由 AIGC 触发，仅当分数达到 hard_high（≥0.99）—— 与 judge() 一致
    aigc_triggered = aigc_ready and aigc_score >= THRESHOLDS["aigc_hard_high"]
    # 灰带：分数落在 [0.9, 0.99)，疑似但非极高（高滤镜真实自拍也常落此区间）
    aigc_gray = aigc_ready and THRESHOLDS["aigc_high_risk"] <= aigc_score < THRESHOLDS["aigc_hard_high"]

    # 频域证据（2026-10-02 新增）：生成器无关的独立信号，与 AIGC 互为补盲
    spc = _first(evidence, "spectral")
    spec_score = spc.get("spectral_score")
    spec_ready = spec_score is not None
    spec_strong = spec_ready and spec_score >= THRESHOLDS["spectral_strong"]
    spec_gray = spec_ready and THRESHOLDS["spectral_gray"] <= spec_score < THRESHOLDS["spectral_strong"]

    caveat = ("以上结论来自算法比对，不是法律意义上的鉴定意见。"
              "分数高不代表一定有罪（压缩、滤镜也会留痕），分数低也不代表绝对干净。")

    # 防御性初始化：历史坑是某分支漏设某个字段导致 UnboundLocalError。
    # 这里先给三个字段默认值，后面分支覆盖；即使哪个分支忘了写也不会崩。
    headline = ""
    summary = ""
    what_to_do = ""
    # 处置建议条目：四档齐全，缺档位会 KeyError —— 见 ACTION_PLAYBOOK 定义。
    action_playbook = ACTION_PLAYBOOK[risk_level]

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
                "action_playbook": action_playbook,
            }
        elif aigc_triggered:
            # 整图 AI 生成：这是 TruFor 查不出的盲区，由本域微调的 AIGC 模型兜底
            headline = "检测到整图由 AI 生成，建议人工复核"
            summary = (f"AI 生成检测给出 {aigc_score}"
                       f"（≥{THRESHOLDS['aigc_high_risk']}），"
                       "说明这张图很可能是整张由 AI 生成的，而不是相机拍出来的。")
            if spec_ready:
                summary += (f"频域取证同时给出倾向分 {spec_score}"
                            + ("，两条互相独立的证据方向一致。" if spec_score >= THRESHOLDS["spectral_gray"]
                               else "，但频域证据偏弱（重压或强锐化的真实图也可能这样），因此仍需人工确认。"))
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
        elif spec_strong:
            # 级联捞回 / 频域主导：本域模型没给出高置信，但与压缩历史无关的频域证据很强
            headline = "频域证据显示这张图不像相机直出，建议人工复核"
            summary = (
                f"频域取证给出倾向分 {spec_score}（≥{THRESHOLDS['spectral_strong']}）："
                "这张图的高频能量与频谱规整度明显偏离真实相机直出照片。"
                "这条证据与「是谁生成的」无关，因此能补上本域模型换生成器就失效的盲区。")
            if aigc_ready:
                summary += (f"本域 AI 生成模型给出 {aigc_score}"
                            + ("（落在不确定区间，模型自己拿不准）。" if aigc_gray or aigc_score < THRESHOLDS["aigc_high_risk"]
                               else "。"))
            else:
                summary += "本域 AI 生成模型本次未参与判断。"
            summary += ("需要提醒的是：重压缩、强锐化或重度降噪的真实照片也可能呈现类似频谱，"
                        "所以这不构成结论，只说明值得人工看一眼原图。")
            what_to_do = ("标记待人工确认，复核前暂缓对外发布；"
                          "让审核员对照原始拍摄图或品牌方素材库确认。")
        elif aigc_gray:
            headline = "AI 生成检测偏高但未到极高，建议人工复核"
            summary = (f"AI 生成检测给出 {aigc_score}（落在 0.9~0.99 不确定区间），"
                       "单独不足以判定为整图 AI 生成：高滤镜真实自拍也可能落在此区间。"
                       "建议人工结合原图来源与 TruFor 篡改热力图复核后再决定处置。")
            what_to_do = ("标记待人工确认，复核前暂缓对外发布；"
                          "若品牌方能提供带 C2PA 凭证的原始原图，则可排除。")
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
        "action_playbook": action_playbook,
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

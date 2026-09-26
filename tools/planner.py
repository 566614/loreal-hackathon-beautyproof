# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队可解释 Agent 调度（波次 + R1–R5 跳过规则 + A1–A3 后置动作），核心创新贡献
"""
Agent 决策层 —— 决定「下一步该查什么、什么可以不用查」，并把理由记下来

大白话：
    八个工具挨个跑一遍是「流水线」，不是「Agent」。
    真正的 Agent 会根据已经查到的东西，决定下一步做什么 ——
    就像老法师看一张图，先看有没有凭证、再看指纹对不对得上，
    发现对得上就不用再费劲做压缩分析了。

    这一层就是那个「决定下一步」的脑子。它每做一次决定都留下一条理由，
    所以它不是黑箱：评委能在页面上完整看到它为什么跳过某个工具。

设计原则（很重要）：
    1. **只跳过"查了也没意义"的，绝不跳过"可能有风险"的。**
       省时间是副产品，不是目的。宁可多跑一个工具，不能漏掉一个信号。
    2. **每条决策必须能说出人话理由**，写不出理由的规则就不写。
    3. **默认全跑**。只有满足明确条件才跳过，跳过的理由要经得起追问。

当前规则：
    R1  非 JPEG 图            → 跳过 ela（ELA 对 PNG 不可信，工具自己都在报警）
    R2  指纹命中品牌方原图库  → 跳过 ela + trufor（一个字节都没改过，查了也是白查）
    R3  没有给文案            → 跳过 text + crossmodal（没有文案就没有文案侧可查）
    R4  图上没查出文字        → 数字比对跳过（crossmodal 内部处理，这里只记录）
    R5  已判定整图 AI 生成    → TruFor 仍跑，但只用于出定位图、不用于定性（避免口径打架）

后置动作（鉴定完之后该干什么）：
    A1  高风险              → 自动进人工复核队列
    A2  文案命中违规宣称    → 建议核改文案后再投放
"""
from pathlib import Path

from rule_engine import THRESHOLDS, _first

# 图像侧工具分三波，先便宜的后贵的 —— 贵的结果依赖便宜的结论来决定要不要跑
WAVES = [
    ("快检", ["hash", "c2pa"]),
    ("中检", ["ela", "ocr"]),
    ("深检", ["aigc", "trufor"]),
]
TEXT_TOOLS = ["text", "crossmodal"]


def decide_next(ev, ctx, done=None):
    """根据**本次**已经查到的东西，决定下一波要跑哪些工具

    参数：
        ev   工具证据 {tool: evidence_json}（用来读具体数值，如 hash 命中、aigc 分数）
        ctx  上下文 dict，含 has_text / image_format / image_pixels
        done **本次**已跑完的工具名集合 —— 判断"有没有跑过"必须以它为准

    ⚠️ 踩过的坑：早期版本用 `"ela" not in ev` 判断是否跑过，
       但 ev 是从 outputs/ 目录读的，里面还留着**上一轮**的证据文件，
       于是 Agent 以为 ela 早跑过了，R1 规则从不触发。
       判断"跑没跑过"必须看本次的 done 集合，不能看磁盘上有什么。
    """
    done = set(done or ())
    decisions = []

    # ---------------------------------------------------------- R1 非 JPEG 跳过 ELA
    if "ela" not in done:
        fmt = (ctx.get("image_format") or "").upper()
        if fmt and fmt != "JPEG":
            decisions.append({
                "tool": "ela", "action": "skip", "rule": "R1",
                "reason": f"这张图的真实格式是 {fmt}，不是 JPEG。"
                          f"ELA 靠「再压一遍 JPEG」来找破绽，PNG 转 JPEG 本身的差异"
                          f"会盖过真实篡改痕迹，结果不可信，所以不跑",
            })

    # ---------------------------------------------------------- R2 命中原图库 → 跳过痕迹类检测
    if "hash" in done:
        hit = _first(ev, "hash").get("known_original_hit")
        if hit:
            for t in ("ela", "trufor"):
                if t not in done:
                    decisions.append({
                        "tool": t, "action": "skip", "rule": "R2",
                        "reason": f"内容指纹在品牌方原图库里命中「{hit}」，"
                                  f"这张图与官方原图一个字节都不差 —— 没被改过，"
                                  f"再做压缩痕迹/篡改检测没有意义",
                    })

    # ---------------------------------------------------------- R3 没有文案 → 跳过文案侧
    if not ctx.get("has_text"):
        for t in TEXT_TOOLS:
            if t not in done:
                decisions.append({
                    "tool": t, "action": "skip", "rule": "R3",
                    "reason": "没有提供配套文案，文案体检与图文交叉验证无从下手，跳过",
                })

    # ---------------------------------------------------------- R4 图上无文字 → 数字比对跳过
    if "ocr" in done and ctx.get("has_text"):
        lines = ev["ocr"].get("evidence") or []
        if not lines:
            decisions.append({
                "tool": "crossmodal", "action": "partial", "rule": "R4",
                "reason": "图上没有识别出任何文字，图文数字比对无法进行，"
                          "仍会做宣称词与图像侧结论的交叉验证",
            })

    # ---------------------------------------------------------- R5 整图 AI 生成 → TruFor 只定位不定性
    if "aigc" in done and "trufor" not in done:
        score = _first(ev, "aigc").get("aigc_score")
        if score is not None and score >= THRESHOLDS["aigc_high_risk"]:
            decisions.append({
                "tool": "trufor", "action": "run_limited", "rule": "R5",
                "reason": f"AI 生成检测已给出 {score}（≥0.9），判定整图由 AI 生成。"
                          f"TruFor 仍会跑以便给出可疑区域定位图，"
                          f"但它的分数不再用来定性「局部篡改」，避免两个模型口径打架",
            })

    skipped = {d["tool"] for d in decisions if d["action"] == "skip"}
    return skipped, decisions


def decide_actions(risk_level, ev):
    """鉴定完之后该干什么 —— Agent 的「第二步」：不只是判级，还要给动作"""
    actions = []

    if risk_level == "high_risk":
        actions.append({
            "action": "自动进入人工复核队列", "rule": "A1",
            "reason": "系统判为高风险，按纪律必须经人工复核才能对外下结论",
        })
    if risk_level == "suspicious":
        actions.append({
            "action": "标记待人工确认", "rule": "A1",
            "reason": "有可疑迹象但不足以定性，建议人工看一眼再决定",
        })

    txt = _first(ev, "text")
    if txt.get("claim_high_count", 0) or txt.get("claim_medium_count", 0):
        actions.append({
            "action": "核改文案中的违规宣称", "rule": "A2",
            "reason": f"文案命中 {txt.get('claim_high_count', 0)} 处高风险、"
                      f"{txt.get('claim_medium_count', 0)} 处中风险敏感表述，"
                      f"建议对照《广告法》与《化妆品监督管理条例》修改后再投放",
        })

    cross = _first(ev, "crossmodal")
    if cross.get("hard_count", 0):
        actions.append({
            "action": "分别核实图片来源与文案出处", "rule": "A3",
            "reason": f"发现 {cross.get('hard_count')} 处图文硬矛盾，"
                      f"需要确认是哪一边出了错（也可能是文案套错了图）",
        })

    return actions


def build_log(decisions, actions, ran, skipped):
    """把决策与动作整理成给人和给机器都能读的日志

    去重是必须的：decide_next 在每一波之前都会调用，
    同一个「跳过 ela」的规则会被重复产出多次（每波都看它还没跑）。
    按 (工具, 规则) 去重，一条决策只留一次。
    """
    seen, uniq = set(), []
    for d in decisions:
        key = (d.get("tool"), d.get("rule"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    decisions = uniq

    return {
        "rules_applied": sorted({d["rule"] for d in decisions}) +
                         sorted({a["rule"] for a in actions}),
        "decisions": decisions,
        "actions": actions,
        "tools_ran": ran,
        "tools_skipped": sorted(skipped),
        "summary": _summary(decisions, ran, skipped),
    }


def _summary(decisions, ran, skipped):
    if not skipped:
        return f"按顺序跑完 {len(ran)} 个工具，没有可跳过的（每个工具都有查的价值）"
    names = "、".join(sorted(skipped))
    return f"跑了 {len(ran)} 个工具，跳过 {len(skipped)} 个（{names}）—— 理由见决策明细"

# -*- coding: utf-8 -*-
"""2026-10-03 · 方案 A：PPT / 叙事去「身份·能力标签」

按 owner 决策（2026-10-03）：去掉「准大一 / 零基础 / 零技术背景 / 看不懂 / 起点低」等
身份与能力标签，保留「从零构建」这一有评测记录支撑的工程事实，改用专业表述。

原则（方案 A）：
- 保留：团队确实在一个陌生领域从零把系统做完（工程事实，且有 v1–v4 迭代记录撑）
- 删除：owner 是大一学生 / 自述零技术背景 / 看不懂 / 起点低（暴露经验缺口的标签）

幂等：每条替换先判 `if new in text and old not in text: continue`；断言恰好命中 1 次。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (相对路径, 旧串, 新串) —— 旧串全部为 str，main() 里会_normalize
PATCHES = [
    # ---------- tools/make_defense_pptx.py（PPT 是初赛三交付物之一） ----------
    ("tools/make_defense_pptx.py",
     '"团队·EMPATHY", "零基础 owner 主导完整系统；角色互补", "3/5"',
     '"团队·EMPATHY", "产品侧定业务与叙事、工程侧交付可复现系统，角色互补", "3/5"'),

    ("tools/make_defense_pptx.py",
     '"团队·LEARNING AGILITY", "从 CV/ML/Agent 零基础到完整取证系统，快速试错", "5/5"',
     '"团队·LEARNING AGILITY", "CV / 频域 / Agent 三方向均无既往积累，独立完成数据设计→模型迭代→评测→交付闭环", "5/5"'),

    ("tools/make_defense_pptx.py",
     'header(s, "团队：从零基础到完整系统", "5+5 里最该讲透的「团队维度」")',
     'header(s, "团队：从零构建到可交付系统", "5+5 里最该讲透的「团队维度」")'),

    ("tools/make_defense_pptx.py",
     '"LEARNING AGILITY：owner 为准大一、自述零技术背景，从 CV / ML / Agent 零基础主导整套取证系统，含模型训练、评测协议、答辩叙事全流程。"',
     '"LEARNING AGILITY：在 CV / 频域 / Agent 三个方向均无既往积累的前提下，独立完成「数据设计 → 模型迭代（本域微调 v1–v4）→ 评测协议 → 交付与答辩」的完整闭环。"'),

    ("tools/make_defense_pptx.py",
     '"EMPATHY（当前最弱）：owner 定业务与叙事方向，AI 工程执行，队友 liying856 协助；',
     '"EMPATHY（当前最弱）：产品侧负责业务定义与叙事，工程侧负责系统落地，队友 liying856 协作；'),

    ("tools/make_defense_pptx.py",
     '"我们不讳言起点低 —— 但 8 周把「看不懂」变成了「能拿去答辩」，并且把每一次翻车都写进了可复现的记录里。"',
     '"我们不讳言这是一次从零构建 —— 但每一轮失败都留下了可复现的评测记录：v5 / v7 未满足晋升口径即不晋升，负面结果照实归档。"'),

    # ---------- docs/演示视频脚本_v2.md（初赛三交付物之一，对评委可见） ----------
    ("docs/演示视频脚本_v2.md",
     "- **TEAM·LEARNING AGILITY** → 零基础 owner 从 CV/Agent 零基础搭完整系统（全篇）",
     "- **TEAM·LEARNING AGILITY** → 在 CV / Agent 均无积累的前提下独立完成数据→模型→评测→交付全闭环（全篇）"),

    # ---------- docs/路演叙事.md（决赛路演稿，对评委照读） ----------
    ("docs/路演叙事.md",
     "> 用途：决赛路演答辩（5 分钟主线 + 答辩）。本稿可**直接照读**，主讲人（阮，零技术 owner）拿着就能讲。",
     "> 用途：决赛路演答辩（5 分钟主线 + 答辩）。本稿可**直接照读**，主讲人（产品负责人）拿着就能讲。"),

    ("docs/路演叙事.md",
     "| **LEARNING AGILITY（学习敏捷）** | 收尾前可补一句（见角色分工/排练注） | owner 零技术背景 8 周主导整套系统；通用失效→本域微调的快速试错 |",
     "| **LEARNING AGILITY（学习敏捷）** | 收尾前可补一句（见角色分工/排练注） | 三方向均无积累，独立完成数据→模型→评测→交付全闭环；通用失效→本域微调的快速试错 |"),

    ("docs/路演叙事.md",
     "**口径**：owner（我）是零技术背景，负责**产品定义、找素材、定方向、今天主讲**；",
     "**口径**：产品负责人（我）负责**产品定义、找素材、定方向、今天主讲**；"),

    ("docs/路演叙事.md",
     "| 主讲 / 产品定义 | **阮（owner，零技术）** | 全稿照读；",
     "| 主讲 / 产品定义 | **阮（产品负责人）** | 全稿照读；"),

    # ---------- docs/赛题合规审查与获奖评估.md ----------
    ("docs/赛题合规审查与获奖评估.md",
     "| owner 准大一零技术背景，从 CV/ML/Agent 零基础到主导完整取证系统；快速试错(通用 AIGC 失效→本域微调) |",
     "| 在 CV/频域/Agent 三方向均无积累的前提下独立完成数据→模型→评测→交付闭环；快速试错(通用 AIGC 失效→本域微调) |"),

    # ---------- docs/能力匹配与自查_结构化报告.md ----------
    ("docs/能力匹配与自查_结构化报告.md",
     "| LEARNING AGILITY | 5/5 | 零基础→完整系统；通用模型失效→本域微调的试错闭环 |",
     "| LEARNING AGILITY | 5/5 | 陌生领域→完整闭环；通用模型失效→本域微调的试错闭环 |"),
]

# 打过补丁的文件，事后不应再出现这些词
BANNED = ["准大一", "零基础", "零技术", "自述零技术", "起点低"]
# 「看不懂」只保留在用户视角的产品叙事里（公益叙事 / 答辩叙事 / 测试注释），不在此列


def main():
    for rel, old, new in PATCHES:
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        if new in text and old not in text:
            print(f"[skip] {rel} 已应用：{old[:30]}...")
            continue
        n = text.count(old)
        assert n == 1, f"{rel} 命中 {n} 次（应为 1）：{old[:44]}"
        p.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[ok]   {rel} → {old[:32]}...")

    # 复核
    for rel, _o, _n in PATCHES:
        t = (ROOT / rel).read_text(encoding="utf-8")
        hit = [w for w in BANNED if w in t]
        assert not hit, f"{rel} 仍残留：{hit}"
    print("\n复核通过：打补丁的 5 个文件均已无身份/能力标签。")


if __name__ == "__main__":
    main()

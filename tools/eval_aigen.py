# -*- coding: utf-8 -*-
"""评测「整图 AI 生成」检测器的区分度（可复现，随时重跑）。

跑一遍参考图，输出每张的 ai_prob、判定与真值是否一致，
并给出「AI 图最低分 - 真实图最高分」这个区分度指标。

注意 ground_truth 的口径：tampered_* 是「真实照片被局部 PS 过」，
它不是整图 AI 生成，正确答案应为 real —— 这一项专门验证模型
不会把「被人 PS 过」当成「AI 生成的」。

用法: python tools/_verify_aigc.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import aigc_tool  # noqa: E402

# 参考图：ground_truth 标的是「整图是不是 AI 生成的」
# 注意 tampered_* 是「真实照片被局部 PS 过」—— 不是整图 AI，正确答案是 real
CASES = [
    ("data/ai/ai_01.png", "ai"),
    ("data/ai/ai_02.png", "ai"),
    ("data/ai/ai_17.png", "ai"),   # 验证集（未参与训练）
    ("data/ai/ai_18.png", "ai"),   # 验证集（未参与训练）
    ("data/ai/ai_19.png", "ai"),   # 验证集（未参与训练）
    ("data/ai/ai_20.png", "ai"),   # 验证集（未参与训练）
    ("data/real/real_01.jpg", "real"),
    ("data/real/real_02.png", "real"),
    ("data/real/real_03.png", "real"),
    ("data/clean/clean_01.png", "real"),
    ("data/clean/clean_02_recompressed.png", "real"),
    ("data/tampered/tampered_01_copy_move.png", "real"),   # 验证集
    ("data/tampered/tampered_02_splice.png", "real"),      # 验证集
    ("data/tampered/tampered_03_text_edit.png", "real"),   # 验证集
]


def main():
    print("模型目录:", aigc_tool.model_path())
    print(f"{'ai_prob':>8}  {'判定':<6} {'真值':<6} {'✓/✗':<4} 文件")
    print("-" * 72)
    rows = []
    for rel, gt in CASES:
        p = REPO / rel
        if not p.exists():
            print(f"{'--':>8}  {'缺文件':<6} {gt:<6}      {rel}")
            continue
        score, label, ok = aigc_tool.run_aigc(p)
        if not ok:
            print(f"{'--':>8}  {'不可用':<6} {gt:<6}      {rel}  ({label})")
            continue
        pred = "ai" if score >= 0.5 else "real"
        hit = "✓" if pred == gt else "✗"
        rows.append((score, gt, rel, hit))
        print(f"{score:>8.4f}  {pred:<6} {gt:<6} {hit:<4} {rel}")

    ai_scores = [s for s, gt, _, _ in rows if gt == "ai"]
    real_scores = [s for s, gt, _, _ in rows if gt == "real"]
    hits = sum(1 for _, _, _, h in rows if h == "✓")
    print("-" * 72)
    if ai_scores:
        print(f"AI 图   ai_prob: min {min(ai_scores):.4f}  均值 {sum(ai_scores)/len(ai_scores):.4f}  max {max(ai_scores):.4f}")
    if real_scores:
        print(f"真实图  ai_prob: min {min(real_scores):.4f}  均值 {sum(real_scores)/len(real_scores):.4f}  max {max(real_scores):.4f}")
    if ai_scores and real_scores:
        gap = min(ai_scores) - max(real_scores)
        print(f"区分度（AI 最低分 - 真实最高分）= {gap:+.4f}   "
              f"{'完全分开' if gap > 0 else '仍有重叠'}")
    print(f"命中率 {hits}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

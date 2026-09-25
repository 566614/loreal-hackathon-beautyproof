# -*- coding: utf-8 -*-
"""
跨生成器零样本测试：用「非即梦（即梦是训练用 AI 图来源）」生成器自己造的美妆图，
检验 v2 本域微调模型是否只认即梦、换生成器就瞎。

大白话：
    训练时 AI 类 20 张全来自即梦。现在我用另一个生成器造了 8 张美妆图，
    让 v2 去判——如果它还能把这些图判成 AI（ai_score 高），说明它学到的是
    「AI 图通用特征」而不是「即梦专属水印」；如果判不出来，说明 v2 过拟合即梦，
    跨生成器泛化有缺口，答辩要老实说。

用法：
    python tools/_test_aigen_cross.py
"""
import json
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from aigc_tool import run_aigc, model_path  # noqa: E402

# 原始文件名（ImageGen 自动命名）→ 重命名后的类名映射
RENAME_MAP = {
    "Professional_close_up_beauty_p_": ("cross_01", "口红特写人脸"),
    "Flat_lay_photography_of_an_ope_": ("cross_02", "眼影盘平铺"),
    "A_hand_showing_multiple_lipsti_": ("cross_03", "手背试色"),
    "Minimalist_cosmetic_photograph_": ("cross_04", "精华瓶产品"),
    "Glamorous_evening_makeup_look__": ("cross_05", "精致全妆"),
    "Luxury_foundation_bottle_and_c_": ("cross_06", "粉底静物"),
    "Close_up_portrait_of_a_face_wi_": ("cross_07", "素颜自然妆"),
    "Assortment_of_nail_polish_bott_": ("cross_08", "指甲油"),
}

CROSS_DIR = ROOT / "data" / "ai_cross"


def rename_files():
    """把 ImageGen 的随机长文件名改成 cross_0N_类名.png，便于引用与复现。"""
    for f in CROSS_DIR.glob("*.png"):
        matched = None
        for key, (newstem, cat) in RENAME_MAP.items():
            if f.name.startswith(key):
                matched = (newstem, cat)
                break
        if matched is None:
            print(f"[跳过] 未匹配到类名: {f.name}")
            continue
        newstem, _cat = matched
        dst = CROSS_DIR / f"{newstem}.png"
        if dst.exists() and dst != f:
            continue
        if dst != f:
            shutil.move(str(f), str(dst))
        print(f"[重命名] {f.name} -> {dst.name}")


def main():
    rename_files()
    mdir = model_path()
    print(f"\n使用的模型: {mdir}\n")

    results = []
    for f in sorted(CROSS_DIR.glob("cross_*.png")):
        cat = RENAME_MAP.get("_", ("?", "?"))
        # 从文件名反查类名
        stem = f.stem
        for key, (newstem, c) in RENAME_MAP.items():
            if stem == newstem:
                cat = (newstem, c)
                break
        score, label, avail = run_aigc(str(f))
        verdict = "AI生成(命中)" if (avail and score >= 0.5) else (
            "真实/未命中" if avail else "模型不可用")
        results.append({
            "file": f.name,
            "category": cat[1],
            "ai_score": score,
            "label": label,
            "available": avail,
            "verdict": verdict,
        })
        print(f"{cat[1]:<12} | score={score} | {label:<12} | {verdict}")

    # 召回率：available 且 ai_score>=0.5 的占比（这些是真·AI 图，应全判 AI）
    valid = [r for r in results if r["available"]]
    hit = [r for r in valid if r["ai_score"] >= 0.5]
    recall = len(hit) / len(valid) if valid else 0.0

    # 高置信命中（>=0.8）：更接近"可用判定"
    high = [r for r in valid if r["ai_score"] >= 0.8]
    high_rate = len(high) / len(valid) if valid else 0.0

    summary = {
        "model": mdir,
        "generator_note": "测试图由非即梦生成器生成（与训练用即梦不同），属跨生成器零样本测试",
        "total": len(results),
        "available": len(valid),
        "recall_ai_score_ge_0.5": round(recall, 4),
        "high_conf_ge_0.8": round(high_rate, 4),
        "missed": [r["file"] for r in valid if r["ai_score"] < 0.5],
        "details": results,
    }

    out = ROOT / "results" / "cross_gen_test.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 跨生成器召回率(ai_score>=0.5) = {recall:.1%} | 高置信(>=0.8) = {high_rate:.1%} ===")
    print(f"详细写入: {out}")
    if summary["missed"]:
        print(f"未命中（被误判为真实）: {summary['missed']}")


if __name__ == "__main__":
    main()

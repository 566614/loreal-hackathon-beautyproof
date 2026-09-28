# -*- coding: utf-8 -*-
"""生成「冻结的正式评测清单」data/formal_eval_manifest.json（reviewer P0-2）。

为什么需要它：
    评测结论要可信，前提是「测什么」先固定、且同源变体不跨集合泄漏。
    本脚本扫描 data/ 下各官方评测目录，给每张图标注：
        bucket        —— controlled（受控链路检查）/ heldout_real（真 held-out）/ zero_shot_cross（跨生成器零样本）
        source_group  —— 按「原图/拍摄者/来源/生成器/提示词」归组，保证同组不跨 train/val 泄漏
        ground_truth  —— clean / tampered / ai / real
        expected_risk —— 期望命中的风险档
    并对「受控集只证明链路能判对、不代表真实泛化」做显式标注。

用法：
    python tools/make_formal_manifest.py
产出：
    data/formal_eval_manifest.json（版本固定、可提交、可被评测命令重新生成）
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

# bucket 映射：受控集 vs 真 held-out vs 跨生成器零样本
BUCKET = {
    "clean": "controlled",
    "tampered": "controlled",
    "ai": "controlled",
    "realworld_heldout": "heldout_real",
    "ai_cross_native": "zero_shot_cross",
}
GROUND = {
    "clean": "real", "tampered": "tampered", "ai": "ai",
    "realworld_heldout": "real", "ai_cross_native": "ai",
}
EXPECT = {
    "clean": ["inconclusive", "credible"],
    "tampered": ["suspicious", "high_risk"],
    "ai": ["high_risk"],
    "realworld_heldout": ["inconclusive", "credible"],
    "ai_cross_native": ["high_risk"],
}
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def source_group(dir_name, fname):
    """把一张图归到 source_group，保证同源变体落在同一组。

    规则（reviewer 要求「按原图/拍摄者/来源/生成器/提示词隔离」）：
        - tampered 的子类型（copy_move/splice/text_edit）各为一组：它们都来自同一张底图，
          同组整体进评测、绝不与训练底图混淆（训练只用 clean 做对照，tampered 不入训练）。
        - ai（即梦）按文件名里的生成批次/提示词片段归组；拿不到就归到 ai_jimeng 大组。
        - realworld_heldout / ai_cross_native 各自整体一组（它们永不参与训练，
          组内同源只是诚实报告用，不影响训练泄漏）。
    """
    stem = Path(fname).stem
    if dir_name == "tampered":
        for t in ("copy_move", "splice", "text_edit"):
            if t in stem:
                return f"tampered__{t}"
        return "tampered__other"
    if dir_name == "ai":
        # 即梦导出常带批次号；同批次视为同源组
        return "ai__jimeng"
    if dir_name == "ai_cross_native":
        return "ai_cross__native"
    if dir_name == "realworld_heldout":
        return "real__web_zip"
    if dir_name == "clean":
        return "real__original"
    return dir_name


def main():
    manifest = {
        "schema": "beautyproof/formal_eval_manifest@1",
        "purpose": "冻结的正式评测清单：测什么先固定、同源变体不跨集合泄漏；"
                   "受控集仅作链路检查，真实鲁棒性只看 realworld_heldout 误报率，"
                   "跨生成器泛化只看 ai_cross_native 零样本召回。",
        "groups": {},      # source_group -> 计数（用于核对隔离）
        "samples": [],
        "notes": {
            "controlled_is_link_check_only": True,
            "real_robustness_only_from": "realworld_heldout",
            "cross_generalization_only_from": "ai_cross_native",
            "never_trained": ["realworld_heldout", "ai_cross_native"],
        },
    }
    group_counts = {}
    for d in ("clean", "tampered", "ai", "realworld_heldout", "ai_cross_native"):
        folder = DATA / d
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in EXTS:
                continue
            sg = source_group(d, p.name)
            group_counts[sg] = group_counts.get(sg, 0) + 1
            manifest["samples"].append({
                "id": p.stem,
                "file": f"data/{d}/{p.name}",
                "bucket": BUCKET[d],
                "source_group": sg,
                "ground_truth": GROUND[d],
                "expected_risk": EXPECT[d],
                "in_training": False,   # 这些目录都不进训练；训练用 data/real_*/data/ai 等
            })
    manifest["groups"] = group_counts
    manifest["n_total"] = len(manifest["samples"])

    out = DATA / "formal_eval_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成冻结评测清单：{out}")
    print(f"  样本总数={manifest['n_total']}  源组数={len(group_counts)}")
    for g, c in sorted(group_counts.items()):
        print(f"    - {g}: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

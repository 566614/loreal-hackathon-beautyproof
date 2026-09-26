# -*- coding: utf-8 -*-
"""伪标注（self-training）：用已上线的本域微调 AIGC 模型，把团队「自产、未标注」的
真实美妆图扩充进 REAL 训练类，缓解「训练真实图太少 → 真实美妆图被误报成 AI」的问题。

合规边界（赛题2「数据来源合规 + 原创性」）
------------------------------------------
- 只读取团队自己产出 / 收集、尚未标注的真实美妆图（默认 data/xhs_real_unlabeled/）。
- **绝不**下载、爬取任何第三方数据集；**绝不**引入 GenImage / COCO / ImageNet（作训练集）
  或任何规则外数据。
- 判定规则保守：
  * 模型高置信判为「真实」（ai_prob <= REAL_CONF=0.25）的图 → 复制进 data/real_extra/，
    纳入 REAL 训练池（伪标签 = real）。
  * 模型高置信判为「AI」（ai_prob >= AI_REJECT=0.5）的图 → 排除，避免污染 REAL 类，
    并写入清单提示人工确认（可能是 AI 图混进来了）。
  * 中间模糊带（0.25 < ai_prob < 0.5）的图 → 跳过，不纳入，交人工决定。
- 产出 results/pseudo_label_real.json 清单（每张图的最终处置 + 模型分数），全程可核查。

这是机器学习里标准的「self-training / 伪标注」手段：用模型给自有未标注数据打软标签，
只采纳高置信样本做扩充。所有数据都是团队自有，没有外采，符合赛题要求。
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UNLABELED_DIR = REPO / "data" / "xhs_real_unlabeled"   # 团队自产未标注真实美妆图放这里
EXTRA_REAL_DIR = REPO / "data" / "real_extra"          # 伪标注为真实的图归这里
MANIFEST = REPO / "results" / "pseudo_label_real.json"

REAL_CONF = 0.25   # ai_prob <= 此值 → 高置信真实，纳入
AI_REJECT = 0.50   # ai_prob >= 此值 → 高置信 AI，排除
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def collect_unlabeled(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS)


def main():
    unlabeled = collect_unlabeled(UNLABELED_DIR)
    print(f"未标注真实美妆图目录：{UNLABELED_DIR}", flush=True)
    if not unlabeled:
        print("  → 该目录为空。请把团队自产 / 收集的真实美妆图（非 AI 生成）放进去，"
              "再运行本脚本做伪标注扩充。\n"
              "  （这些图必须是你们自己拍的、或用户/队友提供的真实图，不能来自第三方数据集。）",
              flush=True)
        return 1

    # 复用已上线的推理入口（单一事实来源），避免重复实现阈值逻辑
    sys.path.insert(0, str(REPO / "tools"))
    from aigc_tool import run_aigc

    EXTRA_REAL_DIR.mkdir(parents=True, exist_ok=True)
    (REPO / "results").mkdir(exist_ok=True)

    accepted, rejected, skipped = [], [], []
    for img in unlabeled:
        ai_score, _label, available = run_aigc(img)
        if not available or ai_score is None:
            skipped.append({"file": img.name, "reason": "model_unavailable", "ai_prob": None})
            continue
        if ai_score <= REAL_CONF:
            dst = EXTRA_REAL_DIR / img.name
            # 同名则加后缀，避免覆盖
            if dst.exists():
                dst = EXTRA_REAL_DIR / f"{img.stem}__{abs(hash(str(img))) & 0xffff}{img.suffix}"
            shutil.copy(str(img), str(dst))
            accepted.append({"file": img.name, "ai_prob": round(ai_score, 4), "copied_to": dst.name})
        elif ai_score >= AI_REJECT:
            rejected.append({"file": img.name, "ai_prob": round(ai_score, 4),
                             "reason": "高置信判为 AI，已排除（请人工核实是否为 AI 图混入）"})
        else:
            skipped.append({"file": img.name, "ai_prob": round(ai_score, 4),
                            "reason": "模糊带，未采纳"})

    manifest = {
        "source_dir": str(UNLABELED_DIR),
        "extra_real_dir": str(EXTRA_REAL_DIR),
        "thresholds": {"real_conf_max": REAL_CONF, "ai_reject_min": AI_REJECT},
        "counts": {"scanned": len(unlabeled), "accepted_real": len(accepted),
                   "rejected_ai": len(rejected), "skipped": len(skipped)},
        "accepted_real": accepted, "rejected_ai": rejected, "skipped": skipped,
        "note": "accepted_real 中的图已复制到 data/real_extra/，下一步用 train_aigen_v4.py "
                "并入 REAL 训练类；全部为团队自产图，符合赛题数据合规要求。",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n扫描 {len(unlabeled)} 张：纳入 REAL {len(accepted)} 张，排除 AI {len(rejected)} 张，"
          f"跳过模糊 {len(skipped)} 张", flush=True)
    print(f"纳入的图已复制到 {EXTRA_REAL_DIR}", flush=True)
    print(f"清单已写 {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

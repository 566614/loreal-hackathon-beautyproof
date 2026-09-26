# -*- coding: utf-8 -*-
# 来源：原创 —— 真实世界 held-out 评测（测试集绝不回灌训练）
"""在「从未参与训练」的图上量化模型表现，防止在训练用过的图上自夸准确率。

用法：
    # 真实图目录 → 量化「真实图被误判为 AI」的误报率
    python tools/eval_realworld_heldout.py data/realworld_heldout --model beautyproof_aigen_v5 --tag v5

    # AI 图目录 → 量化召回率（如跨生成器 held-out）
    python tools/eval_realworld_heldout.py data/ai_cross_native --model beautyproof_aigen_v4 --kind ai --tag v4_cross

为什么单独写这个工具：
    1. 评测口径必须「先 pin 模型再跑」，并断言 model_path() 真的解析到目标模型——
       否则权重缺失时 aigc_tool 会静默回退到旧模型，评出来的就不是你以为的那个版本
       （v4 曾因 config.json 漏写而静默失效，教训见 2026-09-26 日志）。
    2. 同时输出分档统计与逐图分数清单，误报率 / 召回率都可从同一份 JSON 复核。
"""
import argparse
import datetime
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def collect(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser(description="真实世界 held-out 评测（pin 模型 + 防静默回退）")
    ap.add_argument("dirs", nargs="+", help="图片目录（真实图用 --kind real，AI 图用 --kind ai）")
    ap.add_argument("--model", required=True, help="要评测的模型名，如 beautyproof_aigen_v5")
    ap.add_argument("--kind", choices=["real", "ai"], default="real",
                    help="real=统计误报率（真实图被判成 AI 的比例）；ai=统计召回率")
    ap.add_argument("--tag", default=None, help="输出文件名标识（默认用模型名）")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "tools"))
    import aigc_tool
    from rule_engine import THRESHOLDS

    # --- pin 模型并断言真的解析到了它（防静默回退） ---
    aigc_tool.MODEL_PRIORITY = [args.model]
    resolved = aigc_tool.model_path()
    if not resolved or Path(resolved).name != args.model:
        print(f"[FAIL] 模型 pin 失败：要评 {args.model}，实际解析到 {resolved}", flush=True)
        return 1

    pos_thr = aigc_tool.AI_POSITIVE_THRESHOLD      # 阳性阈值（0.6）
    high_thr = THRESHOLDS["aigc_high_risk"]        # 高危阈值（0.9）

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "resolved_model_dir": resolved,
        "kind": args.kind,
        "thresholds": {"positive": pos_thr, "high_risk": high_thr},
        "dirs": {},
    }

    for d in args.dirs:
        folder = Path(d)
        imgs = collect(folder)
        if not imgs:
            print(f"[警告] 目录为空或不存在：{folder}", flush=True)
            continue
        rows, unavailable = [], []
        for p in imgs:
            score, label, available = aigc_tool.run_aigc(p)
            if not available or score is None:
                unavailable.append({"file": p.name, "reason": str(label)[:200]})
                continue
            rows.append({"file": p.name, "ai_prob": score})
        probs = [r["ai_prob"] for r in rows]
        n = len(probs)
        n_pos = sum(1 for x in probs if x >= pos_thr)
        n_high = sum(1 for x in probs if x >= high_thr)
        n_abstain = sum(1 for x in probs if aigc_tool.AI_INCONCLUSIVE_LOW <= x < pos_thr)
        entry = {
            "dir": str(folder),
            "n": n, "n_unavailable": len(unavailable),
            "mean_prob": round(statistics.mean(probs), 4) if n else None,
            "median_prob": round(statistics.median(probs), 4) if n else None,
            "max_prob": max(probs) if n else None,
            "n_ge_positive": n_pos, "n_ge_high": n_high, "n_abstain_band": n_abstain,
            "per_file": rows, "unavailable": unavailable,
        }
        if args.kind == "real":
            entry["false_positive_rate_positive"] = round(n_pos / n, 4) if n else None
            entry["false_positive_rate_high"] = round(n_high / n, 4) if n else None
        else:
            entry["recall_positive"] = round(n_pos / n, 4) if n else None
            entry["recall_high"] = round(n_high / n, 4) if n else None
        report["dirs"][folder.name] = entry

        tag = "误报" if args.kind == "real" else "召回"
        print(f"\n=== {folder.name}（{args.model}, {args.kind}）===", flush=True)
        print(f"  n={n}  mean={entry['mean_prob']}  median={entry['median_prob']}  max={entry['max_prob']}", flush=True)
        print(f"  ≥{pos_thr}: {n_pos}/{n}  ≥{high_thr}: {n_high}/{n}  模糊带: {n_abstain}", flush=True)
        if n:
            print(f"  {tag}_rate(≥{pos_thr}) = {entry.get('false_positive_rate_positive', entry.get('recall_positive'))}", flush=True)

    tag = args.tag or args.model
    out = REPO / "results" / f"realworld_eval_{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""跨生成器 / 跨风格零样本召回评测 —— 新批次 data/ai_cross_native2（12 张，2026-10-02）

用法：
    python tools/eval_crossgen_native2.py

产出：
    results/crossgen_native2_eval.json

铁律（违反=评测无效，务必保留）：
  1. 本目录的图是 held-out，只做评测，绝不进任何训练集。
  2. **先 pin 模型再跑，并断言 model_path() 真的解析到目标模型**。
     权重缺失 / config.json 漏写时 aigc_tool 会静默回退到旧模型，
     评出来的就不是你以为的那个版本（v4 曾因 config.json 漏写静默失效）。
     → 本脚本只临时改 aigc_tool.MODEL_PRIORITY（进程内，不落盘、不改生产代码）。
  3. 直接调 aigc_tool.run_aigc(path)，不走 tools/pipeline.py --fast（那个会跳过 aigc）。
  4. 缺失/报错的样本只记missing_files，绝不进分母；必须输出
     n_total / n_evaluated / missing_files。
  5. 绝不引用训练用过的图报准确率。
"""
import datetime
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET_MODEL = "beautyproof_aigen_v4"          # 生产模型（MobileNetV3 本域微调二分类）
NEW_DIR = REPO / "data" / "ai_cross_native2"     # 本次新生成的 12 张
PREV_DIR = REPO / "data" / "ai_cross_native"     # 上一批 12 张（同口径复算做对照）
EXTS = (".jpg", ".jpeg", ".png", ".webp")

# 沙箱守卫：单个 turn 内删除 ≥50 次文件会被 SIGTERM。
# → 本脚本不做任何文件删除/落盘中间证据，只在最后写一次结果 JSON。


def collect(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def load_style_tags(folder: Path):
    """读 style_tag：新批次从目录内 MANIFEST.json；旧批次回落到 results/crossgen_eval.json
    里已落盘的 style_hint（那批生成时没写 MANIFEST）。读不到就标 unknown。"""
    mf = folder / "MANIFEST.json"
    if mf.exists():
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
        out = {}
        for item in data.get("images", []):
            key = item.get("file") or f"{item.get('id')}.png"
            out[key] = {"id": item.get("id"), "style_tag": item.get("style_tag", "unknown")}
        if out:
            return out

    legacy = REPO / "results" / "crossgen_eval.json"
    if legacy.exists():
        try:
            rows = json.loads(legacy.read_text(encoding="utf-8")).get("per_image", [])
        except Exception:  # noqa: BLE001
            rows = []
        return {
            r["file"]: {"id": Path(r["file"]).stem, "style_tag": r.get("style_hint", "unknown")}
            for r in rows if "file" in r
        }
    return {}


def eval_dir(aigc_tool, folder: Path, pos_thr: float, high_thr: float, hard_thr: float, label: str):
    imgs = collect(folder)
    n_total = len(imgs)
    meta = load_style_tags(folder)
    rows, missing = [], []

    for p in imgs:
        info = meta.get(p.name, {})
        row = {
            "id": info.get("id", p.stem),
            "file": p.name,
            "style_tag": info.get("style_tag", "unknown"),
        }
        try:
            score, _lbl, available = aigc_tool.run_aigc(p)
        except Exception as e:  # noqa: BLE001
            missing.append({"file": p.name, "reason": f"exception: {type(e).__name__}: {str(e)[:200]}"})
            print(f"  [缺失] {p.name}: {row['reason'] if 'reason' in row else type(e).__name__}", flush=True)
            continue
        if not available or score is None:
            missing.append({"file": p.name, "reason": str(_lbl)[:200]})
            print(f"  [缺失] {p.name}: {str(_lbl)[:120]}", flush=True)
            continue
        row["ai_prob"] = round(float(score), 4)
        rows.append(row)
        print(f"  {p.name}  ai_prob={row['ai_prob']}", flush=True)

    probs = [r["ai_prob"] for r in rows]
    n = len(probs)
    n_hard = sum(1 for x in probs if x >= hard_thr)  # 0.99 aigc_hard_high：自动定级线
    n_high = sum(1 for x in probs if x >= high_thr)  # 0.90 aigc_high_risk：召回口径主指标
    n_05 = sum(1 for x in probs if x >= 0.5)
    n_pos = sum(1 for x in probs if x >= pos_thr)    # 0.6 阳性线
    misses = [
        {"id": r["id"], "style_tag": r["style_tag"], "ai_prob": r["ai_prob"]}
        for r in rows if r["ai_prob"] < high_thr
    ]
    miss_05 = [
        {"id": r["id"], "style_tag": r["style_tag"], "ai_prob": r["ai_prob"]}
        for r in rows if r["ai_prob"] < 0.5
    ]
    miss_hard = [
        {"id": r["id"], "style_tag": r["style_tag"], "ai_prob": r["ai_prob"]}
        for r in rows if r["ai_prob"] < hard_thr
    ]

    entry = {
        "label": label,
        "dir": str(folder),
        "n_total": n_total,
        "n_evaluated": n,
        "missing_files": missing,
        "thresholds_used": {
            "recall_line": high_thr,
            "hard_high_line": hard_thr,
            "loose_line": 0.5,
            "positive_line": pos_thr,
        },
        "recall_at_0.9": round(n_high / n, 4) if n else None,
        "recall_at_0.99": round(n_hard / n, 4) if n else None,
        "recall_at_0.5": round(n_05 / n, 4) if n else None,
        "recall_at_positive": round(n_pos / n, 4) if n else None,
        "n_detected_at_0.9": n_high,
        "n_detected_at_0.99": n_hard,
        "n_detected_at_0.5": n_05,
        "mean_prob": round(statistics.mean(probs), 4) if n else None,
        "median_prob": round(statistics.median(probs), 4) if n else None,
        "min_prob": min(probs) if n else None,
        "max_prob": max(probs) if n else None,
        "misses": misses,
        "misses_at_0.5": miss_05,
        "misses_at_0.99": miss_hard,
        "per_image": rows,
    }

    print(f"\n=== {label}（{TARGET_MODEL}）===", flush=True)
    print(f"  n_total={n_total}  n_evaluated={n}  missing={len(missing)}", flush=True)
    print(f"  mean={entry['mean_prob']}  median={entry['median_prob']}  min={entry['min_prob']}  max={entry['max_prob']}", flush=True)
    print(f"  recall@0.9(main) = {n_high}/{n} = {entry['recall_at_0.9']}", flush=True)
    print(f"  recall@0.99(hard) = {n_hard}/{n} = {entry['recall_at_0.99']}", flush=True)
    print(f"  recall@0.5(loose) = {n_05}/{n} = {entry['recall_at_0.5']}", flush=True)
    if misses:
        print("  漏检(<0.9):", flush=True)
        for m in misses:
            print(f"    - {m['id']}  {m['style_tag']}  {m['ai_prob']}", flush=True)
    return entry


def main():
    sys.path.insert(0, str(REPO / "tools"))
    import aigc_tool
    from rule_engine import THRESHOLDS

    # --- pin 模型并断言真的解析到它（防静默回退） ---
    # 临时改进程内变量，脚本退出即失效；不修改 tools/aigc_tool.py 任何文件内容。勿回退。
    aigc_tool.MODEL_PRIORITY = [TARGET_MODEL]
    resolved = aigc_tool.model_path()
    resolved_name = Path(resolved).name if resolved else None
    if not resolved or resolved_name != TARGET_MODEL:
        print(f"[FAIL] 模型 pin 失败：要评 {TARGET_MODEL}，实际解析到 {resolved}", flush=True)
        return 1
    if TARGET_MODEL not in str(resolved):
        print(f"[FAIL] 断言失败：解析路径不含 {TARGET_MODEL}：{resolved}", flush=True)
        return 1
    print(f"[PIN OK] aigc_tool.model_path() -> {resolved}", flush=True)
    print(f"[PIN OK] MODEL_PRIORITY = {aigc_tool.MODEL_PRIORITY}（已临时 pin，勿回退）", flush=True)

    pos_thr = aigc_tool.AI_POSITIVE_THRESHOLD      # 0.6 阳性线
    high_thr = THRESHOLDS["aigc_high_risk"]        # 0.90 召回主口径
    hard_thr = THRESHOLDS["aigc_hard_high"]        # 0.99 自动定级线

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_file": resolved_name,
        "model_dir": resolved,
        "model_priority_pinned": list(aigc_tool.MODEL_PRIORITY),
        "pin_assertion": "aigc_tool.MODEL_PRIORITY 被临时设为 [beautyproof_aigen_v4] 并断言 model_path() 解析结果含 v4",
        "inference_entrypoint": "aigc_tool.run_aigc(path) 直接调用（未走 pipeline --fast）",
        "thresholds": {
            "recall_line_0.9": high_thr,
            "hard_high_line_0.99": hard_thr,
            "positive_line_0.6": pos_thr,
            "note": "recall_at_0.9 为主指标（aigc_high_risk 灰带线，召回即召回）；"
                    "recall_at_0.99 为自动定级线；recall_at_0.5 为宽松口径",
        },
        "train_split": False,
        "train_split_note": "ai_cross_native2 为 held-out 评测集，未参与任何训练",
    }

    report["current_batch"] = eval_dir(aigc_tool, NEW_DIR, pos_thr, high_thr, hard_thr,
                                       label="ai_cross_native2 (2026-10-02 新生成12张)")
    report["previous_batch"] = eval_dir(aigc_tool, PREV_DIR, pos_thr, high_thr, hard_thr,
                                        label="ai_cross_native (上一批 12 张，同口径对照)")

    c, pv = report["current_batch"], report["previous_batch"]
    report["comparison"] = {
        "new_recall_at_0.9": c["recall_at_0.9"],
        "prev_recall_at_0.9": pv["recall_at_0.9"],
        "delta_recall_at_0.9": (
            round(c["recall_at_0.9"] - pv["recall_at_0.9"], 4)
            if c["recall_at_0.9"] is not None and pv["recall_at_0.9"] is not None else None
        ),
        "new_recall_at_0.99": c["recall_at_0.99"],
        "prev_recall_at_0.99": pv["recall_at_0.99"],
        "new_recall_at_0.5": c["recall_at_0.5"],
        "prev_recall_at_0.5": pv["recall_at_0.5"],
        "new_miss_count": len(c["misses"]),
        "prev_miss_count": len(pv["misses"]),
    }

    out = REPO / "results" / "crossgen_native2_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写 {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

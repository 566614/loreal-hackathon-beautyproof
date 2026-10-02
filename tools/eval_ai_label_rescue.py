# -*- coding: utf-8 -*-
"""
量化「可见 AI 标识」这一层到底补了多少盲区，以及它自己会不会乱判。

要回答三个问题，每个都必须有数字：
    Q1 补盲：在全新风格的 AI 图上，模型单独判 9/12，加上标识层能到几/12？
    Q2 误伤：在**真实图**（含永不训练 held-out）上，标识层会不会把真图判成 AI？
    Q3 覆盖率：AI 图里到底有多少张带可见标识？（这决定这招能救多少、也决定诚实的边界话术）

诚实评测纪律（同仓库其他评测脚本）：
    1. 跑之前 pin 住模型，断言 model_path() 解析到的确实是 v4，**不许静默回退**；
    2. 不走 pipeline --fast，直接调 aigc_tool.run_aigc；
    3. 没跑/缺图的样本记进 missing_files，**绝不进分母**；
    4. 真实图与 AI 图的判据分开算，不混成一个"准确率"。

用法（venv python）：
    python tools/eval_ai_label_rescue.py
产物：results/ai_label_rescue.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

import ai_label_tool  # noqa: E402
import aigc_tool  # noqa: E402

PIN = ["beautyproof_aigen_v4"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AIGC_LINE = 0.9  # 与 rule_engine.THRESHOLDS["aigc_high_risk"] 对齐：≥ 此判"模型认为整图 AI"

DATASETS = [
    ("ai_cross_native2", "data/ai_cross_native2", "ai",
     "2026-10-02 新生成的 12 张全新风格 AI 图（held-out，模型没见过）"),
    ("ai_cross_native", "data/ai_cross_native", "ai",
     "2026-09-26 生成的 12 张 AI 图（held-out，模型没见过）"),
    ("ai_cross", "data/ai_cross", "ai",
     "跨生成器 held-out 集（即梦/平台风格）"),
    ("realworld_heldout", "data/realworld_heldout", "real",
     "真实图 held-out 10 张（按 seed 20260926 切出，**永不进训练**）"),
    ("real_xhs", "data/real_xhs", "real",
     "小红书真实精修美妆图（训练用过的类，只用来确认标识层不乱判）"),
]


def pin_model():
    """pin 住生产模型并断言 —— 防止「静默回退旧模型还报通过」重演。"""
    aigc_tool.MODEL_PRIORITY = list(PIN)
    mp = aigc_tool.model_path()
    name = getattr(mp, "name", str(mp))
    if "v4" not in str(mp):
        raise SystemExit(f"PIN FAIL: 期望 v4，实际解析到 {mp}")
    return str(mp)


def eval_dir(rel_dir, label):
    d = REPO / rel_dir
    if not d.exists():
        return {"dir": rel_dir, "n_total": 0, "n_evaluated": 0, "missing_files": ["<目录不存在>"], "rows": []}
    rows, missing = [], []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in IMG_EXT:
            continue
        row = {"file": f.name, "aigc_score": None, "ai_label": None, "label_text": None,
               "label_conf": None, "error": None}
        try:
            row["aigc_score"] = float(aigc_tool.run_aigc(str(f))[0])
        except Exception as e:                      # 单张失败不中断整批
            row["error"] = f"aigc:{type(e).__name__}"
        try:
            ev = ai_label_tool.run(str(f))
            e0 = ev["evidence"][0]
            row["ai_label"] = bool(ev["ai_label_detected"])
            bh = e0.get("best_hit") or {}
            row["label_text"] = bh.get("text")
            row["label_conf"] = bh.get("confidence")
        except Exception as e:
            row["error"] = (row["error"] or "") + f"|ai_label:{type(e).__name__}"
        if row["aigc_score"] is None and row["ai_label"] is None:
            missing.append(f.name)
            continue
        rows.append(row)
        print(f"    {f.name:26s} aigc={row['aigc_score']} label={row['ai_label']} "
              f"({row['label_text']})", flush=True)
        time.sleep(0.05)

    n = len(rows)
    model_hits = sum(1 for r in rows if (r["aigc_score"] or 0) >= AIGC_LINE)
    label_hits = sum(1 for r in rows if r["ai_label"])
    rescued = [r["file"] for r in rows
               if r["ai_label"] and (r["aigc_score"] or 0) < AIGC_LINE]
    system_hits = sum(1 for r in rows
                      if r["ai_label"] or (r["aigc_score"] or 0) >= AIGC_LINE)
    out = {
        "dir": rel_dir, "label": label, "n_total": n, "n_evaluated": n,
        "missing_files": missing,
        "model_only_hits": model_hits,
        "ai_label_hits": label_hits,
        "rescued_by_label": rescued,
        "n_rescued": len(rescued),
        "system_hits": system_hits,
    }
    out["model_only_rate"] = round(model_hits / n, 4) if n else None
    out["ai_label_rate"] = round(label_hits / n, 4) if n else None
    out["system_rate"] = round(system_hits / n, 4) if n else None
    return out


def main():
    model = pin_model()
    print(f"pinned model: {model}", flush=True)
    results = {}
    for key, rel, label, desc in DATASETS:
        print(f"[{key}] {desc}", flush=True)
        r = eval_dir(rel, label)
        r["desc"] = desc
        results[key] = r
        print(f"  -> model {r['model_only_hits']}/{r['n_evaluated']} | "
              f"label {r['ai_label_hits']}/{r['n_evaluated']} | "
              f"system {r['system_hits']}/{r['n_evaluated']} | rescued {r['n_rescued']}", flush=True)

    ai_keys = [k for k, _, l, _ in DATASETS if l == "ai"]
    real_keys = [k for k, _, l, _ in DATASETS if l == "real"]
    ai_n = sum(results[k]["n_evaluated"] for k in ai_keys)
    ai_model = sum(results[k]["model_only_hits"] for k in ai_keys)
    ai_label = sum(results[k]["ai_label_hits"] for k in ai_keys)
    ai_sys = sum(results[k]["system_hits"] for k in ai_keys)
    real_n = sum(results[k]["n_evaluated"] for k in real_keys)
    real_label_fp = sum(results[k]["ai_label_hits"] for k in real_keys)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "model_file": model,
        "pin_assertion": f"MODEL_PRIORITY 被 pin 为 {PIN} 并断言 model_path() 解析结果含 v4",
        "inference_entrypoint": "aigc_tool.run_aigc 直接调用（未走 pipeline --fast）",
        "aigc_line": AIGC_LINE,
        "ai_sets": {
            "n": ai_n,
            "model_only": ai_model,
            "model_only_rate": round(ai_model / ai_n, 4) if ai_n else None,
            "ai_label_hits": ai_label,
            "ai_label_rate": round(ai_label / ai_n, 4) if ai_n else None,
            "system_hits": ai_sys,
            "system_rate": round(ai_sys / ai_n, 4) if ai_n else None,
        },
        "real_sets": {
            "n": real_n,
            "ai_label_false_positives": real_label_fp,
            "ai_label_fp_rate": round(real_label_fp / real_n, 4) if real_n else None,
        },
        "per_dataset": results,
        "interpretation": (
            "标识层只在图上确实带「AI生成/AI绘制」等可见标识时生效：它把一部分模型盲区救回来，"
            "但覆盖率取决于图上有没有标识（见 ai_label_rate），对洗掉标识的 AI 图无效；"
            "真实图上的误报数见 real_sets.ai_label_false_positives。两者必须一起讲，不能只报好看的那个。"
        ),
    }
    out = REPO / "results" / "ai_label_rescue.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ai_sets": summary["ai_sets"], "real_sets": summary["real_sets"]},
                     ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

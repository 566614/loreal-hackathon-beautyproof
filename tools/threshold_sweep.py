# -*- coding: utf-8 -*-
"""阈值单变量实验 —— reviewer P1-3「逐轮只改一个主要变量（样本/预处理/阈值/证据融合）」。

本轮**只动「阈值」这一个变量**：模型权重、训练数据、预处理、证据融合方式全部不动。
在**冻结测试集**上扫描 AIGC 判定阈值，回答一个问题：

    是否存在某个阈值，能在「不增加真实图高风险误报」的前提下，
    把跨生成器零样本召回提上去？

为什么只用这两个集合做决策：
    reviewer 明确要求「受控样本 6/6 可作为链路检查，不作为真实泛化准确率」。
    因此本实验只用两个真实泛化集合：
        - realworld_heldout（10 张永不出训的真实图）→ 误报率
        - ai_cross_native（12 张跨生成器零样本 AI 图）→ 召回率
    受控集（data/ai、data/tampered、data/clean）不参与阈值决策，只作链路自检。

上一轮结论回顾（样本变量，v6）：v6 把跨生成器召回做到 6/12=50%，
低于 v4 生产基线 66.7%，故按 reviewer 规则「未改善不替换生产模型」而弃用 v6。
本轮改试阈值变量，看判别面本身有没有可调节的余量。

用法：
    python tools/threshold_sweep.py
输出：
    results/threshold_sweep_v4.json
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

EVAL = REPO / "results" / "closedloop_eval.json"
OUT = REPO / "results" / "threshold_sweep_v4.json"

# 生产阈值（必须与 tools/rule_engine.py THRESHOLDS 保持一致）
try:
    import rule_engine
    BASELINE_HIGH = rule_engine.THRESHOLDS["aigc_high_risk"]
    BASELINE_HARD = rule_engine.THRESHOLDS["aigc_hard_high"]
    # 灰带下沿（AIGC 侧）。依据 rule_engine.py 第 40 / 65 行：AIGC 分数
    # ∈ [aigc_high_risk=0.9, aigc_hard_high=0.99) 即判 suspicious（灰带，转人工不自动下架）——
    # 也就是说灰带入口就是 aigc_high_risk 本身，规则引擎里**没有**独立的 aigc_suspicious 键。
    # 这里用 .get 兼容将来新增该键的情形，不臆造当前不存在的阈值。
    BASELINE_SUSP = rule_engine.THRESHOLDS.get("aigc_suspicious", BASELINE_HIGH)
except Exception:
    BASELINE_HIGH, BASELINE_HARD = 0.9, 0.99
    BASELINE_SUSP = BASELINE_HIGH

REAL_SET = "realworld_heldout"
AI_SET = "ai_cross_native"

SWEEP_STEP = 0.01


def load_scores(eval_path, group):
    """从正式评测结果里取 (文件名, aigc_score)；只保留真正有分数的样本。"""
    data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    ds = data.get("datasets", {}).get(group, {})
    rows, skipped = [], []
    for s in ds.get("samples", []):
        sc = (s.get("key_scores") or {}).get("aigc_score")
        if sc is None:
            skipped.append(s.get("file"))
            continue
        rows.append((s.get("file"), float(sc)))
    meta = {
        "n_total": ds.get("n_total"),
        "n_evaluated": ds.get("n_evaluated"),
        "coverage": ds.get("coverage"),
        "n_scored": len(rows),
        "skipped_no_score": skipped,
    }
    return rows, meta


def evaluate(real_rows, ai_rows, t):
    """给定阈值，算误报率（真实图被判 AI）与召回率（AI 图被判 AI）。"""
    n_real, n_ai = len(real_rows), len(ai_rows)
    fp = sum(1 for _f, s in real_rows if s >= t)
    hit = sum(1 for _f, s in ai_rows if s >= t)
    return {
        "threshold": round(t, 4),
        "real_fp": fp,
        "real_n": n_real,
        "real_fp_rate": round(fp / n_real, 4) if n_real else None,
        "ai_hit": hit,
        "ai_n": n_ai,
        "ai_recall": round(hit / n_ai, 4) if n_ai else None,
    }


def build_sweep(real_rows, ai_rows):
    """0→1 扫描；额外把生产用的几个阈值精确纳入，避免浮点漏掉。"""
    grid = [round(i * SWEEP_STEP, 4) for i in range(0, 101)]
    for extra in (BASELINE_SUSP, BASELINE_HIGH):
        grid.append(round(extra, 4))
    seen, out = set(), []
    for t in sorted(grid):
        if t in seen:
            continue
        seen.add(t)
        out.append(evaluate(real_rows, ai_rows, t))
    return out


def classify(rows, base_recall, base_fp):
    """按 reviewer 的采纳规则给每个工作点打标签。

    规则（P1-3 验收）：只有留出集改善、且没有明显增加真图高风险误报时才采纳。
    这里细分两种「改善」：
        better_recall —— 召回变高，且误报没有变差
        better_fp     —— 误报变低，且召回没有变差
    """
    for r in rows:
        rec, fp = r["ai_recall"], r["real_fp_rate"]
        r["better_recall"] = bool(rec is not None and rec > base_recall and fp <= base_fp)
        r["better_fp"] = bool(fp is not None and fp < base_fp and rec >= base_recall)
        r["dominated_by_baseline"] = bool(
            rec is not None and fp is not None and rec <= base_recall and fp >= base_fp
        )
    return rows


def main():
    t0 = time.time()
    if not EVAL.exists():
        print(f"[错误] 找不到正式评测结果：{EVAL}")
        print("       请先跑 python tools/run_dataset_closedloop.py --require-full-coverage")
        return 1

    real_rows, real_meta = load_scores(EVAL, REAL_SET)
    ai_rows, ai_meta = load_scores(EVAL, AI_SET)
    if not real_rows or not ai_rows:
        print("[错误] 冻结集里没有可用的 aigc 分数，无法扫描阈值")
        return 1

    eval_data = json.loads(EVAL.read_text(encoding="utf-8"))
    base = evaluate(real_rows, ai_rows, BASELINE_HIGH)
    base_recall, base_fp = base["ai_recall"], base["real_fp_rate"]

    sweep = classify(build_sweep(real_rows, ai_rows), base_recall, base_fp)

    better_recall_pts = [r for r in sweep if r["better_recall"]]
    better_fp_pts = [r for r in sweep if r["better_fp"]]

    # 真实图 / AI 图 的分数分布（解释为什么阈值调不动）
    real_scores = sorted(s for _f, s in real_rows)
    ai_scores = sorted(s for _f, s in ai_rows)
    band = None
    if better_fp_pts:
        lo = min(r["threshold"] for r in better_fp_pts)
        hi = max(r["threshold"] for r in better_fp_pts)
        band = {"lo": lo, "hi": hi, "width": round(hi - lo, 4), "n_points": len(better_fp_pts)}

    # 结论：按 reviewer 规则判断是否采纳
    if better_recall_pts:
        verdict = "ADOPT_RECALL"
        verdict_text = (
            f"存在 {len(better_recall_pts)} 个阈值能在不恶化误报的前提下提升召回，"
            f"可考虑采纳（需在冻结集外复核后再上线）。"
        )
    elif better_fp_pts:
        verdict = "FOUND_BUT_FRAGILE"
        verdict_text = (
            f"没有任何阈值能提升召回；但有 {len(better_fp_pts)} 个阈值能在不损失召回的前提下"
            f"把真实图高风险误报从 {base_fp} 降到 0（阈值区间 {band['lo']}–{band['hi']}，宽度仅 {band['width']}）。"
            f"该区间是靠「卡掉那一张 {max(real_scores):.4f} 的误报样本」得到的，属对单样本的贴合，"
            f"且抬高阈值会压缩未见生成器（分数可能落在 0.9x）的判出空间、反伤召回。"
            f"按 reviewer「不显著改善不替换」的口径，本轮**不改动生产阈值**，仅记录该边界。"
        )
    else:
        verdict = "NO_IMPROVEMENT"
        verdict_text = "不存在任何比生产阈值更优的工作点：阈值这个变量已无可调余量。"

    result = {
        "schema": "beautyproof/threshold_sweep@1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": {
            "changed_variable": "threshold",
            "unchanged": ["model_weights", "training_data", "preprocessing", "evidence_fusion"],
            "note": "本轮只改阈值（reviewer P1-3 要求的单变量对照）",
        },
        "model_version": "beautyproof_aigen_v4",
        "data_version": {
            "eval_result_file": str(EVAL.relative_to(REPO)),
            "eval_generated_at": eval_data.get("generated_at"),
            "real_set": REAL_SET,
            "ai_set": AI_SET,
            "real_meta": real_meta,
            "ai_meta": ai_meta,
            "note": "受控集不参与阈值决策（reviewer：受控仅作链路检查）",
        },
        "thresholds": {
            "aigc_high_risk": BASELINE_HIGH,
            "aigc_hard_high": BASELINE_HARD,
        },
        "run_command": "python tools/threshold_sweep.py",
        "baseline_at_high_risk": base,
        "score_distribution": {
            "real_scores": real_scores,
            "ai_scores": ai_scores,
            "real_max": max(real_scores),
            "ai_min_among_detected": min([s for s in ai_scores if s >= BASELINE_HIGH], default=None),
        },
        "findings": {
            "n_thresholds_swept": len(sweep),
            "better_recall_points": better_recall_pts,
            "better_fp_points_count": len(better_fp_pts),
            "zero_fp_band": band,
        },
        "verdict": verdict,
        "verdict_text": verdict_text,
        "elapsed_sec": round(time.time() - t0, 2),
        "sweep": sweep,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 终端摘要
    print(f"模型 {result['model_version']} · 冻结集 {REAL_SET}({len(real_rows)}) / {AI_SET}({len(ai_rows)})")
    print(f"生产阈值 high_risk={BASELINE_HIGH} → 召回 {base_recall} · 真实误报率 {base_fp}")
    print(f"扫描 {len(sweep)} 个阈值：能提召回的 {len(better_recall_pts)} 个，能降误报的 {len(better_fp_pts)} 个")
    print(f"真实图分数 max={max(real_scores):.4f} · AI 图被判出的最低分="
          f"{result['score_distribution']['ai_min_among_detected']}")
    print(f"\n判定：{verdict}\n{verdict_text}")
    print(f"\n已写入：{OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

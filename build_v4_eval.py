# -*- coding: utf-8 -*-
"""构造 results/v4_eval.json：用 v4 已生效状态重跑的 6 案例 + 与 v3 基线对比。
读取 batch_run 产出的 results/dataset_eval.json（本次 v4 结果），抽取每个案例的
aigc_score / trufor_score 作为 key_scores，并与 results/dataset_eval_baseline_v3.json 对比。
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
OUT = RESULTS / "v4_eval.json"
BASELINE = RESULTS / "dataset_eval_baseline_v3.json"
V4 = RESULTS / "dataset_eval.json"


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def get_key_scores(stem):
    """从 outputs/ 证据文件抽取 aigc_score 与 trufor_score。"""
    ks = {"aigc_model_version": "v4", "aigc_model": "beautyproof_aigen_v4"}
    aigc_f = REPO / "outputs" / f"aigc_{stem}.json"
    if aigc_f.exists():
        d = load_json(aigc_f)
        ev = d.get("evidence", [{}])
        ks["aigc_score"] = ev[0].get("aigc_score") if ev else None
    tru_f = REPO / "outputs" / f"trufor_{stem}.json"
    if tru_f.exists():
        d = load_json(tru_f)
        ev = d.get("evidence", [{}])
        ks["trufor_score"] = ev[0].get("trufor_score") if ev else None
        ks["trufor_area_ratio"] = ev[0].get("tampered_area_ratio") if ev else None
    return ks


def main():
    v4 = load_json(V4)
    base = load_json(BASELINE)
    base_by_id = {r["id"]: r for r in base["results"]}

    out_results = []
    for r in v4["results"]:
        stem = r["id"]
        item = {
            "id": r["id"],
            "ground_truth": r["ground_truth"],
            "expected_risk": r.get("expected_risk"),
            "actual_risk": r["actual_risk"],
            "key_scores": get_key_scores(stem),
            "reasons": r.get("reasons", []),
            "tools_used": r.get("tools_used", []),
            "hit": r.get("hit"),
            "exact": r.get("exact"),
        }
        out_results.append(item)

    # ---- baseline_compare：逐项 v4 vs v3 ----
    compare = {}
    issues = []

    # ① clean 不误报：aigc_score < 0.5
    for cid in ("clean_01", "clean_02_recompressed"):
        v4k = next(x for x in out_results if x["id"] == cid)["key_scores"]
        b = base_by_id.get(cid, {})
        # 从 baseline reasons 里取 aigc_score
        b_aigc = None
        for line in b.get("reasons", []):
            if "AI 生成概率" in line:
                try:
                    b_aigc = float(line.split("概率")[1].split("（")[0].strip())
                except Exception:
                    b_aigc = None
        v4_aigc = v4k.get("aigc_score")
        ok = (v4_aigc is not None and v4_aigc < 0.5)
        compare[cid] = {
            "check": "clean 不误报 (aigc_score < 0.5)",
            "v4_aigc_score": v4_aigc,
            "v3_aigc_score": b_aigc,
            "pass": bool(ok),
        }
        if not ok:
            issues.append(f"{cid}: v4 aigc_score={v4_aigc} 未 <0.5，出现误报退化")

    # ② 篡改召回：trufor_score >= 0.5
    for cid in ("tampered_01_copy_move", "tampered_02_splice", "tampered_03_text_edit"):
        v4k = next(x for x in out_results if x["id"] == cid)["key_scores"]
        b = base_by_id.get(cid, {})
        b_tru = None
        for line in b.get("reasons", []):
            if "TruFor" in line:
                try:
                    b_tru = float(line.split("篡改分数")[1].split("（")[0].strip())
                except Exception:
                    b_tru = None
        v4_tru = v4k.get("trufor_score")
        ok = (v4_tru is not None and v4_tru >= 0.5)
        compare[cid] = {
            "check": "篡改召回 (trufor_score >= 0.5)",
            "v4_trufor_score": v4_tru,
            "v3_trufor_score": b_tru,
            "pass": bool(ok),
        }
        if not ok:
            issues.append(f"{cid}: v4 trufor_score={v4_tru} < 0.5，篡改召回退化")

    # ③ ai_17 仍 high_risk
    cid = "ai_17"
    v4r = next(x for x in out_results if x["id"] == cid)["actual_risk"]
    v4k = next(x for x in out_results if x["id"] == cid)["key_scores"]
    b = base_by_id.get(cid, {})
    ok = (v4r == "high_risk")
    compare[cid] = {
        "check": "ai_17 仍 high_risk",
        "v4_actual_risk": v4r,
        "v4_aigc_score": v4k.get("aigc_score"),
        "v3_actual_risk": b.get("actual_risk"),
        "pass": bool(ok),
    }
    if not ok:
        issues.append(f"{cid}: v4 actual_risk={v4r}，未达 high_risk，退化")

    regressed = len(issues) > 0
    summary = {
        "v4_model_loaded": "beautyproof_aigen_v4",
        "cases_total": len(out_results),
        "cases_hit": sum(1 for x in out_results if x.get("hit")),
        "cases_exact": sum(1 for x in out_results if x.get("exact")),
        "regression": regressed,
        "issues": issues,
        "conclusion": "v4 与 v3 基线相比无退化：clean 不误报、篡改召回保持、ai_17 仍 high_risk"
                     if not regressed else "v4 相对 v3 存在退化，见 issues",
    }

    out = {
        "results": out_results,
        "baseline_compare": compare,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written:", OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

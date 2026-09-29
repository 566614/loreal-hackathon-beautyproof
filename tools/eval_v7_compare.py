# -*- coding: utf-8 -*-
# 来源：原创 —— v4 vs v7 冻结集单变量对比评测（reviewer P1-3 晋升判定）
"""在冻结测试集上并排评测 v4（生产）与 v7（候选），产出晋升判定。

冻结集（全部不参与 v7 训练）：
  真实类：data/realworld_heldout（10，永不训练）
          data/real_filter_selfie heldout 段（10，SPLIT_MANIFEST 冻结，永不训练）
  AI 类： data/ai_cross_native（12，零样本）、data/ai_mj_sd_flux_heldout
  受控回归：data/ai（20）、data/clean（2）

晋升口径（reviewer P1-3）：held-out 改善 且 真实 high_risk 误报不升。
防呆：先 pin MODEL_PRIORITY 再断言 model_path() 解析到目标模型（防静默回退）。
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results" / "v7_eval.json"
FS_SPLIT = REPO / "data" / "real_filter_selfie" / "SPLIT_MANIFEST.json"
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def collect(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS) if folder.exists() else []


def score_all(model, files):
    import aigc_tool
    aigc_tool.MODEL_PRIORITY = [model]
    resolved = aigc_tool.model_path()
    assert resolved and Path(resolved).name == model, \
        f"模型 pin 失败：要评 {model}，实际解析到 {resolved}"
    rows = []
    for p in files:
        s, lab, ok = aigc_tool.run_aigc(p)
        rows.append({"file": p.name, "score": s if ok else None})
    return rows


def summarize(rows, kind):
    import rule_engine
    high = rule_engine.THRESHOLDS["aigc_high_risk"]
    pos = 0.6
    scores = [r["score"] for r in rows if r["score"] is not None]
    n = len(scores)
    out = {"n": n, "n_unavailable": len(rows) - n,
           "mean": round(sum(scores) / n, 4) if n else None,
           "max": max(scores) if n else None,
           "n_ge_high": sum(1 for s in scores if s >= high),
           "n_ge_pos": sum(1 for s in scores if s >= pos)}
    if kind == "real":
        out["fp_rate_high"] = round(out["n_ge_high"] / n, 4) if n else None
        out["fp_rate_pos"] = round(out["n_ge_pos"] / n, 4) if n else None
    else:
        out["recall_high"] = round(out["n_ge_high"] / n, 4) if n else None
        out["recall_pos"] = round(out["n_ge_pos"] / n, 4) if n else None
    return out


def main():
    fs = json.loads(FS_SPLIT.read_text(encoding="utf-8"))
    fs_heldout = [REPO / "data" / "real_filter_selfie" / f for f in fs["heldout_never_train"]]

    sets = [
        ("realworld_heldout", "real", collect(REPO / "data" / "realworld_heldout")),
        ("filter_selfie_heldout", "real", fs_heldout),
        ("ai_cross_native", "ai", collect(REPO / "data" / "ai_cross_native")),
        ("ai_mj_sd_flux_heldout", "ai", collect(REPO / "data" / "ai_mj_sd_flux_heldout")),
        ("controlled_ai", "ai", collect(REPO / "data" / "ai")),
        ("controlled_clean", "real", collect(REPO / "data" / "clean")),
    ]

    report = {"schema": "beautyproof/v7_compare@1",
              "single_variable": "v7 = v4 配方 + real_filter_selfie train 段 20 张（唯一变量）；"
                                 "全部冻结集不参与 v7 训练",
              "promotion_rule": "filter_selfie_heldout 改善 且 realworld_heldout high_risk 误报不升 "
                                "且 ai_cross_native 召回不降 且 controlled_ai 召回不降",
              "sets": {}}
    detail = {}
    for model in ["beautyproof_aigen_v4", "beautyproof_aigen_v7"]:
        detail[model] = {}
        for name, kind, files in sets:
            if not files:
                continue
            rows = score_all(model, files)
            detail[model][name] = {"kind": kind, "rows": rows, "summary": summarize(rows, kind)}
            print(f"{model:24s} {name:24s} {detail[model][name]['summary']}", flush=True)

    for name, kind, _files in sets:
        s4 = detail["beautyproof_aigen_v4"].get(name, {}).get("summary")
        s7 = detail["beautyproof_aigen_v7"].get(name, {}).get("summary")
        if not s4 or not s7:
            continue
        entry = {"kind": kind, "v4": s4, "v7": s7}
        if kind == "real":
            entry["delta_fp_high"] = round((s7["fp_rate_high"] or 0) - (s4["fp_rate_high"] or 0), 4)
        else:
            entry["delta_recall_high"] = round((s7["recall_high"] or 0) - (s4["recall_high"] or 0), 4)
        report["sets"][name] = entry

    s4 = {k: v["summary"] for k, v in detail["beautyproof_aigen_v4"].items()}
    s7 = {k: v["summary"] for k, v in detail["beautyproof_aigen_v7"].items()}
    checks = {
        "filter_selfie_improved": (s7["filter_selfie_heldout"]["fp_rate_high"]
                                   < s4["filter_selfie_heldout"]["fp_rate_high"]),
        "realworld_fp_not_worse": (s7["realworld_heldout"]["fp_rate_high"]
                                   <= s4["realworld_heldout"]["fp_rate_high"]),
        "crossgen_not_worse": (s7["ai_cross_native"]["recall_high"]
                               >= s4["ai_cross_native"]["recall_high"]),
        "controlled_ai_not_worse": (s7["controlled_ai"]["recall_high"]
                                    >= s4["controlled_ai"]["recall_high"]),
    }
    checks["all_pass"] = all(checks.values())
    report["promotion_checks"] = checks
    report["decision"] = ("ADOPT_V7：把 aigc_tool.MODEL_PRIORITY 置顶 v7 并重跑闭环评测"
                          if checks["all_pass"] else
                          "KEEP_V4：不满足晋升口径，v7 仅诚实归档（参照 v5/v6 先例）")
    report["per_file"] = {m: {k: v["rows"] for k, v in d.items()} for m, d in detail.items()}

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 晋升判定 ===", flush=True)
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"  decision: {report['decision']}", flush=True)
    print(f"已写 {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

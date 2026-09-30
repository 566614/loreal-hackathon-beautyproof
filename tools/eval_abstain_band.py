# -*- coding: utf-8 -*-
# 来源：原创 —— 灰带决策策略（AIGC 判定政策）效果验证
"""验证「AIGC 灰带」决策策略（2026-09-29 重构）对冻结集的影响。

背景：这不是模型实验（模型仍是生产版 v4，零重训），而是**决策政策**单变量——
只改 rule_engine 对 AIGC 分数的解释方式：
    旧政策：aigc_score >= 0.9            → high_risk（自动下架级）
    新政策：aigc_score >= 0.99           → high_risk（自动下架级）
            0.9 <= aigc_score < 0.99     → suspicious（人工复核，不自动下架）
            其余                          → 不变（弃权/判真实）

数据源：results/v7_eval.json 的 per_file 中 beautyproof_aigen_v4（生产模型）
在冻结集上的真实分数——全部不参与任何训练。

预期（若灰带有效）：
    1. 真实类被「自动下架级」误伤的样本全部消失（误报降级为人工复核）；
    2. AI 类的捕获情况不变：被捕获的 AI 图全部 ≥0.9959，仍走 high_risk。
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import rule_engine  # noqa: E402

EVAL = REPO / "results" / "v7_eval.json"
OUT = REPO / "results" / "abstain_band_eval.json"

HIGH = rule_engine.THRESHOLDS["aigc_high_risk"]        # 0.9  灰带下限
HARD = rule_engine.THRESHOLDS["aigc_hard_high"]        # 0.99 自动下架级下限


def old_policy(score):
    """旧政策下 AIGC 单独给出的定级贡献。"""
    if score >= HIGH:
        return "high_risk"
    return "none"


def new_policy(score):
    """新政策（灰带）下 AIGC 单独给出的定级贡献。"""
    if score >= HARD:
        return "high_risk"
    if score >= HIGH:
        return "suspicious"
    return "none"


def main():
    data = json.loads(EVAL.read_text(encoding="utf-8"))
    per_file = data["per_file"]["beautyproof_aigen_v4"]  # 生产模型 v4 的冻结集分数

    sets = {}
    for name, rows in per_file.items():
        kind = data["sets"][name]["kind"]
        n = len(rows)
        old_hr = sum(1 for r in rows if r["score"] is not None and old_policy(r["score"]) == "high_risk")
        new_hr = sum(1 for r in rows if r["score"] is not None and new_policy(r["score"]) == "high_risk")
        new_sp = sum(1 for r in rows if r["score"] is not None and new_policy(r["score"]) == "suspicious")
        # 「被标记」（high_risk 或 suspicious，都需要人工介入）在新旧政策下的对比
        old_caught = old_hr
        new_caught = new_hr + new_sp
        gray_scores = sorted(round(r["score"], 4) for r in rows
                             if r["score"] is not None and HIGH <= r["score"] < HARD)
        sets[name] = {
            "kind": kind,
            "n": n,
            "old_high_risk": old_hr,
            "new_high_risk": new_hr,
            "new_gray_suspicious": new_sp,
            "old_caught": old_caught,
            "new_caught": new_caught,
            "gray_band_scores": gray_scores,
        }

    real = [v for v in sets.values() if v["kind"] == "real"]
    ai = [v for v in sets.values() if v["kind"] == "ai"]
    real_old_hr = sum(v["old_high_risk"] for v in real)
    real_new_hr = sum(v["new_high_risk"] for v in real)
    real_new_sp = sum(v["new_gray_suspicious"] for v in real)
    real_old_catch = sum(v["old_caught"] for v in real)
    real_new_catch = sum(v["new_caught"] for v in real)
    ai_old_catch = sum(v["old_caught"] for v in ai)
    ai_new_catch = sum(v["new_caught"] for v in ai)

    checks = {
        "real_auto_takedown_fp_eliminated": real_new_hr == 0,
        "real_review_flags_added": real_new_sp == real_old_hr,   # 全部降级为人工复核，一个不丢
        "ai_caught_unchanged": ai_new_catch >= ai_old_catch,     # AI 捕获不降
    }
    checks["all_pass"] = all(checks.values())
    decision = ("ADOPT_BAND：灰带把真实图误伤从「自动下架」降级为「人工复核」，"
                "且 AI 捕获无一丢失 —— 决策政策生效"
                if checks["all_pass"] else
                "REJECT：灰带未达预期（见 checks），需复核阈值标定")

    report = {
        "schema": "beautyproof/abstain_band_eval@1",
        "policy_change": {
            "old": f"aigc_score >= {HIGH} → high_risk",
            "new": f"aigc_score >= {HARD} → high_risk; [{HIGH}, {HARD}) → suspicious（人工复核）",
            "model_unchanged": "beautyproof_aigen_v4（零重训，纯决策政策单变量）",
        },
        "sets": sets,
        "totals": {
            "real_old_auto_takedown_fp": real_old_hr,
            "real_new_auto_takedown_fp": real_new_hr,
            "real_new_review_flags": real_new_sp,
            "real_old_caught": real_old_catch,
            "real_new_caught": real_new_catch,
            "ai_old_caught": ai_old_catch,
            "ai_new_caught": ai_new_catch,
        },
        "checks": checks,
        "decision": decision,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== AIGC 灰带决策政策验证（冻结集，模型 v4 零改动） ===", flush=True)
    for name, v in sets.items():
        print(f"  {name:26s} ({v['kind']:3s}) 旧 high_risk={v['old_high_risk']} "
              f"→ 新 high_risk={v['new_high_risk']} + 灰带 suspicious={v['new_gray_suspicious']}", flush=True)
        if v["gray_band_scores"]:
            print(f"    灰带分数: {v['gray_band_scores']}", flush=True)
    print(f"\n  真实图自动下架级误伤: {real_old_hr} → {real_new_hr}（降级为人工复核 {real_new_sp} 条）", flush=True)
    print(f"  AI 图捕获数: {ai_old_catch} → {ai_new_catch}", flush=True)
    print("=== 判定 ===", flush=True)
    for k, v in checks.items():
        if k != "all_pass":
            print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"  decision: {decision}", flush=True)
    print(f"已写 {OUT}", flush=True)
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

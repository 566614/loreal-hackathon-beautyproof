# -*- coding: utf-8 -*-
"""
文案侧评测 —— 跑 data/text_cases.json，算方向命中率，写 results/text_eval.json

用法：
    python tools/eval_text.py

大白话：
    图像侧我们有 6/6 的命中率，文案侧刚做出来，也得拿同一把尺子量一遍。
    每条 case 都是「文案 + 一张图 + 人工标注的期望结果」，
    跑完对答案，看哪些对了、哪些错了、错在哪。

为什么必须包含防误报样本：
    只放「该报警的」样本，命中率能刷到 100%，但那叫自欺欺人。
    tc_01 / tc_04 / tc_06 / tc_08 是故意放进去的「不该报警」样本 ——
    它们跑错了才是真问题：说明系统见谁都咬。
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import pipeline  # noqa: E402
import rule_engine  # noqa: E402


def run_case(case):
    stem = case["image_stem"]
    text = case["text"]

    pipeline.run_text_tools(stem, text, verbose=False)
    ev = rule_engine.load_evidence(REPO, stem)

    t0 = (ev.get("text", {}).get("evidence") or [{}])[0]
    c0 = (ev.get("crossmodal", {}).get("evidence") or [{}])[0]

    claim_high = t0.get("claim_high_count", 0)
    claim_medium = t0.get("claim_medium_count", 0)
    conflict = c0.get("conflict_level")

    risk, reasons = rule_engine.judge(ev)

    checks = {
        "claim_high_ok": claim_high >= case["expect_claim_high"],
        "claim_medium_ok": claim_medium >= case["expect_claim_medium"],
        "conflict_ok": conflict == case["expect_conflict"],
        "risk_ok": risk == case["expected_risk"],
    }
    return {
        "id": case["id"],
        "tests": case["tests"],
        "image_stem": stem,
        "text_preview": text[:40] + ("…" if len(text) > 40 else ""),
        "observed": {
            "claim_high": claim_high,
            "claim_medium": claim_medium,
            "conflict_level": conflict,
            "risk_level": risk,
        },
        "expected": {
            "claim_high_min": case["expect_claim_high"],
            "claim_medium_min": case["expect_claim_medium"],
            "conflict_level": case["expect_conflict"],
            "risk_level": case["expected_risk"],
        },
        "checks": checks,
        "all_ok": all(checks.values()),
        "reasons": reasons[:2],
    }


def main():
    cases = json.loads((REPO / "data" / "text_cases.json").read_text(encoding="utf-8"))["cases"]

    rows = [run_case(c) for c in cases]
    n = len(rows)

    def rate(key):
        return round(sum(1 for r in rows if r["checks"][key]) / n, 4)

    summary = {
        "total_cases": n,
        "risk_direction_hit": sum(1 for r in rows if r["checks"]["risk_ok"]),
        "risk_direction_rate": rate("risk_ok"),
        "claim_high_rate": rate("claim_high_ok"),
        "claim_medium_rate": rate("claim_medium_ok"),
        "conflict_rate": rate("conflict_ok"),
        "fully_passed": sum(1 for r in rows if r["all_ok"]),
        # 防误报样本跑错 = 系统见谁都咬，这类错误比漏报更严重，单独列出来
        "false_alarm_cases": [r["id"] for r in rows
                              if r["expected"]["risk_level"] == "inconclusive"
                              and not r["checks"]["risk_ok"]],
    }

    out = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "cases": rows,
    }
    (REPO / "results" / "text_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"文案侧评测：{summary['risk_direction_hit']}/{n} 方向命中 "
          f"({summary['risk_direction_rate']:.0%})")
    print(f"  违禁宣称高危  {summary['claim_high_rate']:.0%}")
    print(f"  违禁宣称中危  {summary['claim_medium_rate']:.0%}")
    print(f"  图文冲突判定  {summary['conflict_rate']:.0%}")
    print()
    for r in rows:
        flag = "OK " if r["all_ok"] else "!! "
        print(f"  {flag}{r['id']:22s} 风险={r['observed']['risk_level']:13s} "
              f"冲突={str(r['observed']['conflict_level']):15s} "
              f"宣称(高/中)={r['observed']['claim_high']}/{r['observed']['claim_medium']}")
        if not r["all_ok"]:
            print(f"      期望 风险={r['expected']['risk_level']} "
                  f"冲突={r['expected']['conflict_level']} "
                  f"宣称≥{r['expected']['claim_high_min']}/{r['expected']['claim_medium_min']}")
    print(f"\n已保存到: results/text_eval.json")
    return 0 if summary["risk_direction_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Replay frozen observations with reviewer labels; never execute tools or models."""

import argparse
import hashlib
import json
from pathlib import Path

from review_harness import ACTIONS, decide


def load_cases(paths):
    cases = []
    for path in paths:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(value, dict) and "cases" in value:
            value = value["cases"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("样本文件须为导出的单条样本、样本数组或包含 cases 的数据包")
        cases.extend(value)
    validate_cases(cases)
    return sorted(cases, key=lambda case: case["sample_id"])


def validate_cases(cases):
    ids, groups = set(), {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"sample_id", "group", "split", "kind", "observation", "feedback"}:
            raise ValueError("样本须含 sample_id、group、split、kind、observation、feedback，导出时请移除额外字段")
        for key in ("sample_id", "group", "kind"):
            if not isinstance(case[key], str) or not case[key].strip() or case[key] != case[key].strip():
                raise ValueError(f"{key} 须为非空文本")
        if case["sample_id"] in ids:
            raise ValueError("样本编号重复")
        ids.add(case["sample_id"])
        if case["split"] not in ("development", "holdout"):
            raise ValueError("split 须为 development 或 holdout")
        previous = groups.setdefault(case["group"], case["split"])
        if previous != case["split"]:
            raise ValueError("同来源组不可跨开发集与留出集")
        feedback = case["feedback"]
        if not isinstance(feedback, dict) or set(feedback) != {"correct_action", "reason"}:
            raise ValueError("feedback 须含 correct_action 与 reason")
        if (not isinstance(feedback["correct_action"], str) or feedback["correct_action"] not in ACTIONS
                or not isinstance(feedback["reason"], str) or not feedback["reason"].strip()):
            raise ValueError("反馈须含允许的动作与审核员纠正理由")
        observation = case["observation"]
        keys = {"market_known", "has_image", "uncertain_ocr_count", "ocr_status", "context_conflict", "followup_used", "claims"}
        if not isinstance(observation, dict) or set(observation) != keys:
            raise ValueError("观测字段须与 Harness 导出契约一致，禁止附带原文和图片")
        if any(type(observation[key]) is not bool for key in ("market_known", "has_image", "context_conflict", "followup_used")):
            raise ValueError("观测状态须为布尔值")
        count = observation["uncertain_ocr_count"]
        if type(count) is not int or count < 0:
            raise ValueError("uncertain_ocr_count 须为非负整数")
        if observation["ocr_status"] not in ("completed", "insufficient_evidence", "model_unavailable"):
            raise ValueError("OCR 状态无效")
        if not isinstance(observation["claims"], list):
            raise ValueError("claims 须为数组")
        evidence_ids = set()
        for claim in observation["claims"]:
            if not isinstance(claim, dict) or set(claim) != {"id", "source", "quantified", "product_known", "version_known", "status"}:
                raise ValueError("宣称观测须使用 Harness 导出字段")
            if not isinstance(claim["id"], str) or not claim["id"] or claim["id"] in evidence_ids:
                raise ValueError("宣称证据编号须为唯一非空文本")
            evidence_ids.add(claim["id"])
            if claim["source"] not in ("body", "image") or claim["status"] not in ("证据不足", "存在冲突", "有证据支持"):
                raise ValueError("宣称来源或状态无效")
            if any(type(claim[key]) is not bool for key in ("quantified", "product_known", "version_known")):
                raise ValueError("宣称观测状态须为布尔值")
    for split in ("development", "holdout"):
        if sum(case["split"] == split for case in cases) < 6:
            raise ValueError(f"{split} 至少需要 6 条已标注样本")


def compare(cases):
    rows = []
    for case in cases:
        baseline = decide(case["observation"], "v0")
        candidate = decide(case["observation"], "v1")
        expected = case["feedback"]["correct_action"]
        base_hit, candidate_hit = baseline["action"] == expected, candidate["action"] == expected
        rows.append({
            "sample_id": case["sample_id"], "group": case["group"], "kind": case["kind"],
            "correct_action": expected, "reviewer_reason": case["feedback"]["reason"],
            "v0": baseline, "v1": candidate, "v0_correct": base_hit, "v1_correct": candidate_hit,
            "improved": candidate_hit and not base_hit, "regressed": base_hit and not candidate_hit,
        })
    return {
        "count": len(rows), "v0_hits": sum(row["v0_correct"] for row in rows),
        "v1_hits": sum(row["v1_correct"] for row in rows),
        "improved": [row["sample_id"] for row in rows if row["improved"]],
        "regressed": [row["sample_id"] for row in rows if row["regressed"]], "cases": rows,
    }


def replay(cases):
    validate_cases(cases)
    cases = sorted(cases, key=lambda case: case["sample_id"])
    # Candidate selection reads only development observations and feedback.
    development = compare([case for case in cases if case["split"] == "development"])
    selected = "v1" if development["improved"] and not development["regressed"] else "v0"
    holdout = compare([case for case in cases if case["split"] == "holdout"])
    reasons = []
    if selected != "v1":
        reasons.append("开发集未选出有改进且无新增错误的候选策略")
    if not holdout["improved"]:
        reasons.append("冻结留出集没有改对至少一例")
    if holdout["regressed"]:
        reasons.append("冻结留出集出现新增错误动作")
    fingerprint = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "schema": "beautyproof/harness-replay@1", "dataset_sha256": fingerprint,
        "baseline_policy": "v0", "selected_on_development": selected,
        "development": development, "holdout": holdout,
        "recommendation": "suggest_human_adoption" if not reasons else "keep_baseline",
        "reasons": reasons or ["开发集选出 v1，留出集至少改对一例且无新增错误，建议人工复核采用"],
        "execution": {"tools_executed": 0, "models_called": 0, "verdicts_modified": False},
        "evaluation_scope": "取证动作与审核员标签的一致性；原始证据、功效结论和模型权重由原审核链路负责。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=Path, default=[Path(__file__).resolve().parents[1] / "data/harness_replay.json"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = replay(load_cases(args.cases))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"{result['recommendation']} ({result['dataset_sha256']})")
    else:
        print(content)


if __name__ == "__main__":
    main()

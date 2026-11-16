"""Compare expanded data against the v2 training pool using locked product groups."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from doubao_damage import DETAIL_SHORT_SIDE, MIN_SHORT_SIDE
from train_doubao_damage import (Candidate, EXTENDED, RECALL_TARGET, SCHEMA, by_condition, choose_threshold,
                                 grouped_folds, read_pairs, score_rows, train_head)

CANDIDATE_NAMES = (
    "relative-color", "relative-residuals", "relative-spectrum", "relative-contour",
    "relative-residuals-color", "relative-residuals-shape", "relative-all", "boost-residuals",
    "boost-contour", "boost-all",
)
CANDIDATES = {candidate.name: candidate for candidate in EXTENDED if candidate.name in CANDIDATE_NAMES}
BASELINES = {mode: next(candidate for candidate in EXTENDED if candidate.name == name)
             for mode, name in [("whole", "relative-color"), ("region", "auxiliary-residuals")]}


def read_cohorts(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["ai_file"]: row["cohort"] for row in csv.DictReader(stream)}


def split_locked(samples, anchors):
    """Keep v2's five held-out folds and assign any new group to one fold."""
    lookup = {}
    sizes = [0] * 5
    for index, (_, held) in enumerate(grouped_folds(anchors, 5)):
        for pair in held:
            lookup[pair.group] = index
    for pair in samples:
        if pair.group in lookup:
            sizes[lookup[pair.group]] += 1
    counts = Counter(pair.group for pair in samples if pair.group not in lookup)
    for group in sorted(counts, key=lambda key: (-counts[key], key)):
        index = min(range(5), key=lambda i: sizes[i])
        lookup[group] = index
        sizes[index] += counts[group]
    return [([pair for pair in samples if lookup[pair.group] != index],
             [pair for pair in samples if lookup[pair.group] == index]) for index in range(5)]


def calibrate(data, mode, candidate, auxiliary=()):
    predictions = []
    for train, held in grouped_folds(data, 4):
        model = train_head(train, mode, candidate, auxiliary)
        predictions.extend(score_rows(model, held, mode))
    threshold = choose_threshold(predictions)
    return threshold, by_condition(predictions, threshold=threshold)


def rank(conditions):
    aucs = [view["auc"] for view in conditions.values()]
    review_rates = [view["false_positive_rate"] for view in conditions.values()]
    return min(aucs) + 0.20 * (1 - np.mean(review_rates)), np.mean(aucs)


def select(data, mode, candidates):
    entries = []
    for candidate in candidates.values():
        threshold, results = calibrate(data, mode, candidate)
        entries.append((rank(results), candidate, threshold, results))
    _, best, threshold, _ = max(entries, key=lambda entry: entry[0])
    return best, threshold, {entry[1].name: entry[3] for entry in entries}


def summarized(rows, cohorts, which):
    return by_condition([row for row in rows if cohorts[row["pair"]] in which])


def evaluate(pairs, cohorts, mode, candidates):
    data = [pair for pair in pairs if pair.annotation == "damage"
            and all(value is not None for value in pair.views["matched"][mode])]
    original = [pair for pair in data if cohorts[pair.name] == "original_high"]
    v2_auxiliary = [pair for pair in data if cohorts[pair.name] == "earlier_review"]
    locked = split_locked(data, original)
    baseline_records, expanded_records, choices = [], [], []
    for index, (train, held) in enumerate(locked, 1):
        allowed = {pair.group for pair in train}
        baseline_train = [pair for pair in original if pair.group in allowed]
        baseline_aux = [pair for pair in v2_auxiliary if pair.group in allowed]
        candidate = BASELINES[mode]
        threshold, _ = calibrate(baseline_train, mode, candidate, baseline_aux)
        baseline = train_head(baseline_train, mode, candidate, baseline_aux)
        rows = score_rows(baseline, held, mode, threshold=threshold)
        baseline_records.extend({**row, "fold": index, "method": "v2_pool", "mode": mode} for row in rows)
        candidate, threshold, _ = select(train, mode, candidates)
        head = train_head(train, mode, candidate)
        rows = score_rows(head, held, mode, threshold=threshold)
        expanded_records.extend({**row, "fold": index, "method": "expanded", "mode": mode} for row in rows)
        choices.append(candidate.name)
    candidate, cut, comparison = select(data, mode, candidates)
    final = train_head(data, mode, candidate)
    final["threshold"] = cut
    transfer, baseline_transfer = {}, {}
    for batch in sorted({pair.batch for pair in data}):
        held = [pair for pair in data if pair.batch == batch]
        seen = {pair.group for pair in held}
        train = [pair for pair in data if pair.batch != batch and pair.group not in seen]
        cross_batch_train = [pair for pair in original if pair.batch != batch and pair.group not in seen]
        if len({pair.group for pair in cross_batch_train}) >= 4:
            cross_batch_aux = [pair for pair in v2_auxiliary if pair.batch != batch and pair.group not in seen]
            previous = BASELINES[mode]
            threshold, _ = calibrate(cross_batch_train, mode, previous, cross_batch_aux)
            baseline = train_head(cross_batch_train, mode, previous, cross_batch_aux)
            baseline_transfer[batch] = by_condition(score_rows(baseline, held, mode, threshold=threshold))
        if len({pair.group for pair in train}) >= 5:
            variant, threshold, _ = select(train, mode, candidates)
            transfer[batch] = by_condition(score_rows(train_head(train, mode, variant), held, mode, threshold=threshold))
    scopes = {"original": {"original_high"}, "additional": {"earlier_review", "new_review"},
              "all": {"original_high", "earlier_review", "new_review"}}
    comparison_by_scope = {
        name: {"v2_pool": summarized(baseline_records, cohorts, wanted),
               "expanded": summarized(expanded_records, cohorts, wanted)}
        for name, wanted in scopes.items()
    }
    summary = {"training_count": len(data), "groups": len({pair.group for pair in data}), "final": candidate.name,
               "outer_fold_choices": choices, "candidate_comparison": comparison, "review_threshold": cut,
               "holdout_by_scope": comparison_by_scope, "batch_transfer": transfer,
               "v2_pool_batch_transfer": baseline_transfer}
    return final, summary, baseline_records + expanded_records


def combined_results(predictions):
    matching = {}
    for row in predictions:
        if row["method"] != "expanded":
            continue
        key = (row["mode"], row["condition"], row["fold"], row["pair"], row["label"])
        reference = matching.setdefault(key, {**row, "score": None, "threshold": 0.0})
        if row["score"] is not None and row["threshold"] is not None:
            margin = row["score"] - row["threshold"]
            if reference["score"] is None or margin > reference["score"]:
                reference["score"] = margin
    return {mode: by_condition([row for row in matching.values() if row["mode"] == mode])
            for mode in BASELINES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cohorts = read_cohorts(args.annotations)
    report = {"cohorts": dict(Counter(cohorts.values())), "recall_target": RECALL_TARGET, "scales": {}}
    model = {"schema": SCHEMA, "id": "doubao-damage-cpu-v3", "heads": {}, "detail_heads": {}}
    predictions = []
    candidates = CANDIDATES.copy()
    for suffix, families in (
        ("texture-conditioned", ("texture_conditioned",)),
        ("texture-color", ("texture_conditioned", "chroma")),
        ("texture-texture", ("texture_conditioned", "rgb", "chroma", "fine_edges")),
        ("texture-residuals", ("texture_conditioned", "residual_shape", "cooccurrence", "laplacian")),
    ):
        candidates[suffix] = Candidate(suffix, families, method="paired", augmented=True)
        candidates[suffix + "-boost"] = Candidate(suffix + "-boost", families, method="boost", augmented=True)
    for side, destination in ((MIN_SHORT_SIDE, "heads"), (DETAIL_SHORT_SIDE, "detail_heads")):
        pairs, excluded = read_pairs(args.annotations, args.real_dir, args.ai_dir, short_side=side)
        original_groups = {pair.group for pair in pairs if cohorts[pair.name] == "original_high"}
        if len(original_groups) < 5:
            parser.error("扩展标注需要至少 5 个 original_high 商品组来保留 v2 留出分组。")
        scale_report = {"excluded": excluded, "modes": {}}
        report["scales"][side] = scale_report
        for mode in BASELINES:
            head, results, rows = evaluate(pairs, cohorts, mode, candidates)
            predictions.extend({**row, "short_side": side} for row in rows)
            model[destination][mode] = head
            scale_report["modes"][mode] = results
            print(side, mode, results["final"], flush=True)
    report["dual_scale_holdout"] = combined_results(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(model, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    report["annotations_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    report["weights_bytes"] = len(encoded)
    report["weights_sha256"] = hashlib.sha256(encoded).hexdigest()
    (args.output / "model.json").write_bytes(encoded)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report["dual_scale_holdout"], ensure_ascii=False))


if __name__ == "__main__":
    main()

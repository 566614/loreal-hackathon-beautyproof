"""Check fixed predictions against the Doubao damage accuracy and false-positive targets."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


def target_metrics(rows):
    positives = negatives = hits = false_alerts = correct_reals = undecided = 0
    for row in rows:
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Expected a binary label, received {label}")
        positives += label
        negatives += 1 - label
        if row["score"] in (None, "") or row["threshold"] in (None, ""):
            undecided += 1
            continue
        score, threshold = float(row["score"]), float(row["threshold"])
        if not math.isfinite(score) or not math.isfinite(threshold):
            raise ValueError("Each scored prediction needs a finite score and cutoff")
        if score >= threshold:
            hits += label
            false_alerts += 1 - label
        else:
            correct_reals += 1 - label
    total = positives + negatives
    correct = hits + correct_reals
    return {
        "positives": positives, "negatives": negatives, "true_positives": hits, "false_positives": false_alerts,
        "undecided": undecided, "accuracy": correct / total if total else None,
        "false_positive_rate": false_alerts / negatives if negatives else None,
        "recall": hits / positives if positives else None,
        "passes": bool(positives and negatives and undecided == 0 and
                       20 * correct > 19 * total and 20 * false_alerts < negatives),
    }


def evaluate_predictions(rows, method):
    slices = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            slices[(row["protocol"], row["mode"], row["condition"])].append(row)
    if not slices:
        raise ValueError(f"No predictions for {method}")
    results = [{"protocol": protocol, "mode": mode, "condition": condition, **target_metrics(slice_rows)}
               for (protocol, mode, condition), slice_rows in sorted(slices.items())]
    return {"method": method, "minimum_accuracy_exclusive": 0.95,
            "maximum_false_positive_rate_exclusive": 0.05,
            "passes_all_reported_slices": all(result["passes"] for result in results), "slices": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8-sig", newline="") as stream:
        report = evaluate_predictions(csv.DictReader(stream), args.method)
    message = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(message + "\n", encoding="utf-8")
    else:
        print(message)
    return 0 if report["passes_all_reported_slices"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

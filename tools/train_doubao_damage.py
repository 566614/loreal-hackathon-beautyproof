"""Train and evaluate the local damage-review detector on manually reviewed pairs."""
import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from doubao_damage import FAMILY_LABELS, MODEL_PATH, SCHEMA, TILE, decode, extract, prepare, predict_vector, validate_roi

CONDITIONS = {"matched": (85, False), "recompressed": (60, False), "reduced_detail": (85, True)}
MODEL_CANDIDATES = {
    **{key: [key] for key in FAMILY_LABELS},
    "residuals": ["rgb", "gaussian", "median", "laplacian", "directional"],
    "all_pixel_cues": list(FAMILY_LABELS),
}
CONTROLS = {"texture": ["texture"], "edges": ["edges"]}
PENALTIES = (0.1, 1.0)
MODE_NAMES = ("whole", "region")


@dataclass
class Pair:
    name: str
    real_name: str
    group: str
    batch: str
    confidence: str
    views: dict
    original_bytes: tuple


def read_pairs(path, real_dir, ai_dir):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs, seen_ai, original_groups = [], set(), {}
    excluded = Counter()
    for row in rows:
        if row["label"] != "damage" or row["confidence"] not in ("high", "medium"):
            excluded[row["label"]] += 1
            continue
        if not row["group"]:
            raise ValueError(f"缺少商品组：{row['ai_file']}")
        original = (real_dir / row["real_file"]).read_bytes()
        edited = (ai_dir / row["ai_file"]).read_bytes()
        source_hash = hashlib.sha256(original).hexdigest()
        if original_groups.setdefault(source_hash, row["group"]) != row["group"]:
            raise ValueError(f"同一实拍原图被标入多个组：{row['real_file']}")
        edited_hash = hashlib.sha256(edited).hexdigest()
        if edited_hash in seen_ai:
            excluded["duplicate"] += 1
            continue
        seen_ai.add(edited_hash)
        views = {condition: {mode: [] for mode in MODE_NAMES} for condition in CONDITIONS}
        for data, field in ((original, "real_roi"), (edited, "ai_roi")):
            image, _ = decode(data)
            roi = validate_roi(json.loads(row[field]))
            for condition, (quality, smaller) in CONDITIONS.items():
                prepared = prepare(image, quality=quality, smaller=smaller)
                whole, region, _, _ = extract(prepared, roi, study_periodicity=True) if prepared is not None else (None,) * 4
                views[condition]["whole"].append(whole)
                views[condition]["region"].append(region)
        if any(v is None for v in views["matched"]["whole"]):
            excluded["too_small"] += 1
            continue
        pairs.append(Pair(row["ai_file"], row["real_file"], row["group"], row["batch"],
                          row["confidence"], views, (len(original), len(edited))))
    return pairs, dict(excluded)


def grouped_folds(pairs, k):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair.group].append(pair)
    batches = sorted({pair.batch for pair in pairs})
    total = Counter(pair.batch for pair in pairs)
    folds = [[] for _ in range(k)]
    counts = [Counter() for _ in folds]
    ordered = sorted(grouped.values(), key=lambda group: (-len(group), group[0].group))
    for group in ordered:
        batch_count = Counter(pair.batch for pair in group)
        index = min(range(k), key=lambda i: (
            sum(batch_count[b] * counts[i][b] / total[b] for b in batches),
            len(folds[i]), i))
        folds[index].extend(group)
        counts[index].update(batch_count)
    for i, fold in enumerate(folds):
        yield [pair for j, other in enumerate(folds) if i != j for pair in other], fold


def train_head(pairs, mode, families, penalty):
    counts = Counter(pair.group for pair in pairs)
    rows, labels, weights = [], [], []
    for pair in pairs:
        for label, values in enumerate(pair.views["matched"][mode]):
            rows.append(np.concatenate([values[key] for key in families]))
            labels.append(label)
            weights.append(1 / counts[pair.group])
    x = np.asarray(rows)
    y = np.asarray(labels)
    weights = np.asarray(weights)
    weights /= weights.sum()
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2)).clip(1e-4)
    z = np.column_stack((np.ones(len(x)), np.clip((x - mean) / scale, -8, 8)))
    parameters = np.zeros(z.shape[1])
    regularizer = np.eye(len(parameters)) * penalty
    regularizer[0, 0] = 0
    # Newton steps for L2 logistic regression with a balanced weight per product.
    for _ in range(25):
        p = 1 / (1 + np.exp(-np.clip(z @ parameters, -25, 25)))
        gradient = z.T @ ((p - y) * weights) + regularizer @ parameters
        hessian = (z.T * (p * (1 - p) * weights)) @ z + regularizer
        step = np.linalg.solve(hessian, gradient)
        parameters -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"families": families, "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": parameters[1:].tolist(), "intercept": float(parameters[0]), "threshold": None}


def score_rows(head, pairs, mode, conditions=CONDITIONS, *, threshold=None):
    rows = []
    for pair in pairs:
        for condition in conditions:
            for label, values in enumerate(pair.views[condition][mode]):
                rows.append({"file": pair.name if label else pair.real_name, "pair": pair.name,
                             "group": pair.group, "batch": pair.batch, "confidence": pair.confidence,
                             "condition": condition, "label": label,
                             "score": float(predict_vector(head, values)) if values is not None else None,
                             "threshold": threshold})
    return rows


def metrics(rows, *, threshold=None):
    valid = [row for row in rows if row["score"] is not None]
    positives = np.array([row["score"] for row in valid if row["label"]])
    negatives = np.array([row["score"] for row in valid if not row["label"]])
    auc = None
    if positives.size and negatives.size:
        relation = positives[:, None] - negatives[None, :]
        auc = float(np.mean((relation > 0) + 0.5 * (relation == 0)))
    tp, fp, pos_decided, neg_decided = 0, 0, 0, 0
    for row in valid:
        cut = threshold if threshold is not None else row["threshold"]
        if cut is None:
            continue
        alert = int(row["score"] >= cut)
        if row["label"]:
            tp += alert
            pos_decided += 1
        else:
            fp += alert
            neg_decided += 1
    return {"auc": auc, "positives": int(positives.size), "negatives": int(negatives.size),
            "negative_images": len({row["file"] for row in valid if not row["label"]}),
            "groups": len({row["group"] for row in rows}),
            "recall": tp / pos_decided if pos_decided else None,
            "false_positive_rate": fp / neg_decided if neg_decided else None,
            "true_positives": tp, "false_positives": fp,
            "undecided": len(rows) - pos_decided - neg_decided,
            "rows": len(rows)}


def choose_threshold(rows):
    choices = [1.0] + sorted({row["score"] for row in rows if row["score"] is not None})
    candidates = []
    for threshold in choices:
        record = metrics(rows, threshold=threshold)
        if record["false_positive_rate"] is not None and record["false_positive_rate"] <= 0.10:
            candidates.append((record["recall"], -record["false_positive_rate"], threshold))
    return max(candidates)[2] if candidates else None


def by_condition(rows):
    return {condition: metrics([row for row in rows if row["condition"] == condition])
            for condition in CONDITIONS}


def compare_candidates(pairs, mode):
    k = min(4, len({pair.group for pair in pairs}))
    ranking = []
    comparison = {}
    for name, families in {**MODEL_CANDIDATES, **CONTROLS}.items():
        for penalty in PENALTIES if name in MODEL_CANDIDATES else (0.1,):
            scores = []
            for train, held in grouped_folds(pairs, k):
                head = train_head(train, mode, families, penalty)
                scores.extend(score_rows(head, held, mode))
            result = by_condition(scores)
            comparison[f"{name}:{penalty:g}"] = result
            nominal = [row for row in scores if row["condition"] == "matched"]
            aucs = [r["auc"] for r in result.values()]
            if name in MODEL_CANDIDATES and all(auc is not None for auc in aucs):
                ranking.append((min(aucs), result["matched"]["auc"], -len(families), name, penalty, nominal))
    _, _, _, name, penalty, nominal = max(ranking, key=lambda row: row[:5])
    return name, penalty, choose_threshold(nominal), comparison


def evaluate(pairs, mode):
    usable = [p for p in pairs if all(v is not None for v in p.views["matched"][mode])]
    primary = [p for p in usable if p.confidence == "high"]
    medium = [p for p in usable if p.confidence == "medium"]
    group_count = len({p.group for p in primary})
    if group_count < 4:
        return None, {"reason": "高置信独立商品组少于 4 个", "primary_pairs": len(primary)}, []
    scores, secondary_scores, selected = [], [], Counter()
    for fold, (train, held) in enumerate(grouped_folds(primary, min(5, group_count)), 1):
        candidate, penalty, threshold, _ = compare_candidates(train, mode)
        selected[candidate] += 1
        head = train_head(train, mode, MODEL_CANDIDATES[candidate], penalty)
        rows = score_rows(head, held, mode, threshold=threshold)
        scores.extend({**row, "fold": fold, "eval": "primary"} for row in rows)
        held_groups = {pair.group for pair in held}
        rows = score_rows(head, [p for p in medium if p.group in held_groups], mode, threshold=threshold)
        secondary_scores.extend({**row, "fold": fold, "eval": "medium"} for row in rows)

    candidate, penalty, threshold, comparison = compare_candidates(primary, mode)
    head = train_head(primary, mode, MODEL_CANDIDATES[candidate], penalty)
    unseen_group = {p.group for p in primary}
    rows = score_rows(head, [p for p in medium if p.group not in unseen_group], mode, threshold=threshold)
    secondary_scores.extend({**row, "fold": None, "eval": "medium"} for row in rows)
    summary = by_condition(scores)
    main = summary["matched"]
    accepted = (group_count >= 10 and main["recall"] is not None and main["recall"] >= 0.60
                and main["false_positive_rate"] <= 0.10 and threshold is not None)
    if accepted:
        head["threshold"] = threshold
    batches = sorted({pair.batch for pair in primary})
    batch_transfer = {}
    for batch in batches:
        held = [p for p in primary if p.batch == batch]
        held_groups = {p.group for p in held}
        train = [p for p in primary if p.batch != batch and p.group not in held_groups]
        if len({p.group for p in train}) < 4:
            continue
        name, reg, cut, _ = compare_candidates(train, mode)
        model = train_head(train, mode, MODEL_CANDIDATES[name], reg)
        batch_transfer[batch] = by_condition(score_rows(model, held, mode, threshold=cut))
    return head, {
        "primary_pairs": len(primary), "medium_pairs": len(medium), "group_count": group_count,
        "nested_group_cv": summary,
        "by_batch": {batch: by_condition([row for row in scores if row["batch"] == batch]) for batch in batches},
        "heldout_batch": batch_transfer, "medium_group_disjoint": by_condition(secondary_scores),
        "fold_winners": dict(selected), "final_candidate": candidate, "penalty": penalty,
        "candidate_comparison": comparison, "trial_threshold": threshold, "threshold_enabled": accepted,
    }, scores + secondary_scores


def paired_spectrum_study(pairs):
    """Compare autocorrelation at JPEG block harmonics against other spatial lags."""
    names = ("高频能量占比", "最高频段占比", "频谱峰均比", "水平滞后2", "垂直滞后2",
             "水平滞后4", "垂直滞后4", "水平滞后8", "垂直滞后8", "水平滞后16", "垂直滞后16")
    batches = sorted({p.batch for p in pairs})
    records = []
    per_frequency = []
    for batch in batches:
        for condition in CONDITIONS:
            grouped = defaultdict(list)
            maps, grids = defaultdict(list), defaultdict(list)
            for pair in pairs:
                if pair.batch != batch:
                    continue
                real, ai = pair.views[condition]["whole"]
                grouped[pair.group].append((ai["spectrum"] - real["spectrum"])[:len(names)])
                maps[pair.group].append(ai["spectrum_map"] - real["spectrum_map"])
                grids[pair.group].append(ai["jpeg_grid"] - real["jpeg_grid"])
            if not grouped:
                continue
            deltas = np.stack([np.median(np.stack(values), axis=0) for values in grouped.values()])
            group_maps = np.stack([np.median(values, axis=0) for values in maps.values()])
            per_frequency.append((group_maps > 0).mean(axis=0))
            grid_deltas = np.stack([np.median(values, axis=0) for values in grids.values()])
            records.append({"batch": batch, "condition": condition, "groups": len(deltas),
                            "median_jpeg_grid_delta_xy": np.median(grid_deltas, axis=0).tolist(),
                            "median_ai_minus_real": dict(zip(names, np.median(deltas, axis=0).tolist())),
                            "positive_group_fraction": dict(zip(names, (deltas > 0).mean(axis=0).tolist()))})
    peaks = []
    if per_frequency:
        stable = np.min(per_frequency, axis=0)
        # Restrict to one half of the symmetric spectrum and skip JPEG's block harmonics.
        for y, x in itertools.product(range(TILE // 2 + 1), range(TILE)):
            fx = x if x <= TILE//2 else x - TILE
            if (0.18 <= np.hypot(fx, y)/TILE <= 0.5 and fx % 3 != 0 and y % 3 != 0
                    and stable[y, x] >= 0.8):
                peaks.append({"x_bin": fx, "y_bin": y, "tile_side": TILE,
                              "minimum_positive_group_fraction": float(stable[y, x])})
    return {"by_batch_and_condition": records, "repeatable_non_jpeg_peak_candidates": peaks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    pairs, excluded = read_pairs(args.annotations, args.real_dir, args.ai_dir)
    summary = {"included": dict(Counter(p.confidence for p in pairs)), "excluded": excluded, "modes": {}}
    primary = [p for p in pairs if p.confidence == "high"]
    leaking_scores = [{"score": -size, "label": label, "threshold": None, "file": name, "group": p.group}
                      for p in primary for label, (size, name) in
                      enumerate(zip(p.original_bytes, (p.real_name, p.name)))]
    summary["file_size_shortcut_diagnostic"] = metrics(leaking_scores)
    summary["periodicity_study"] = paired_spectrum_study(primary)
    model = {"schema": SCHEMA, "id": "doubao-damage-cpu-v1", "heads": {}}
    all_scores = []
    for mode in MODE_NAMES:
        head, result, scores = evaluate(pairs, mode)
        summary["modes"][mode] = result
        if head is not None:
            model["heads"][mode] = head
        all_scores.extend({**row, "mode": mode} for row in scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if all_scores:
        with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_scores[0]))
            writer.writeheader()
            writer.writerows(all_scores)
    summary["annotation_sha256"] = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if model["heads"]:
        encoded = json.dumps(model, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        summary["weights_bytes"] = len(encoded)
        summary["weights_sha256"] = hashlib.sha256(encoded).hexdigest()
        args.model.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=args.model.parent, suffix=".tmp", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(encoded)
        pending.replace(args.model)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({mode: result.get("nested_group_cv", result)
                      for mode, result in summary["modes"].items()}, ensure_ascii=False))
    return 0 if model["heads"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

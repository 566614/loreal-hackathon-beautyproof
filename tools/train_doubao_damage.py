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

from doubao_damage import (FAMILY_LABELS, MODEL_PATH, SCHEMA, TILE, decode, extract, prepare,
                           predict_matrix, validate_roi)

CONDITIONS = {"matched": (85, False), "recompressed": (60, False), "reduced_detail": (85, True)}
MODE_NAMES = ("whole", "region")
LEGACY_FAMILIES = ("rgb", "gaussian", "median", "laplacian", "directional", "spectrum")
RESIDUALS = ("gaussian", "laplacian", "directional", "residual_shape", "cooccurrence")
COLOR = ("rgb", "chroma")
SHAPE = ("fine_edges", "edges")
RECALL_TARGET = 0.90


@dataclass(frozen=True)
class Candidate:
    name: str
    families: tuple
    penalty: float = 0.5
    method: str = "linear"
    augmented: bool = False
    auxiliary: bool = False


LEGACY = [
    Candidate(f"{name}:{penalty:g}", keys, penalty)
    for name, keys in {**{key: (key,) for key in LEGACY_FAMILIES},
                       "residuals": LEGACY_FAMILIES[:-1], "all_pixel_cues": LEGACY_FAMILIES}.items()
    for penalty in (0.1, 1.0)
]
EXTENDED = [
    Candidate("relative-residuals", RESIDUALS, method="paired", augmented=True),
    Candidate("relative-color", COLOR, method="paired", augmented=True),
    Candidate("relative-spectrum", ("spectrum", "residual_shape"), method="paired", augmented=True),
    Candidate("relative-contour", SHAPE, method="paired", augmented=True),
    Candidate("relative-residuals-color", RESIDUALS + COLOR, method="paired", augmented=True),
    Candidate("relative-residuals-shape", RESIDUALS + SHAPE, method="paired", augmented=True),
    Candidate("relative-all", RESIDUALS + COLOR + SHAPE + ("spectrum",), method="paired", augmented=True),
    Candidate("boost-residuals", RESIDUALS + COLOR, method="boost", augmented=True),
    Candidate("boost-contour", SHAPE, method="boost", augmented=True),
    Candidate("boost-all", RESIDUALS + COLOR + SHAPE + ("spectrum",), method="boost", augmented=True),
    Candidate("auxiliary-residuals", RESIDUALS + COLOR, method="paired", augmented=True, auxiliary=True),
    Candidate("auxiliary-boost", RESIDUALS + COLOR, method="boost", augmented=True, auxiliary=True),
]


@dataclass
class Pair:
    name: str
    real_name: str
    group: str
    batch: str
    confidence: str
    views: dict
    original_bytes: tuple
    annotation: str


def read_pairs(path, real_dir, ai_dir):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs, seen_ai, original_groups = [], set(), {}
    excluded = Counter()
    # Prefer a labeled damage example if an export appears under two names.
    for row in sorted(rows, key=lambda row: (row["label"] != "damage", row["confidence"] != "high")):
        usable = row["label"] == "damage" or (
            row["label"] == "outside" and row["confidence"] == "high"
            and "版本" not in row["reason"] and "多商品" not in row["reason"]
        )
        if not usable or row["confidence"] not in ("high", "medium"):
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
                whole, region, _, _ = (extract(prepared, roi, study_periodicity=row["label"] == "damage",
                                                     extra=FAMILY_LABELS)
                                      if prepared is not None else (None,) * 4)
                views[condition]["whole"].append(whole)
                views[condition]["region"].append(region)
        if any(v is None for v in views["matched"]["whole"]):
            excluded["too_small"] += 1
            continue
        pairs.append(Pair(row["ai_file"], row["real_file"], row["group"], row["batch"],
                          row["confidence"], views, (len(original), len(edited)), row["label"]))
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


def _eligible_auxiliary(primary, auxiliary):
    training_groups = {pair.group for pair in primary}
    return [pair for pair in auxiliary if pair.group in training_groups]


def _training_vectors(primary, auxiliary, mode, candidate):
    pairs = [(pair, 1.0) for pair in primary]
    if candidate.auxiliary:
        pairs.extend((pair, 0.35) for pair in _eligible_auxiliary(primary, auxiliary))
    group_weights = Counter()
    for pair, weight in pairs:
        group_weights[pair.group] += weight
    conditions = CONDITIONS if candidate.augmented else ("matched",)
    rows, weights = [], []
    for pair, weight in pairs:
        for condition in conditions:
            features = pair.views[condition][mode]
            if all(value is not None for value in features):
                rows.append([np.concatenate([value[key] for key in candidate.families]) for value in features])
                weights.append(weight / (len(conditions) * group_weights[pair.group]))
    return np.asarray(rows), np.asarray(weights)


def _boosted_stumps(x, labels, weights, limit=18):
    cuts = np.quantile(x, np.linspace(0.1, 0.9, 9), axis=0)
    indices = np.tile(np.arange(x.shape[1]), len(cuts))
    cuts = cuts.ravel()
    comparisons = np.where(x[:, indices] >= cuts, 1.0, -1.0)
    signal = labels * 2.0 - 1.0
    stumps = []
    for _ in range(limit):
        edges = (weights * signal) @ comparisons
        index = int(np.argmax(np.abs(edges)))
        edge = float(np.clip(edges[index], -0.995, 0.995))
        if abs(edge) < 1e-8:
            break
        strength = np.arctanh(edge) * 0.5
        stumps.append([int(indices[index]), float(cuts[index]), float(strength)])
        weights *= np.exp(-signal * comparisons[:, index] * strength)
        weights /= weights.sum()
    return stumps


def train_head(pairs, mode, candidate, auxiliary=()):
    pair_rows, pair_weights = _training_vectors(pairs, auxiliary, mode, candidate)
    x = pair_rows.reshape(-1, pair_rows.shape[-1])
    y = np.tile([0, 1], len(pair_rows))
    weights = np.repeat(pair_weights, 2)
    weights /= weights.sum()
    head = {"families": candidate.families, "threshold": None}
    if candidate.method == "boost":
        return {**head, "stumps": _boosted_stumps(x, y, weights)}

    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2)).clip(1e-4)
    standardized = np.clip((x - mean) / scale, -8, 8)
    z = np.column_stack((np.ones(len(x)), standardized))
    if candidate.method == "paired":
        delta = standardized[1::2] - standardized[::2]
        pair_design = np.column_stack((np.zeros(len(z)), np.stack((-delta, delta), axis=1).reshape(x.shape)))
        z = np.concatenate((z, pair_design))
        y = np.concatenate((y, y))
        weights = np.concatenate((weights, 0.7 * weights))
        weights /= weights.sum()
    parameters = np.zeros(z.shape[1])
    regularizer = np.eye(len(parameters)) * candidate.penalty
    regularizer[0, 0] = 0
    for _ in range(25):
        p = 1 / (1 + np.exp(-np.clip(z @ parameters, -25, 25)))
        gradient = z.T @ ((p - y) * weights) + regularizer @ parameters
        hessian = (z.T * (p * (1 - p) * weights)) @ z + regularizer
        step = np.linalg.solve(hessian, gradient)
        parameters -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {**head, "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": parameters[1:].tolist(), "intercept": float(parameters[0])}


def score_rows(head, pairs, mode, conditions=CONDITIONS, *, threshold=None):
    rows, vectors = [], []
    for pair in pairs:
        for condition in conditions:
            for label, values in enumerate(pair.views[condition][mode]):
                rows.append({"file": pair.name if label else pair.real_name, "pair": pair.name,
                             "group": pair.group, "batch": pair.batch, "confidence": pair.confidence,
                             "condition": condition, "label": label, "score": None, "threshold": threshold})
                if values is not None:
                    vectors.append((rows[-1], np.concatenate([values[key] for key in head["families"]])))
    if vectors:
        outputs = predict_matrix(head, np.stack([vector for _, vector in vectors]))
        for (row, _), score in zip(vectors, outputs):
            row["score"] = float(score)
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
    """Pick the highest cutoff that reaches the sensitivity target on every development view."""
    cuts = []
    for condition in CONDITIONS:
        positives = sorted((row["score"] for row in rows if row["label"] and row["score"] is not None
                            and row["condition"] == condition), reverse=True)
        if not positives:
            return None
        cuts.append(positives[int(np.ceil(RECALL_TARGET * len(positives))) - 1])
    return min(cuts)


def by_condition(rows, *, threshold=None):
    return {condition: metrics([row for row in rows if row["condition"] == condition], threshold=threshold)
            for condition in CONDITIONS}


def compare_candidates(pairs, mode, auxiliary=(), *, include_legacy_control=False):
    plans = list(grouped_folds(pairs, min(4, len({pair.group for pair in pairs}))))
    ranking, baseline_ranking, comparison = [], [], {}
    for candidate in LEGACY + EXTENDED:
        scores = []
        for train, held in plans:
            head = train_head(train, mode, candidate, auxiliary)
            scores.extend(score_rows(head, held, mode))
        threshold = choose_threshold(scores)
        result = by_condition(scores, threshold=threshold)
        comparison[candidate.name] = result
        aucs = [r["auc"] for r in result.values()]
        rates = [r["false_positive_rate"] for r in result.values()]
        if threshold is None or any(value is None for value in aucs + rates):
            continue
        rank = (min(aucs), result["matched"]["auc"], -len(candidate.families))
        # Prefer strong ranking across transformations; use the high-recall operating point as a secondary signal.
        ranking.append(((min(aucs) + 0.20 * (1 - np.mean(rates)), np.mean(aucs), -len(candidate.families)),
                        candidate, threshold))
        if candidate in LEGACY:
            baseline_ranking.append((rank, candidate, threshold))
    _, candidate, threshold = max(ranking, key=lambda row: row[0])
    baseline = max(baseline_ranking, key=lambda row: row[0])[1:] if include_legacy_control else None
    return candidate, threshold, comparison, baseline


def evaluate(pairs, mode):
    usable = [p for p in pairs if all(v is not None for v in p.views["matched"][mode])]
    primary = [p for p in usable if p.confidence == "high" and p.annotation == "damage"]
    medium = [p for p in usable if p.confidence == "medium" and p.annotation == "damage"]
    auxiliary = [p for p in usable if p.confidence != "high" or p.annotation != "damage"]
    group_count = len({p.group for p in primary})
    if group_count < 4:
        return None, {"reason": "高置信独立商品组少于 4 个", "primary_pairs": len(primary)}, []
    scores, legacy_scores, secondary_scores, selected = [], [], [], Counter()
    for fold, (train, held) in enumerate(grouped_folds(primary, min(5, group_count)), 1):
        candidate, threshold, _, (legacy, legacy_threshold) = compare_candidates(
            train, mode, auxiliary, include_legacy_control=True
        )
        selected[candidate.name] += 1
        head = train_head(train, mode, candidate, auxiliary)
        rows = score_rows(head, held, mode, threshold=threshold)
        scores.extend({**row, "fold": fold, "eval": "primary"} for row in rows)
        legacy_head = train_head(train, mode, legacy)
        rows = score_rows(legacy_head, held, mode, threshold=legacy_threshold)
        legacy_scores.extend({**row, "fold": fold, "eval": "legacy_high_recall"} for row in rows)
        held_groups = {pair.group for pair in held}
        rows = score_rows(head, [p for p in medium if p.group in held_groups], mode, threshold=threshold)
        secondary_scores.extend({**row, "fold": fold, "eval": "medium"} for row in rows)

    candidate, threshold, comparison, _ = compare_candidates(primary, mode, auxiliary)
    head = train_head(primary, mode, candidate, auxiliary)
    seen_groups = {p.group for p in primary}
    rows = score_rows(head, [p for p in medium if p.group not in seen_groups], mode, threshold=threshold)
    secondary_scores.extend({**row, "fold": None, "eval": "medium"} for row in rows)
    summary = by_condition(scores)
    head["threshold"] = threshold
    batches = sorted({pair.batch for pair in primary})
    batch_transfer = {}
    for batch in batches:
        held = [p for p in primary if p.batch == batch]
        held_groups = {p.group for p in held}
        train = [p for p in primary if p.batch != batch and p.group not in held_groups]
        if len({p.group for p in train}) < 4:
            continue
        transfer_auxiliary = [pair for pair in auxiliary if pair.batch != batch]
        variant, cut, _, _ = compare_candidates(train, mode, transfer_auxiliary)
        model = train_head(train, mode, variant, transfer_auxiliary)
        batch_transfer[batch] = by_condition(score_rows(model, held, mode, threshold=cut))
    return head, {
        "primary_pairs": len(primary), "medium_pairs": len(medium), "group_count": group_count,
        "auxiliary_pairs_in_final_pool": len(_eligible_auxiliary(primary, auxiliary)) if candidate.auxiliary else 0,
        "nested_group_cv": summary, "legacy_high_recall": by_condition(legacy_scores),
        "by_batch": {batch: by_condition([row for row in scores if row["batch"] == batch]) for batch in batches},
        "heldout_batch": batch_transfer, "medium_group_disjoint": by_condition(secondary_scores),
        "fold_winners": dict(selected), "final_candidate": candidate.name,
        "final_families": candidate.families, "target_dev_recall": RECALL_TARGET,
        "candidate_comparison": comparison, "review_threshold": threshold,
    }, scores + legacy_scores + secondary_scores


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
    summary = {"included": dict(Counter(p.confidence for p in pairs if p.annotation == "damage")),
               "origin_auxiliary": sum(p.annotation != "damage" for p in pairs), "excluded": excluded, "modes": {}}
    primary = [p for p in pairs if p.confidence == "high" and p.annotation == "damage"]
    leaking_scores = [{"score": -size, "label": label, "threshold": None, "file": name, "group": p.group}
                      for p in primary for label, (size, name) in
                      enumerate(zip(p.original_bytes, (p.real_name, p.name)))]
    summary["file_size_shortcut_diagnostic"] = metrics(leaking_scores)
    summary["periodicity_study"] = paired_spectrum_study(primary)
    model = {"schema": SCHEMA, "id": "doubao-damage-cpu-v2", "heads": {}}
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

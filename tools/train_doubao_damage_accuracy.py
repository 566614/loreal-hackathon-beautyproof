"""Choose a compact detector by balanced accuracy on locked product folds."""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from doubao_damage import (BALANCED_SCHEMA, DETAIL_SHORT_SIDE, FAMILY_LABELS, MIN_SHORT_SIDE, decode, extract, prepare)
from train_doubao_damage import COLOR, CONDITIONS, RESIDUALS, SHAPE, Candidate, Pair, by_condition, grouped_folds, score_rows, train_head
from train_doubao_damage_expanded import CANDIDATES as V3_BASE, read_cohorts, select as v3_select, split_locked

SCALES = (MIN_SHORT_SIDE, DETAIL_SHORT_SIDE)
MODES = ("whole", "region")
TEXTURE = ("texture_conditioned",)
NEIGHBORS = ("noise_consistency",)
CODEC = ("codec_response",)
LINE_CUES = ("fracture_stacked",)
PIXELS = TEXTURE + COLOR + RESIDUALS
V3_CANDIDATES = {
    **V3_BASE,
    **{label + suffix: Candidate(label + suffix, features, method=algo, augmented=True)
       for label, features in [
           ("texture-conditioned", TEXTURE),
           ("texture-color", TEXTURE + ("chroma",)),
           ("texture-texture", TEXTURE+COLOR+("fine_edges",)),
           ("texture-residuals", TEXTURE+("residual_shape", "cooccurrence", "laplacian")),
       ] for algo, suffix in (("paired", ""), ("boost", "-boost"))}
}


@dataclass(frozen=True)
class Choice:
    candidate: Candidate
    scales: tuple

    @property
    def label(self):
        return f"{self.candidate.name}@{'+'.join(str(side) for side in self.scales)}"


CHOICES = [
    Choice(Candidate(label, features, penalty=penalty, method=method, augmented=True), scales)
    for label, features, penalty, method, scales in [
        ("color", TEXTURE+COLOR, 0.5, "paired", (288,)),
        ("color-small", TEXTURE+COLOR, 0.5, "paired", (256,)),
        ("color-two-views", TEXTURE+COLOR, 0.5, "paired", (256, 288)),
        ("color-residuals", PIXELS, 0.5, "paired", (288,)),
        ("color-residuals-light", PIXELS, 1.0, "paired", (256,)),
        ("native-consistency", TEXTURE+COLOR+NEIGHBORS, 0.5, "paired", (288,)),
        ("consistent-residuals", PIXELS+NEIGHBORS, 0.5, "paired", (288,)),
        ("reencode-color", TEXTURE+COLOR+CODEC, 0.5, "paired", (288,)),
        ("reencode-residuals", PIXELS+NEIGHBORS+CODEC, 0.5, "paired", (288,)),
        ("reencode-boost", PIXELS+NEIGHBORS+CODEC, 0.5, "boost", (288,)),
        ("reencode-two-views", PIXELS+NEIGHBORS+CODEC, 0.5, "paired", (256, 288)),
        ("reencode-and-lines", PIXELS+NEIGHBORS+CODEC+LINE_CUES, 0.5, "paired", (288,)),
    ]
]


RELEASE_RECIPES = {mode: next(choice for choice in CHOICES if choice.candidate.name == name)
                   for mode, name in (("whole", "reencode-color"), ("region", "reencode-and-lines"))}
# L2 penalty, emphasis for high-scoring real photos, and margin above the balanced cutoff.
LOW_FP_SETTINGS = {"whole": (0.5, 2.5, 0.05), "region": (0.15, 1.5, 0.065)}


def _upload_views(photo):
    """Produce upload bytes first; decode them before any detector normalization."""
    yield "matched", photo
    width, height = photo.size
    variants = (("recompressed", min(1, 1024/max(width, height)), 60),
                ("reduced_detail", min(1, 256/min(width, height)), 45))
    for name, scale, quality in variants:
        shape = tuple(round(side * scale) for side in photo.size)
        resized = photo.resize(shape, Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        resized.save(encoded, format="JPEG", quality=quality, subsampling=2)
        encoded.seek(0)
        with Image.open(encoded) as loaded:
            yield name, loaded.convert("RGB")


def _features(array, roi, extras):
    if array is None:
        return None, None
    whole, region, _, _ = extract(array, roi, extra=extras)
    return whole, region


def read_features(annotation, real_dir, ai_dir):
    with annotation.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["label"] == "damage"]
    upload, legacy = {side: {} for side in SCALES}, {side: {} for side in SCALES}
    old_extras = tuple(name for name in FAMILY_LABELS if name not in NEIGHBORS+CODEC+LINE_CUES)
    new_extras = old_extras + NEIGHBORS + CODEC + LINE_CUES
    for count, row in enumerate(rows, 1):
        views = {(track, side): {condition: {mode: [] for mode in MODES} for condition in CONDITIONS}
                 for track in ("upload", "legacy") for side in SCALES}
        lengths = []
        for directory, file, box in ((real_dir, "real_file", "real_roi"), (ai_dir, "ai_file", "ai_roi")):
            blob = (directory / row[file]).read_bytes()
            lengths.append(len(blob))
            photo, _ = decode(blob)
            roi = json.loads(row[box])
            for condition, image in _upload_views(photo):
                for side in SCALES:
                    extras = new_extras if side == DETAIL_SHORT_SIDE else old_extras
                    features = _features(prepare(image, short_side=side), roi, extras)
                    for mode, props in zip(MODES, features):
                        views["upload", side][condition][mode].append(props)
                        if condition == "matched":
                            views["legacy", side][condition][mode].append(props)
            for condition, (quality, smaller) in CONDITIONS.items():
                if condition == "matched":
                    continue
                for side in SCALES:
                    features = _features(prepare(photo, quality=quality, smaller=smaller, short_side=side),
                                         roi, old_extras)
                    for mode, props in zip(MODES, features):
                        views["legacy", side][condition][mode].append(props)
        for track, registry in (("upload", upload), ("legacy", legacy)):
            for side in SCALES:
                sample = Pair(row["ai_file"], row["real_file"], row["group"], row["batch"], row["confidence"],
                              views[track, side], tuple(lengths), "damage")
                registry[side][sample.name] = sample
        if count % 10 == 0:
            print("decoded pairs", count, flush=True)
    return upload, legacy


def _pairs(store, side, samples):
    return [store[side][sample.name] for sample in samples]


def fit(choice, mode, store, samples, *, hard_negative_weight=0.0):
    return {side: train_head(_pairs(store, side, samples), mode, choice.candidate,
                             hard_negative_weight=hard_negative_weight)
            for side in choice.scales}


def _merge(rows, *, v3=False):
    merged = defaultdict(list)
    for row in rows:
        merged[(row["pair"], row["condition"], row["label"])].append(row)
    result = []
    for group in merged.values():
        values = [row for row in group if row["score"] is not None]
        score = None
        if values:
            if v3:
                score = max(row["score"] - row["threshold"] for row in values)
            else:
                score = float(np.mean([row["score"] for row in values]))
        result.append({**group[0], "score": score, "threshold": 0.0 if v3 else None})
    return result


def score(heads, mode, store, samples, *, v3=False):
    rows = []
    for side, model in heads.items():
        rows.extend(score_rows(model, _pairs(store, side, samples), mode, threshold=model.get("threshold")))
    return _merge(rows, v3=v3)


def assessment(rows, *, threshold=None):
    result = by_condition(rows, threshold=threshold)
    for condition, report in result.items():
        if report["recall"] is not None:
            report["balanced_accuracy"] = (report["recall"]+1-report["false_positive_rate"])/2
        else:
            report["balanced_accuracy"] = None
    return result


def pick_cut(rows):
    """Calibrate one cutoff on grouped development scores across all upload forms."""
    sets = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition and row["score"] is not None]
        pos = np.array([row["score"] for row in selected if row["label"] == 1])
        neg = np.array([row["score"] for row in selected if row["label"] == 0])
        if not len(pos) or not len(neg):
            raise ValueError(f"No calibration examples for {condition}")
        sets.append((pos, neg))
    cuts = np.linspace(0.24, 0.76, 53)
    classes = np.array([((pos[:, None] >= cuts).mean(axis=0) + (neg[:, None] < cuts).mean(axis=0))/2
                        for pos, neg in sets])
    ranking = 0.6 * classes.mean(axis=0) + 0.4 * classes.min(axis=0)
    best = np.argmax(ranking - 1e-5*np.abs(cuts-0.5))
    return float(cuts[best]), float(ranking[best])


def _calibrate_choice(choice, mode, store, samples, *, hard_negative_weight=0.0):
    predicted = []
    for train, held in grouped_folds(samples, 4):
        heads = fit(choice, mode, store, train, hard_negative_weight=hard_negative_weight)
        predicted.extend(score(heads, mode, store, held))
    threshold, objective = pick_cut(predicted)
    return threshold, objective, assessment(predicted, threshold=threshold)


def _fit_low_fp(choice, mode, store, samples):
    penalty, hard_weight, margin = LOW_FP_SETTINGS[mode]
    recipe = replace(choice, candidate=replace(choice.candidate, penalty=penalty))
    cutoff, _, _ = _calibrate_choice(recipe, mode, store, samples, hard_negative_weight=hard_weight)
    heads = fit(recipe, mode, store, samples, hard_negative_weight=hard_weight)
    return heads, cutoff + margin


def _baseline_heads(mode, train, legacy, *, v3_report=None, v3_cuts=None, fold=None):
    result = {}
    for side in SCALES:
        examples = _pairs(legacy, side, train)
        if fold is None:
            candidate, threshold, _ = v3_select(examples, mode, V3_CANDIDATES)
        else:
            variant = v3_report["scales"][str(side)]["modes"][mode]["outer_fold_choices"][fold-1]
            candidate = V3_CANDIDATES[variant]
            threshold = v3_cuts[(side, mode, fold)]
        result[side] = train_head(examples, mode, candidate)
        result[side]["threshold"] = threshold
    return result


def _filter(samples, upload, mode):
    return [pair for pair in samples if all(
        all(features is not None for features in upload[side][pair.name].views["matched"][mode]) for side in SCALES
    )]


def run(upload, legacy, cohorts, v3_report, v3_cuts, *, low_fp=False):
    evaluations, predictions = {}, []
    model = {"schema": BALANCED_SCHEMA, "id": f"doubao-damage-cpu-v{5 if low_fp else 4}", "heads": {}}
    for mode in MODES:
        samples = _filter(list(upload[SCALES[0]].values()), upload, mode)
        anchor = [sample for sample in samples if cohorts[sample.name] == "original_high"]
        test_records, baseline_records, calibrated_records = [], [], []
        low_fp_records = []
        recipe = RELEASE_RECIPES[mode]
        final_v3_families = {side: V3_CANDIDATES[v3_report["scales"][str(side)]["modes"][mode]["final"]]
                             for side in SCALES}

        def refit_fixed(train):
            return {side: train_head(_pairs(legacy, side, train), mode, candidate)
                    for side, candidate in final_v3_families.items()}
        for fold, (train, held) in enumerate(split_locked(samples, anchor), 1):
            cutoff, _, _ = _calibrate_choice(recipe, mode, upload, train)
            preferred = score(fit(recipe, mode, upload, train), mode, upload, held)
            test_records.extend({**row, "threshold": cutoff, "protocol": "group", "method": "v4",
                                 "mode": mode, "fold": fold} for row in preferred)
            if low_fp:
                heads, strict_cutoff = _fit_low_fp(recipe, mode, upload, train)
                low_fp_records.extend({**row, "threshold": strict_cutoff, "protocol": "group", "method": "v5",
                                       "mode": mode, "fold": fold} for row in score(heads, mode, upload, held))
            controls = _baseline_heads(mode, train, legacy, v3_report=v3_report, v3_cuts=v3_cuts, fold=fold)
            old = score(controls, mode, upload, held, v3=True)
            baseline_records.extend({**row, "protocol": "group", "method": "v3", "mode": mode, "fold": fold} for row in old)
            # Keep the v3 training recipe and head families, average their scores and recalibrate.
            development = []
            for inner_train, inner_held in grouped_folds(train, 4):
                development.extend(score(refit_fixed(inner_train), mode, upload, inner_held))
            alternative_cut, _ = pick_cut(development)
            alternate = score(refit_fixed(train), mode, upload, held)
            calibrated_records.extend({**row, "threshold": alternative_cut, "protocol": "group",
                                       "method": "v3_recalibrated", "mode": mode, "fold": fold} for row in alternate)
        predictions.extend(test_records+baseline_records+calibrated_records+low_fp_records)
        if low_fp:
            final, final_cutoff = _fit_low_fp(recipe, mode, upload, samples)
        else:
            final_cutoff, _, _ = _calibrate_choice(recipe, mode, upload, samples)
            final = fit(recipe, mode, upload, samples)
        primary = recipe.scales[0]
        model["heads"][mode] = {**final[primary], "threshold": final_cutoff}

        transfer = {}
        for batch in sorted({sample.batch for sample in samples}):
            held = [sample for sample in samples if sample.batch == batch]
            known_groups = {sample.group for sample in held}
            train = [sample for sample in samples if sample.batch != batch and sample.group not in known_groups]
            cutoff, _, _ = _calibrate_choice(recipe, mode, upload, train)
            new_rows = score(fit(recipe, mode, upload, train), mode, upload, held)
            old_rows = score(_baseline_heads(mode, train, legacy), mode, upload, held, v3=True)
            transfer[batch] = {"v4": assessment(new_rows, threshold=cutoff), "v3": assessment(old_rows)}
            predictions.extend({**row, "threshold": cutoff, "protocol": batch, "method": "v4",
                                "mode": mode, "fold": None} for row in new_rows)
            predictions.extend({**row, "protocol": batch, "method": "v3",
                                "mode": mode, "fold": None} for row in old_rows)
            if low_fp:
                heads, strict_cutoff = _fit_low_fp(recipe, mode, upload, train)
                rows = score(heads, mode, upload, held)
                transfer[batch]["v5"] = assessment(rows, threshold=strict_cutoff)
                predictions.extend({**row, "threshold": strict_cutoff, "protocol": batch, "method": "v5",
                                    "mode": mode, "fold": None} for row in rows)
        evaluations[mode] = {
            "pairs": len(samples), "final": recipe.label, "cutoff": final_cutoff,
            "group_holdout": {"v3": assessment(baseline_records), "v3_recalibrated": assessment(calibrated_records),
                              "v4": assessment(test_records)},
            "batch_holdout": transfer,
        }
        if low_fp:
            penalty, emphasis, margin = LOW_FP_SETTINGS[mode]
            evaluations[mode]["low_fp_settings"] = {"penalty": penalty, "hard_negative_weight": emphasis,
                                                     "cutoff_margin": margin}
            evaluations[mode]["group_holdout"]["v5"] = assessment(low_fp_records)
        print(mode, recipe.label, evaluations[mode]["group_holdout"], flush=True)
    return model, evaluations, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("annotations", "real-dir", "ai-dir", "v3-report", "v3-predictions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--low-false-positives", action="store_true", help="训练较少误报的 v5 权重")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    v3_report = json.loads(args.v3_report.read_text(encoding="utf-8"))
    annotation_hash = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if v3_report.get("annotations_sha256") != annotation_hash:
        parser.error("对照实验需使用 v3 的同一份标注。")
    with args.v3_predictions.open(encoding="utf-8-sig", newline="") as stream:
        cuts = {(int(row["short_side"]), row["mode"], int(row["fold"])): float(row["threshold"])
                for row in csv.DictReader(stream) if row["method"] == "expanded" and row["threshold"]}
    upload, original = read_features(args.annotations, args.real_dir, args.ai_dir)
    model, results, records = run(upload, original, read_cohorts(args.annotations), v3_report, cuts,
                                  low_fp=args.low_false_positives)
    args.output.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(model, separators=(",", ":"), allow_nan=False).encode("utf-8")
    evaluation = {"annotations_sha256": annotation_hash, "weights_bytes": len(blob),
                  "weights_sha256": hashlib.sha256(blob).hexdigest(),
                  "references": "v3 is refit on its previous grouped recipe and tested on actual uploads",
                  "conditions": {"matched": "unchanged upload", "recompressed": "long edge <=1024, JPEG Q60 upstream",
                                 "reduced_detail": "short edge <=256, JPEG Q45 upstream"},
                  "modes": results}
    (args.output / "model.json").write_bytes(blob)
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=False),
                                                  encoding="utf-8")
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

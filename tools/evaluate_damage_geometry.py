"""Measure whether fracture geometry adds predictive value on locked product folds."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from doubao_damage import DETAIL_SHORT_SIDE, FAMILY_LABELS, decode
from fracture_geometry import fit_surface_sections, prepare_surface, ridge_maps, stack_operator_maps, trace_fracture
from train_doubao_damage import (COLOR, CONDITIONS, RESIDUALS, SHAPE, Candidate,
                                by_condition, read_pairs, score_rows, train_head)
from train_doubao_damage_expanded import calibrate, combined_results, read_cohorts, split_locked

PRESENCE = ("fracture_presence",)
GEOMETRY = ("fracture_geometry",)
PROFILE = ("fracture_profile",)
STACKED = ("fracture_stacked",)
SURFACE = ("fracture_surface",)
BASELINE = RESIDUALS + COLOR + SHAPE + ("spectrum",)
VARIANTS = {
    "line_presence": PRESENCE,
    "geometry": GEOMETRY,
    "cross_section": PROFILE,
    "physical_cues": PRESENCE + GEOMETRY + PROFILE,
    "stacked_operators": STACKED,
    "surface_fit": SURFACE,
    "stacked_and_surface_fit": STACKED + SURFACE,
    "v3_288_fixed": BASELINE,
    "v3_plus_presence": BASELINE + PRESENCE,
    "v3_plus_geometry": BASELINE + GEOMETRY,
    "v3_plus_cross_section": BASELINE + PROFILE,
    "v3_plus_physical_cues": BASELINE + PRESENCE + GEOMETRY + PROFILE,
    "v3_plus_stacked": BASELINE + STACKED,
    "v3_plus_surface_fit": BASELINE + SURFACE,
    "v3_plus_stacked_and_surface": BASELINE + STACKED + SURFACE,
}


def line_damage_names(annotation):
    """Use the annotated visible damage for the subset, independent of the model."""
    with annotation.open(encoding="utf-8-sig", newline="") as stream:
        samples = csv.DictReader(stream)
        ambiguous = ("断裂", "顶部边缘破碎", "尖部崩裂", "壳下角破裂")
        terms = ("裂纹", "裂痕", "裂线", "开裂", "斜裂", "裂口", "碎裂", "擦裂", "纹路")
        return {row["ai_file"] for row in samples
                if row["label"] == "damage" and any(term in row["reason"] for term in terms)
                and not any(term in row["reason"] for term in ambiguous)}


def ranking_within_pair(rows):
    pairs = defaultdict(dict)
    for row in rows:
        if row["score"] is not None:
            pairs[row["pair"]][row["label"]] = row["score"]
    scores = [(case[1] > case[0]) + 0.5*(case[1] == case[0]) for case in pairs.values() if len(case) == 2]
    return {"pairs": len(scores), "ai_above_real_fraction": sum(scores)/len(scores) if scores else None}


def add_crop_first_features(pairs, annotations, real_dir, ai_dir):
    """Keep the locked folds and add only a second set of framed descriptors."""
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["ai_file"]: row for row in csv.DictReader(stream)}
    for pair in pairs:
        row = labels[pair.name]
        if pair.annotation != "damage":
            continue
        for label, (root, name, key) in enumerate(((real_dir, pair.real_name, "real_roi"),
                                                   (ai_dir, pair.name, "ai_roi"))):
            image, _ = decode((root / name).read_bytes())
            roi = json.loads(row[key])
            for condition, (quality, smaller) in CONDITIONS.items():
                features = pair.views[condition]["region"][label]
                if features is None:
                    continue
                crop = prepare_surface(image, roi, quality=quality, smaller=smaller)
                if crop is None:
                    features["native_stack"] = np.zeros(18)
                    features["native_surface"] = np.zeros(6)
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                maps, _ = stack_operator_maps(gray, ridge_maps(gray))
                trace = trace_fracture(maps)
                features["native_stack"] = np.concatenate((trace.presence, trace.geometry, trace.profile))
                features["native_surface"] = fit_surface_sections(gray, maps["angle"], trace.skeleton)


def evaluate(pairs, cohorts, line_subset, *, crop_first=False):
    rows = []
    report = {}
    for mode in ("whole", "region"):
        data = [pair for pair in pairs if pair.annotation == "damage"
                and all(part is not None for part in pair.views["matched"][mode])]
        anchors = [pair for pair in data if cohorts[pair.name] == "original_high"]
        folds = split_locked(data, anchors)
        mode_report = {}
        variants = VARIANTS
        if crop_first and mode == "region":
            variants = {**VARIANTS, "native_stack": ("native_stack",),
                        "native_surface": ("native_surface",),
                        "v3_plus_native_stack": BASELINE+("native_stack",),
                        "v3_plus_native_and_surface": BASELINE+("native_stack", "native_surface")}
        for name, families in variants.items():
            candidate = Candidate(name, families, method="paired", augmented=True)
            records = []
            for index, (train, held) in enumerate(folds, 1):
                cutoff, _ = calibrate(train, mode, candidate)
                head = train_head(train, mode, candidate)
                records.extend({**row, "mode": mode, "fold": index, "variant": name, "method": "expanded"}
                               for row in score_rows(head, held, mode, threshold=cutoff))
            rows.extend(records)
            relevant = [row for row in records if row["pair"] in line_subset]
            modes = {"all": by_condition(records), "line_damage": by_condition(relevant),
                     "within_pair_line_damage_q60": ranking_within_pair(
                         [row for row in relevant if row["condition"] == "recompressed"]
                     )}
            mode_report[name] = modes
        report[mode] = mode_report
        print(mode, len(data), "line-damage pairs:", sum(pair.name in line_subset for pair in data), flush=True)
    return report, rows


def compare_to_deployed(existing, records):
    with existing.open(encoding="utf-8-sig", newline="") as stream:
        baseline = [{**row, "fold": int(row["fold"]), "label": int(row["label"]),
                     "score": float(row["score"]) if row["score"] else None,
                     "threshold": float(row["threshold"]) if row["threshold"] else None}
                    for row in csv.DictReader(stream) if row["method"] == "expanded"]
    variants = {"deployed_v3": combined_results(baseline)}
    tested = {row["variant"] for row in records}
    for name in sorted(tested):
        if name.startswith("v3_plus") or name == "v3_288_fixed":
            replacement = [row for row in records if row["variant"] == name]
            modes = {row["mode"] for row in replacement}
            retained = [row for row in baseline if row["short_side"] == "256" or row["mode"] not in modes]
            variants[name] = combined_results(retained + replacement)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--v3-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-first", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cohorts = read_cohorts(args.annotations)
    pairs, _ = read_pairs(args.annotations, args.real_dir, args.ai_dir, short_side=DETAIL_SHORT_SIDE,
                          extra=tuple(FAMILY_LABELS) + PRESENCE + GEOMETRY + PROFILE + STACKED + SURFACE)
    print("Prepared", sum(pair.annotation == "damage" for pair in pairs), "matched damage pairs", flush=True)
    line_subset = line_damage_names(args.annotations)
    if args.crop_first:
        add_crop_first_features(pairs, args.annotations, args.real_dir, args.ai_dir)
    report, records = evaluate(pairs, cohorts, line_subset, crop_first=args.crop_first)
    report = {"short_side": DETAIL_SHORT_SIDE, "line_damage_examples": sorted(line_subset),
              "standalone_288": report, "dual_scale": compare_to_deployed(args.v3_predictions, records)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

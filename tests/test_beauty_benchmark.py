"""Regression cases for the blind-set boundary and release metrics."""
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from beauty_benchmark import calibrate, origin_report, sha256, validate


def test_derivative_and_generator_cannot_leak_into_blind(tmp_path):
    image = tmp_path / "original.png"
    Image.new("RGB", (24, 24), "red").save(image)
    manifest = tmp_path / "manifest.json"
    row = {"id": "a", "group": "a", "authorization": "consent-1", "image": image.name,
           "sha256": sha256(image), "kind": "generated", "split": "train",
           "generator": "family-a", "generator_version": "v1"}
    blind = dict(row, id="b", group="b", split="blind", reviewers=["x", "y"])
    manifest.write_text(json.dumps([row, blind]), encoding="utf-8")
    with pytest.raises(ValueError, match="generator"):
        validate(manifest)
    blind.update(kind="edited", edit="retouch", original_sha256=row["sha256"])
    blind.pop("generator")
    manifest.write_text(json.dumps([row, blind]), encoding="utf-8")
    with pytest.raises(ValueError, match="original_sha256"):
        validate(manifest)


def test_group_metrics_count_variants_once():
    rows = [({"group": "a", "kind": "real"}, score) for score in [.1, .95]]
    rows += [({"group": "b", "kind": "generated"}, score) for score in [.9, .7]]
    metrics = origin_report(rows, .4, .8)
    assert (metrics["false_positive"]["count"], metrics["false_positive"]["n"]) == (1, 1)
    assert (metrics["recall"]["count"], metrics["recall"]["n"]) == (0, 1)
    assert metrics["uncertain"]["count"] == 1


def test_calibration_never_approves_a_profile():
    rows = []
    for kind, values in [("real", [.1, .2]), ("generated", [.8, .9])]:
        for idx, score in enumerate(values):
            signal = {"model": "candidate", "weights_sha256": "sha", "preprocessing": "crop", "score": score}
            rows.append(({"kind": kind, "group": f"{kind}-{idx}"}, {"signals": {"origin": signal}}))
    profile = calibrate(rows, "manifest")
    assert profile["high"] == .8
    assert profile["approved"] is False
    assert profile["metrics"]["recall"]["rate"] == 1

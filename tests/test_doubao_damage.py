import io
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import doubao_damage


def sample_image():
    image = Image.new("RGB", (480, 320), (214, 189, 154))
    draw = ImageDraw.Draw(image)
    draw.line([(55, 60), (300, 240), (430, 210)], fill="#352118", width=5)
    return image


def encode(image, kind="JPEG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=kind, **kwargs)
    return stream.getvalue()


def test_coordinates_use_exif_orientation_and_export_region_preview(tmp_path):
    image = sample_image()
    exif = image.getexif()
    exif[274] = 6
    data = encode(image, exif=exif)
    model = tmp_path / "model.json"

    def head(dim):
        return {"families": ["rgb"], "mean": [0] * dim, "scale": [1] * dim,
                "weights": [0] * dim, "intercept": 0, "threshold": None}

    model.write_text(json.dumps({"schema": doubao_damage.SCHEMA, "id": "fixture",
                                 "heads": {"whole": head(8), "region": head(12)}}), encoding="utf-8")
    result = doubao_damage.analyze_image(data, [0.15, 0.2, 0.7, 0.8], model_path=model)
    assert result["image_size"] == [320, 480]
    assert result["source_marker"] == "not_found"
    assert result["whole"]["status"] == result["region"]["status"] == "insufficient"
    assert result["whole"]["score"] == result["region"]["score"] == 0.5
    assert result["edge_preview_png"]
    assert result["device"] == "cpu"


def test_metadata_survives_image_too_small_for_scoring(tmp_path):
    entries = PngInfo()
    entries.add_itxt("XML:com.adobe.xmp", '<TC260:AIGC>{"Label":"1"}</TC260:AIGC>')
    data = encode(Image.new("RGB", (64, 64)), "PNG", pnginfo=entries)
    result = doubao_damage.analyze_image(data, model_path=tmp_path / "missing.json")
    assert result["source_marker"] == "declared_ai"
    assert result["whole"]["score"] is None
    assert "像素" in result["whole"]["reason"]


def test_small_region_keeps_the_default_whole_image_decision():
    data = encode(sample_image())
    whole = doubao_damage.analyze_image(data)
    selected = doubao_damage.analyze_image(data, [.1, .1, .11, .11])
    assert selected["whole"] == whole["whole"]
    assert selected["region"]["status"] == "insufficient"
    assert selected["region"]["score"] is None
    assert "采样点" in selected["region"]["reason"]


def test_default_missing_model_reports_the_same_cause_for_whole_and_region(tmp_path):
    result = doubao_damage.analyze_image(encode(sample_image()), [.2, .2, .8, .8],
                                        model_path=tmp_path/"missing.npz")
    assert result["model"] is None
    assert result["whole"]["status"] == result["region"]["status"] == "insufficient"
    assert result["whole"]["reason"] == result["region"]["reason"]


@pytest.mark.parametrize("roi", [[0.5, 0.2, 0.4, 0.7], [0.1, 0.5, 1.1, 1.0], [False, 0.1, 1, 1]])
def test_invalid_region_fails_before_image_decode(roi):
    with pytest.raises(ValueError, match="坐标"):
        doubao_damage.analyze_image(b"", roi)


def test_v2_contour_head_activates_review_at_its_own_threshold(tmp_path):
    model = tmp_path / "v2.json"
    model.write_text(json.dumps({"schema": doubao_damage.SCHEMA, "id": "doubao-damage-cpu-v2",
                                 "heads": {"whole": {"families": ["fine_edges"],
                                                     "threshold": 0.3, "stumps": [[0, 0.0, 1.0]]}}}),
                     encoding="utf-8")
    result = doubao_damage.analyze_image(encode(sample_image()), model_path=model)
    assert result["whole"]["status"] == "similar"
    assert result["whole"]["score"] > 0.5
    assert result["whole"]["threshold"] == 0.3
    assert result["whole"]["cues"] == [{"label": doubao_damage.FAMILY_LABELS["fine_edges"],
                                      "direction": "raises"}]


def test_recall_threshold_uses_weakest_development_condition():
    from train_doubao_damage import CONDITIONS, choose_threshold
    positives = [(condition, score / 100 - 0.1 * offset)
                 for offset, condition in enumerate(CONDITIONS)
                 for score in range(50, 60)]
    rows = [{"condition": condition, "label": 1, "score": score} for condition, score in positives]
    # Overlapping negatives are allowed at this operating point.
    rows.extend({"condition": condition, "label": 0, "score": 0.59} for condition in CONDITIONS)
    assert choose_threshold(rows) == pytest.approx(0.31)



def test_expanded_split_preserves_previous_product_folds():
    from types import SimpleNamespace
    from train_doubao_damage import grouped_folds
    from train_doubao_damage_expanded import split_locked

    anchor = [SimpleNamespace(group=f"original-{i}", batch="batch-a") for i in range(10)]
    variants = [SimpleNamespace(group="original-0", batch="batch-b"),
                SimpleNamespace(group="new-one", batch="batch-b"),
                SimpleNamespace(group="new-one", batch="batch-b")]
    for (_, earlier_held), (train, held) in zip(grouped_folds(anchor, 5),
                                                split_locked(anchor + variants, anchor)):
        held_groups = {row.group for row in held}
        assert {row.group for row in earlier_held} <= held_groups
        assert not held_groups.intersection(row.group for row in train)
    assert len([held for _, held in split_locked(anchor + variants, anchor)
                if sum(item.group == "new-one" for item in held) == 2]) == 1


def test_texture_conditioned_features_work_with_small_box():
    import numpy as np

    image = doubao_damage.prepare(sample_image())
    whole, region, _, _ = doubao_damage.extract(
        image, [0.17, 0.27, 0.29, 0.40], extra=("texture_conditioned",)
    )
    assert len(whole["texture_conditioned"]) == 27
    assert len(region["texture_conditioned"]) == 36
    assert np.isfinite(whole["texture_conditioned"]).all()
    assert np.isfinite(region["texture_conditioned"]).all()



def test_detail_scale_can_recover_a_missed_image(tmp_path):
    model = tmp_path / "model.json"
    weak = {"families": ["rgb"], "threshold": 0.6, "stumps": []}
    hit = {"families": ["texture_conditioned"], "threshold": 0.4, "stumps": []}
    payload = {"schema": doubao_damage.SCHEMA, "id": "dual-scale",
               "heads": {"whole": weak}, "detail_heads": {"whole": hit}}
    model.write_text(json.dumps(payload), encoding="utf-8")
    result = doubao_damage.analyze_image(encode(sample_image()), model_path=model)
    assert result["whole"]["status"] == "similar"
    assert result["whole"]["threshold"] == 0.4
    assert result["region"] is None



def test_v4_uses_the_trained_resolution(tmp_path, monkeypatch):
    head = {"families": ["rgb"], "mean": [0]*8, "scale": [1]*8, "weights": [0]*8,
            "intercept": 0.0, "threshold": 0.6}
    package = {"schema": doubao_damage.BALANCED_SCHEMA, "id": "balanced-fixture", "heads": {"whole": head}}
    path = tmp_path / "balanced.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    prepare = doubao_damage.prepare
    called = []

    def observed(image, **kwargs):
        called.append(kwargs.get("short_side"))
        return prepare(image, **kwargs)

    monkeypatch.setattr(doubao_damage, "prepare", observed)
    result = doubao_damage.analyze_image(encode(sample_image()), model_path=path)
    assert called == [288]
    assert result["whole"]["score"] == 0.5
    assert result["whole"]["status"] == "weak"


def test_balanced_calibration_accounts_for_intact_photos_at_each_upload_quality():
    from train_doubao_damage_accuracy import CONDITIONS, pick_cut
    rows = []
    for form, offset in zip(CONDITIONS, (0.0, -0.01, -0.03)):
        rows.extend({"condition": form, "label": 0, "score": value+offset} for value in (0.28, 0.47, 0.49))
        rows.extend({"condition": form, "label": 1, "score": value+offset} for value in (0.60, 0.62, 0.68))
    cut, score = pick_cut(rows)
    assert 0.49 < cut <= 0.57
    assert score == pytest.approx(1.0)


def test_hard_negative_training_reduces_score_for_high_scoring_real_photo():
    import numpy as np
    from train_doubao_damage import Candidate, Pair, train_head

    coordinates = [([-2, 0], [1, 0]), ([-1, 0], [2, 0]), ([-2, 0], [1, -0.5]),
                   ([-1, 0], [1, 0]), ([2, 1], [2, -0.5]), ([1, 1], [1, -1])]
    pairs = []
    for index, (real, generated) in enumerate(coordinates):
        views = {"matched": {"whole": [{"rgb": np.array(real)}, {"rgb": np.array(generated)}]}}
        pairs.append(Pair(f"ai-{index}", f"real-{index}", str(index), "batch", "high", views, (0, 0), "damage"))
    recipe = Candidate("example", ("rgb",), method="paired")
    base = train_head(pairs, "whole", recipe)
    weighted = train_head(pairs, "whole", recipe, hard_negative_weight=2.5)
    real, generated = pairs[4].views["matched"]["whole"]
    assert doubao_damage.predict_vector(weighted, real) < doubao_damage.predict_vector(base, real)
    assert doubao_damage.predict_vector(weighted, generated) > 0.5


def test_accuracy_target_uses_strict_limits_and_counts_undecided_uploads():
    from evaluate_doubao_target import target_metrics

    def prediction(label, score):
        return {"label": label, "score": score, "threshold": 0.5}

    # The first run has precisely one false alert per twenty real photos.
    cohort = [prediction(0, 0.2)] * 19 + [prediction(0, 0.8)] + [prediction(1, 0.8)] * 40
    result = target_metrics(cohort)
    assert result["accuracy"] > 0.95 and result["false_positive_rate"] == 0.05 and not result["passes"]

    cohort[19] = prediction(0, 0.2)
    cohort[20:23] = [prediction(1, 0.2)] * 3
    result = target_metrics(cohort)
    assert result["accuracy"] == 0.95 and result["false_positive_rate"] == 0 and not result["passes"]

    cohort[20] = prediction(1, 0.8)
    assert target_metrics(cohort)["passes"]
    cohort[0] = prediction(0, None)
    result = target_metrics(cohort)
    assert result["undecided"] == 1 and not result["passes"]

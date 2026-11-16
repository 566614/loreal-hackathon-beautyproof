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

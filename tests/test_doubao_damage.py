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

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from beauty_benchmark import sha256, validate
from prepare_detection_images import prepare


def test_uniform_crop_preserves_original_lineage_and_requires_review(tmp_path):
    original, edited, mask = (tmp_path / name for name in ("original.png", "edited.png", "mask.png"))
    Image.new("RGB", (10, 10), "red").save(original)
    Image.new("RGB", (10, 10), "blue").save(edited)
    image = Image.new("L", (10, 10), 0)
    image.putpixel((3, 3), 255)
    image.save(mask)
    common = {"group": "person", "authorization": "test", "split": "train"}
    rows = [dict(common, id="original", kind="real", image=original.name, sha256=sha256(original)),
            dict(common, id="edited", kind="edited", image=edited.name, sha256=sha256(edited),
                 original_sha256=sha256(original), edit="retouch", mask=mask.name, mask_sha256=sha256(mask))]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "prepared"
    prepare(manifest, output, .2)
    result = validate(output / "manifest.json")
    assert result[1]["original_sha256"] == result[0]["sha256"]
    assert all(row["visible_ai_label_reviewed"] is False for row in result)
    with Image.open(output / result[1]["image"]) as pic, Image.open(output / result[1]["mask"]) as label:
        assert pic.size == label.size == (10, 8)
    with Image.open(original) as pic:
        assert pic.size == (10, 10)
    with pytest.raises(ValueError, match="已存在"):
        prepare(manifest, output, .2)


def test_crop_uses_display_orientation_and_rotates_mask_with_image(tmp_path):
    original, edited, mask = (tmp_path / name for name in ("original.jpg", "edited.jpg", "mask.png"))
    exif = Image.Exif()
    exif[274] = 6
    for path, color in ((original, "white"), (edited, "gray")):
        image = Image.new("RGB", (100, 50), color)
        for x in range(80, 100):
            for y in range(50):
                image.putpixel((x, y), (255, 0, 0))
        image.save(path, exif=exif)
    label = Image.new("L", (100, 50), 0)
    label.putpixel((20, 10), 255)
    label.save(mask)
    common = {"group": "person", "authorization": "test", "split": "train"}
    rows = [dict(common, id="original", kind="real", image=original.name, sha256=sha256(original)),
            dict(common, id="edited", kind="edited", image=edited.name, sha256=sha256(edited),
                 original_sha256=sha256(original), edit="retouch", mask=mask.name, mask_sha256=sha256(mask))]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "prepared"
    prepare(manifest, output, .2)
    result = validate(output / "manifest.json")
    with Image.open(output / result[1]["image"]) as image, Image.open(output / result[1]["mask"]) as label:
        assert image.size == label.size == (50, 80)
        assert label.getpixel((39, 20)) == 255
        red, green, blue = image.getpixel((25, 70))
        assert abs(red - green) < 20 and abs(red - blue) < 20
        assert not image.getexif()
    assert result[1]["visible_ai_label_reviewed"] is False

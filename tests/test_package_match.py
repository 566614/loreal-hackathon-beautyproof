import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import package_match
import pipeline


def test_reference_location_and_unrelated_image(tmp_path):
    random = np.random.default_rng(42)
    reference = random.integers(0, 256, (240, 240), dtype=np.uint8)
    target = np.full((400, 450), 128, dtype=np.uint8)
    target[80:320, 100:340] = reference
    ref_path, target_path = tmp_path / "ref.png", tmp_path / "target.png"
    Image.fromarray(reference).save(ref_path)
    Image.fromarray(target).save(target_path)
    found = package_match.locate(target_path, ref_path)
    assert found["status"] == "发现相似图案"
    x1, y1, x2, y2 = found["regions"][0]["bbox"]
    assert abs(x1 - 100) < 8 and abs(y1 - 80) < 8
    assert abs(x2 - 340) < 8 and abs(y2 - 320) < 8
    large_ref = Image.fromarray(reference).resize((960, 960))
    large_target = Image.new("L", (1800, 1600), 128)
    large_target.paste(large_ref, (400, 320))
    large_ref.save(ref_path)
    large_target.save(target_path)
    large = package_match.locate(target_path, ref_path)
    assert large["status"] == "发现相似图案"
    x1, y1, x2, y2 = large["regions"][0]["bbox"]
    assert abs(x1 - 400) < 15 and abs(y1 - 320) < 15
    assert abs(x2 - 1360) < 15 and abs(y2 - 1280) < 15
    Image.new("L", (450, 400), 128).save(target_path)
    assert package_match.locate(target_path, ref_path)["status"] == "无法判断"


def test_reference_is_present_in_analysis_without_affecting_verdict(tmp_path, monkeypatch):
    image, ref = tmp_path / "image.png", tmp_path / "ref.png"
    Image.new("RGB", (80, 80), "white").save(image)
    Image.new("RGB", (80, 80), "black").save(ref)
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    skip = [tool for tool, _, _ in pipeline.TOOL_LIST]
    result = pipeline.analyze(image, text="示例正文", reference_image=ref,
                              skip=skip, verbose=False, publish=False)
    match = result["audit"]["crossmodal"]["package_reference"]
    assert match["status"] == "无法判断"
    assert len(match["reference_sha256"]) == 64
    assert not match["regions"]

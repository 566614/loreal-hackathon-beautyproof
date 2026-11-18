import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import doubao_compact


@pytest.mark.parametrize("codec", ["row-bytes", "row-bits", "delta-bytes", "delta-bits"])
def test_packed_model_preserves_signed_prototypes_and_score(tmp_path, codec):
    dimensions = doubao_compact.DIMENSIONS
    codes = (np.arange(4*dimensions) % 15).astype(np.uint8).reshape(1, 4, dimensions)
    encoded = codes.copy()
    if codec.startswith("delta"):
        encoded[:, 1:] ^= encoded[:, :1]
    data = encoded.ravel()
    if codec.endswith("bits"):
        data = np.packbits((data[:, None] >> np.arange(4, dtype=np.uint8)) & 1)
    header = {"schema": doubao_compact.SCHEMA, "bits": 4, "views": 4, "offset": 7,
              "scale": .5, "shape": [4, dimensions], "codec": codec, "threshold": .125}
    encoded_header = json.dumps(header).encode()
    mean = np.zeros(dimensions, dtype="<f2")
    spread = np.ones(dimensions, dtype="<f2")
    coefficients = np.asarray([-1., .25, -.5, 1.], dtype="<f2")
    payload = (len(encoded_header).to_bytes(4, "little") + encoded_header + mean.tobytes() +
               spread.tobytes() + coefficients.tobytes() + data.tobytes())
    path = tmp_path / "model.bin"
    path.write_bytes(zlib.compress(payload))

    model = doubao_compact.load_model(path)
    centers = (codes.reshape(4, dimensions).astype(np.float32)-7)*.5
    np.testing.assert_array_equal(model["centers"], centers)
    expected = np.exp(-.62*(centers**2).sum(1)/dimensions)@coefficients.astype(np.float32)
    actual = doubao_compact.score_features(model, np.zeros(dimensions, dtype=np.float32))
    assert actual == expected
    assert model["threshold"] == .125


def test_uncalibrated_model_cannot_silently_classify_images(tmp_path):
    header = {"schema": doubao_compact.SCHEMA, "bits": 4, "views": 4, "offset": 7,
              "scale": .5, "shape": [4, doubao_compact.DIMENSIONS],
              "codec": "row-bytes", "threshold": None}
    encoded = json.dumps(header).encode()
    path = tmp_path / "uncalibrated.bin"
    path.write_bytes(zlib.compress(len(encoded).to_bytes(4, "little")+encoded))
    with pytest.raises(ValueError, match="校准阈值"):
        doubao_compact.load_model(path)

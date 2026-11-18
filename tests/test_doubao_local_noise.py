from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import doubao_local_noise


def parameters():
    generator = np.random.default_rng(20261119)
    model = {"schema": np.asarray(doubao_local_noise.SCHEMA),
             "mean": np.zeros(78, dtype=np.float32),
             "spread": np.ones(78, dtype=np.float32),
             "threshold": np.asarray(.5)}
    for index, inputs, outputs in ((0, 78, 32), (3, 32, 16), (5, 16, 1)):
        model[f"layers.{index}.weight"] = generator.normal(0, .1, (outputs, inputs)).astype(np.float32)
        model[f"layers.{index}.bias"] = generator.normal(0, .1, outputs).astype(np.float32)
    return model


def test_region_order_cannot_change_local_evidence_pooling():
    model = parameters()
    generator = np.random.default_rng(20261119)
    regions = generator.normal(size=(2, 256, 78)).astype(np.float32)
    shuffled = regions[:, generator.permutation(256)]
    original = doubao_local_noise.score_features(model, regions)
    permuted = doubao_local_noise.score_features(model, shuffled)
    np.testing.assert_allclose(original, permuted, rtol=0, atol=2e-6)


def test_local_model_requires_a_calibrated_threshold(tmp_path):
    model = parameters()
    del model["threshold"]
    path = tmp_path/"uncalibrated.npz"
    np.savez_compressed(path, **model)
    with pytest.raises(ValueError, match="校准阈值"):
        doubao_local_noise.load_model(path)

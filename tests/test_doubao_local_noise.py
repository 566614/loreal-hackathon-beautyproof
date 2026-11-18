from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import doubao_local_noise


def parameters(calibrated=False):
    generator = np.random.default_rng(20261119)
    model = {"schema": np.asarray(doubao_local_noise.SCHEMA),
             "mean": np.zeros(78, dtype=np.float32),
             "spread": np.ones(78, dtype=np.float32),
             "threshold": np.asarray(.5)}
    for index, inputs, outputs in ((0, 78, 32), (3, 32, 16), (5, 16, 1)):
        model[f"layers.{index}.weight"] = generator.normal(0, .1, (outputs, inputs)).astype(np.float32)
        model[f"layers.{index}.bias"] = generator.normal(0, .1, outputs).astype(np.float32)
    if calibrated:
        model.update({"schema": np.asarray(doubao_local_noise.QUALITY_SCHEMA),
                      "quality_mean": np.zeros(156, dtype=np.float32),
                      "quality_spread": np.ones(156, dtype=np.float32),
                      "quality_weights": generator.normal(0, .1, (156, 4)).astype(np.float32),
                      "quality_temperature": np.asarray(.35),
                      "quality_center": np.asarray([-8., -6., -3., -2.]),
                      "quality_width": np.asarray([2., 2., 3., 3.])})
    return model


@pytest.mark.parametrize("calibrated", [False, True])
def test_region_order_cannot_change_local_evidence_pooling(calibrated):
    model = parameters(calibrated)
    generator = np.random.default_rng(20261119)
    regions = generator.normal(size=(2, 256, 78)).astype(np.float32)
    shuffled = regions[:, generator.permutation(256)]
    original = doubao_local_noise.score_features(model, regions)
    permuted = doubao_local_noise.score_features(model, shuffled)
    np.testing.assert_allclose(original, permuted, rtol=0, atol=2e-6)


def test_neutral_quality_calibration_preserves_the_network_score():
    generator = np.random.default_rng(20261119)
    regions = generator.normal(size=(2, 256, 78)).astype(np.float32)
    original = doubao_local_noise.score_features(parameters(), regions)
    model = parameters(calibrated=True)
    model["quality_center"] = np.zeros(4)
    model["quality_width"] = np.ones(4)
    adjusted = doubao_local_noise.score_features(model, regions)
    np.testing.assert_allclose(original, adjusted, rtol=0, atol=2e-6)


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("roi", [[.09, .09, .1, .1], [.2, .2, .8, .8]])
def test_selected_regions_share_context_and_cannot_raise_the_whole_image_score(calibrated, roi):
    model = parameters(calibrated)
    generator = np.random.default_rng(20261119)
    regions = generator.normal(size=(2, 256, 78)).astype(np.float32)
    whole = doubao_local_noise.score_features(model, regions)
    selected = doubao_local_noise.region_indices(roi)
    local = doubao_local_noise.score_features(model, regions, selected_regions=selected)
    assert np.all(local <= whole)
    full_frame = doubao_local_noise.region_indices([0, 0, 1, 1])
    repeated = doubao_local_noise.score_features(model, regions, selected_regions=full_frame)
    np.testing.assert_array_equal(whole, repeated)


def test_quality_calibration_rejects_zero_spread(tmp_path):
    model = parameters(calibrated=True)
    model["quality_width"][0] = 0
    path = tmp_path/"invalid-quality.npz"
    np.savez_compressed(path, **model)
    with pytest.raises(ValueError, match="画质校准"):
        doubao_local_noise.load_model(path)


def test_local_model_requires_a_calibrated_threshold(tmp_path):
    model = parameters()
    del model["threshold"]
    path = tmp_path/"uncalibrated.npz"
    np.savez_compressed(path, **model)
    with pytest.raises(ValueError, match="校准阈值"):
        doubao_local_noise.load_model(path)


def test_partial_weight_archive_is_a_model_loading_error(tmp_path):
    path = tmp_path/"partial.npz"
    np.savez_compressed(path, **parameters())
    path.write_bytes(path.read_bytes()[:-64])
    with pytest.raises(ValueError, match="模型文件损坏"):
        doubao_local_noise.load_model(path)

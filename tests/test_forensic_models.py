"""Verify IML-ViT receives the same padded pixel values as its official pipeline."""
import sys
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import forensic_models
from forensic_models import _iml_preprocess


def test_iml_pads_black_before_normalization():
    image = Image.new("RGB", (12, 8), "white")
    tensor, size = _iml_preprocess(image)
    assert size == (12, 8)
    assert tuple(tensor.shape) == (1, 3, 1024, 1024)
    expected_black = torch.tensor([-.485 / .229, -.456 / .224, -.406 / .225])
    expected_white = torch.tensor([(1 - .485) / .229, (1 - .456) / .224, (1 - .406) / .225])
    torch.testing.assert_close(tensor[0, :, 20, 20], expected_black)
    torch.testing.assert_close(tensor[0, :, 0, 0], expected_white)


def test_iml_rejects_insufficient_system_and_gpu_memory(monkeypatch):
    monkeypatch.setattr(forensic_models, "_available_system_memory", lambda: 4 * 1024 ** 3)
    with pytest.raises(RuntimeError, match="可用系统内存"):
        forensic_models._check_iml_memory(torch, "cuda")
    with pytest.raises(RuntimeError, match="可用系统内存"):
        forensic_models.iml_vit("unused.png", Path("."))
    monkeypatch.setattr(forensic_models, "_available_system_memory", lambda: 8 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4 * 1024 ** 3, 8 * 1024 ** 3))
    with pytest.raises(RuntimeError, match="可用显存"):
        forensic_models._check_iml_memory(torch, "cuda")


def test_dda_rejects_insufficient_system_and_gpu_memory(monkeypatch):
    monkeypatch.setattr(forensic_models, "_available_system_memory", lambda: 3 * 1024 ** 3)
    with pytest.raises(RuntimeError, match="可用系统内存"):
        forensic_models._check_dda_memory(torch, "cuda")
    with pytest.raises(RuntimeError, match="可用系统内存"):
        forensic_models.dda("unused.png")
    monkeypatch.setattr(forensic_models, "_available_system_memory", lambda: 8 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 * 1024 ** 3, 8 * 1024 ** 3))
    with pytest.raises(RuntimeError, match="可用显存"):
        forensic_models._check_dda_memory(torch, "cuda")

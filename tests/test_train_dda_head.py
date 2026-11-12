"""Exercise the classifier training without requiring forensic model weights."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_dda_head


def test_extract_requires_visible_ai_label_review(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    rows = [{"id": "sample", "kind": "generated", "split": "train"}]
    monkeypatch.setattr(train_dda_head, "validate", lambda path: rows)
    with pytest.raises(ValueError, match="可见 AI 标识"):
        train_dda_head.extract(manifest, tmp_path / "features.pt")


def test_train_head_uses_training_features_and_checks_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    weights = tmp_path / "DDA_ckpt.pth"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(train_dda_head, "_root", lambda: tmp_path)
    monkeypatch.setattr(train_dda_head, "validate", lambda path: [])
    rows = [{"group": "r", "kind": "real", "split": "train", "visible_ai_label_reviewed": True, "visible_ai_label_present": False},
            {"group": "g", "kind": "generated", "split": "train", "visible_ai_label_reviewed": True, "visible_ai_label_present": False},
            {"group": "c", "kind": "generated", "split": "calibration", "visible_ai_label_reviewed": True, "visible_ai_label_present": False}]
    cache = tmp_path / "features.pt"
    torch.save({"features": torch.tensor([[-1., 0.], [1., 0.], [-100., 0.]]), "rows": rows,
                "manifest_sha256": train_dda_head.sha256(manifest),
                "backbone_sha256": train_dda_head.file_hash(weights), "preprocessing": "test"}, cache)
    output = tmp_path / "head.pt"
    train_dda_head.train(cache, manifest, output)
    head = torch.load(output, weights_only=True)
    assert head["head"]["weight"][0, 0] > 0
    manifest.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="清单不一致"):
        train_dda_head.train(cache, manifest, output)

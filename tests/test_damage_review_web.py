"""Request contract for the independent damage reviewer."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
import app


def test_independent_review_forwards_upload_and_region_by_contract(monkeypatch):
    def analyze_image(data, roi):
        assert data == b"sample"
        assert roi == [0.15, 0.35, 0.8, 0.9]
        return {"device": "cpu", "whole": {"status": "insufficient"}}

    monkeypatch.setitem(sys.modules, "doubao_damage", SimpleNamespace(analyze_image=analyze_image))

    def normal_analysis(*_args, **_kwargs):
        raise AssertionError("The regular image pipeline did not belong to this request")

    monkeypatch.setattr(app.pipeline, "analyze", normal_analysis)
    with app.app.test_client() as client:
        response = client.post("/api/damage-review", data={
            "file": (io.BytesIO(b"sample"), "item.jpg"), "roi": "[0.15, 0.35, 0.8, 0.9]",
        })
    assert response.status_code == 200
    assert response.get_json()["device"] == "cpu"


def test_missing_file_and_malformed_region_are_client_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "doubao_damage", SimpleNamespace(analyze_image=None))
    with app.app.test_client() as client:
        assert client.post("/api/damage-review", data={}).status_code == 400
        response = client.post("/api/damage-review", data={
            "file": (io.BytesIO(b"sample"), "item.jpg"), "roi": "[invalid",
        })
    assert response.status_code == 400
    assert "坐标" in response.get_json()["error"]

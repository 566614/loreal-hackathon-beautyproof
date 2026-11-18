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


def test_review_endpoint_uses_packaged_cpu_noise_model_and_preserves_region_contract(monkeypatch):
    from PIL import Image, ImageDraw
    import doubao_damage
    from doubao_local_noise import analyze_image, load_model

    image = Image.new("RGB", (480, 320), (214, 189, 154))
    ImageDraw.Draw(image).line([(65, 70), (350, 245)], fill="#352118", width=5)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    data = stream.getvalue()
    expected = analyze_image(data, load_model(doubao_damage.REVIEW_MODEL_PATH))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("The local-noise request must use its packaged NumPy model")

    monkeypatch.setattr(app.pipeline, "analyze", unexpected)
    monkeypatch.setattr(doubao_damage, "load_model", unexpected)
    with app.app.test_client() as client:
        response = client.post("/api/damage-review", data={
            "file": (io.BytesIO(data), "item.jpg"), "roi": "[0.2,0.2,0.8,0.8]",
        })
    assert response.status_code == 200
    result = response.get_json()
    assert result["model"] == expected["model"]
    assert result["device"] == "cpu" and result["evaluation_scope"] == "original_files"
    assert result["whole"]["score"] == expected["score"]
    assert result["whole"]["threshold"] == expected["threshold"]
    assert result["region"]["score"] <= result["whole"]["score"]
    assert result["region"]["cues"] == []
    assert result["edge_preview_png"]
    assert result["image_size"] == [480, 320]
    assert result["roi"] == [.2, .2, .8, .8]

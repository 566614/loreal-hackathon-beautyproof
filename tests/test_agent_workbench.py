"""Exercise the public turn API with local models or controlled service boundaries."""

import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web import app as web

import doubao_damage
import llm_explainer
import pipeline


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "JOBS", {})
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "catalog.json"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return web.app.test_client()


def photo(color="white"):
    data = io.BytesIO()
    Image.new("RGB", (80, 80), color).save(data, format="PNG")
    return io.BytesIO(data.getvalue()), "item.png"


def turn(client, message="", previous=None, **data):
    if previous:
        data["previous_job_id"] = previous
    response = client.post("/api/agent/turn", data={"message": message, **data})
    assert response.status_code == 200, response.json
    job_id = response.json["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = client.get("/api/job/" + job_id).json
        if snapshot["status"] != "running":
            assert snapshot["status"] == "done", snapshot
            return job_id, snapshot["result"]
        time.sleep(.01)
    pytest.fail("审核回合未完成")


def test_text_review_and_grounded_follow_up(client):
    source = "【青蓝精华】2025版14天水分提升20%。【青蓝精华】2025版14天水分提升40%。"
    job, result = turn(client, "核验宣传文案：" + source)
    assert result["type"] == "review"
    assert result["reports"]["publicity"]["audit"]["context_conflicts"]
    job, answer = turn(client, "哪一句有冲突？", previous=job)
    assert answer["type"] == "answer"
    assert answer["evidence"][0]["quote"] in source
    assert answer["evidence"][0]["detail"] in source
    assert turn(client, "为什么？", previous=job)[1]["type"] == "answer"


def test_image_only_clarifies_and_upload_completes_damage(client):
    _, result = turn(client, file=photo())
    assert result["type"] == "clarify"
    assert "退款破损" in result["message"]
    job, result = turn(client, "审核退款破损")
    assert result["type"] == "clarify"
    assert "附上" in result["message"]
    job, result = turn(client, "我现在上传图片", previous=job, file=photo())
    assert result["route"] == "damage"
    assert result["reports"]["damage"]["preview"].startswith("data:image/jpeg;")
    assert turn(client, "破损分数是什么？", previous=job)[1]["evidence"][0]["label"].startswith("整图")


def test_dual_route_reuses_the_image_and_cleans_scratch(client, monkeypatch):
    seen = []

    def inspect_image(script, path, **kwargs):
        seen.append((kwargs["output_dir"], path.read_bytes()))
        return SimpleNamespace(returncode=1, stderr="")

    def damage(data, roi):
        assert roi == [.1, .2, .7, .8]
        assert data == seen[0][1]
        return {"whole": {"status": "weak", "score": .1, "reason": "本地模型结果"}}

    monkeypatch.setattr(pipeline, "run_tool", inspect_image)
    monkeypatch.setattr(doubao_damage, "analyze_image", damage)
    _, result = turn(client, "同时检查宣传和退款破损", file=photo(), roi="[0.1, 0.2, 0.7, 0.8]")
    assert set(result["reports"]) == {"publicity", "damage"}
    assert "_output_file" not in result["reports"]["publicity"]
    assert not seen[0][0].parent.exists()


def test_new_image_does_not_answer_from_old_evidence(client):
    job, _ = turn(client, "看一下退款破损", file=photo())
    job, result = turn(client, "为什么？", previous=job, file=photo("red"))
    assert result["type"] == "clarify"
    assert "evidence" not in result
    assert turn(client, "只查破损", previous=job, file=photo("red"))[1]["route"] == "damage"


def test_failed_specialist_keeps_other_report(client, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("图片有效像素不足。")

    monkeypatch.setattr(pipeline, "run_tool", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=""))
    monkeypatch.setattr(doubao_damage, "analyze_image", fail)
    _, result = turn(client, "两项都查，宣传核验和退款破损", file=photo())
    assert set(result["reports"]) == {"publicity"}
    assert result["errors"] == {"damage": "图片有效像素不足。"}


def test_model_is_optional_and_timeout_falls_back(client, monkeypatch):
    calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")

    def timeout(request, timeout):
        calls.append((json.loads(request.data), timeout))
        raise TimeoutError

    monkeypatch.setattr(llm_explainer.urllib.request, "urlopen", timeout)
    _, result = turn(client, "帮我看一下", file=photo(), llm="0")
    assert result["type"] == "clarify"
    assert not calls
    _, result = turn(client, "帮我看一下", file=photo())
    assert result["type"] == "clarify"
    payload, timeout_seconds = calls[0]
    assert "帮我看一下" in payload["messages"][0]["content"]
    assert "data:image" not in payload["messages"][0]["content"]
    assert timeout_seconds == 10
    _, result = turn(client, "核验宣传：七天治愈痘痘。")
    assert result["type"] == "review"
    assert result["reports"]["publicity"]["audit"]["checks"]["claim_model"]["status"] == "model_unavailable"


def test_enabled_model_routes_and_only_selects_existing_image_checks(client, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setattr(pipeline, "run_tool", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=""))
    calls = []

    def respond(request, timeout):
        prompt = json.loads(request.data)["messages"][-1]["content"]
        calls.append(prompt)
        response = {"route": "publicity", "copy": "伪造的图片文案"} if len(calls) == 1 else {
            "check_ids": ["ocr", "unregistered_tool"],
        }
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(response)}}]}).encode())

    monkeypatch.setattr(llm_explainer.urllib.request, "urlopen", respond)
    _, result = turn(client, "帮我审一下这张素材", file=photo())
    report = result["reports"]["publicity"]
    assert report["text_input"] is None
    assert report["audit"]["image_review_checks"] == ["ocr"]
    assert len(calls) == 2 and all("data:image/" not in prompt for prompt in calls)


def test_rejects_invalid_upload_and_expired_context(client):
    bad = client.post("/api/agent/turn", data={"file": (io.BytesIO(b"invalid"), "item.png")})
    assert bad.status_code == 400
    expired = client.post("/api/agent/turn", data={"message": "为什么？", "previous_job_id": "expired"})
    assert expired.status_code == 409

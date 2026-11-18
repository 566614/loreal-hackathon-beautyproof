"""Exercise the public turn API with local models or controlled service boundaries."""

import io
import json
import sys
import threading
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
import review_agents


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


HARNESS_COPY = "【青蓝精华】2025版14天水分提升20%。"


def add_study(client, **changes):
    data = {
        "source": "合成测试报告", "excerpt": HARNESS_COPY, "product": "青蓝精华",
        "version": "报告1", "locator": "第2页", "reviewer": "测试审核员",
        "source_type": "efficacy_study", "market": "中国", "product_version": "2025版",
        "reviewer_confirmed": True, **changes,
    }
    response = client.post("/api/reference-catalog", json=data)
    assert response.status_code == 201
    return response.json["reference"]["id"]


def test_harness_followup_support_and_single_use(client, monkeypatch):
    job, initial = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国")
    original = initial["reports"]["publicity"]
    assert original["harness_trace"]["state"] == "waiting_evidence"
    ref = add_study(client)
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **kw: pytest.fail("补证不应重复检测"))
    _, result = turn(client, previous=job, role="reviewer", evidence_followup="1", reference_ids=json.dumps([ref]))
    report = result["reports"]["publicity"]
    assert report["evidence_comparison"][0]["before"] == "证据不足"
    assert report["evidence_comparison"][0]["after"] == "有证据支持"
    assert report["evidence_comparison"][0]["added_references"][0]["id"] == ref
    assert report["harness_trace"]["decision"]["action"] == "finish"
    assert original["audit"]["claims"][0]["status"] == "证据不足"
    duplicate = client.post("/api/agent/turn", data={
        "previous_job_id": job, "role": "reviewer", "evidence_followup": "1", "reference_ids": json.dumps([ref]),
    })
    assert duplicate.status_code == 409


@pytest.mark.parametrize("changes, expected", [
    ({"product_version": "2024版", "excerpt": HARNESS_COPY.replace("2025", "2024")}, "证据不足"),
    ({"market": "美国"}, "证据不足"),
    ({"source_type": "brand_claim"}, "证据不足"),
    ({"excerpt": HARNESS_COPY.replace("20%", "40%")}, "存在冲突"),
])
def test_followup_scope_and_conflicts(client, changes, expected):
    job, _ = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国")
    ref = add_study(client, **changes)
    _, result = turn(client, previous=job, role="reviewer", evidence_followup="1", reference_ids=json.dumps([ref]))
    report = result["reports"]["publicity"]
    assert report["audit"]["claims"][0]["status"] == expected
    assert report["harness_trace"]["decision"]["action"] == "human_review"


def test_followup_rejects_changed_or_unverified_materials(client):
    job, _ = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国")
    ref = add_study(client)
    base = {"previous_job_id": job, "role": "reviewer", "evidence_followup": "1", "reference_ids": json.dumps([ref])}
    for changes in ({"copy": "新的宣传"}, {"market": "美国"}, {"product": "其他精华"},
                    {"file": photo()}, {"roi": "[]"}, {"reference_ids": '["missing"]'},
                    {"reference_ids": "[]"}, {"references": json.dumps([{"source": "自述", "excerpt": HARNESS_COPY}])}):
        assert client.post("/api/agent/turn", data={**base, **changes}).status_code == 400
    assert client.post("/api/agent/turn", data={**base, "previous_job_id": "expired"}).status_code == 409
    assert client.post("/api/agent/turn", data={**base, "role": "consumer"}).status_code == 409


def test_followup_failure_releases_reservation_and_followup_does_not_need_model(client, monkeypatch):
    job, _ = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国")
    ref = add_study(client)
    original = pipeline.reassess_references
    monkeypatch.setattr(pipeline, "reassess_references", lambda *a: (_ for _ in ()).throw(OSError("read failed")))
    data = {"previous_job_id": job, "role": "reviewer", "evidence_followup": "1", "reference_ids": json.dumps([ref])}
    response = client.post("/api/agent/turn", data=data)
    failed_id = response.json["job_id"]
    for _ in range(200):
        if client.get("/api/job/" + failed_id).json["status"] == "error":
            break
        time.sleep(.01)
    assert web.JOBS[job]["context"]["evidence_session"]["followup_job_id"] is None
    monkeypatch.setattr(pipeline, "reassess_references", original)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm_explainer, "deepseek_json", lambda *a, **kw: pytest.fail("续审应离线执行"))
    assert turn(client, **data)[1]["reports"]["publicity"]["audit"]["claims"][0]["status"] == "有证据支持"


def test_followup_reuses_image_snapshot_after_cleanup(client, monkeypatch):
    def image_review(path, copy, options, directory, model, progress, model_lock):
        report = pipeline.analyze(text=copy, publish=False, run_root=directory)
        report["image"] = {"name": "item.png", "sha256": "snapshot"}
        report["audit"]["checks"]["ocr"] = {"status": "completed"}
        return report
    monkeypatch.setattr(review_agents, "publicity_agent", image_review)
    job, initial = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国", file=photo())
    assert initial["reports"]["publicity"]["harness_trace"]["state"] == "waiting_evidence"
    ref = add_study(client)
    _, result = turn(client, previous=job, role="reviewer", evidence_followup="1", reference_ids=json.dumps([ref]))
    assert result["reports"]["publicity"]["image"]["sha256"] == "snapshot"


def test_concurrent_followup_and_answer_cannot_bypass_single_use(client, monkeypatch):
    job, _ = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国")
    answer_job, _ = turn(client, "为什么？", previous=job, role="reviewer")
    ref = add_study(client)
    started, release = threading.Event(), threading.Event()
    original = pipeline.reassess_references

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(pipeline, "reassess_references", blocked)
    data = {"previous_job_id": job, "role": "reviewer", "evidence_followup": "1", "reference_ids": json.dumps([ref])}
    response = client.post("/api/agent/turn", data=data)
    try:
        assert started.wait(5)
        assert client.post("/api/agent/turn", data={**data, "previous_job_id": answer_job}).status_code == 409
    finally:
        release.set()
    for _ in range(200):
        state = client.get("/api/job/" + response.json["job_id"]).json["status"]
        if state == "done":
            break
        time.sleep(.01)
    assert state == "done"


def test_followup_preserves_other_previously_cited_studies(client):
    existing = add_study(client)
    source = HARNESS_COPY + HARNESS_COPY.replace("14天", "7天")
    job, _ = turn(client, "核验宣传文案：" + source, role="reviewer", market="中国")
    added = add_study(client, excerpt=HARNESS_COPY.replace("14天", "7天"))
    _, result = turn(client, previous=job, role="reviewer", evidence_followup="1", reference_ids=json.dumps([added]))
    claims = result["reports"]["publicity"]["audit"]["claims"]
    assert [c["status"] for c in claims] == ["有证据支持", "有证据支持"]
    assert existing in {r["id"] for r in claims[0]["references"]}


def test_model_unavailable_keeps_local_harness_and_followup(client, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm_explainer, "deepseek_json", lambda *a, **kw: None)
    job, initial = turn(client, "核验宣传文案：" + HARNESS_COPY, role="reviewer", market="中国")
    report = initial["reports"]["publicity"]
    assert report["audit"]["checks"]["claim_model"]["status"] == "model_unavailable"
    assert report["harness_trace"]["state"] == "waiting_evidence"
    ref = add_study(client)
    _, result = turn(client, previous=job, role="reviewer", evidence_followup="1", reference_ids=json.dumps([ref]))
    assert result["reports"]["publicity"]["evidence_comparison"][0]["after"] == "有证据支持"


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


def test_reviewer_turn_scores_original_upload_with_packaged_noise_model(client, monkeypatch):
    from doubao_local_noise import analyze_image, load_model

    stream = io.BytesIO()
    Image.effect_noise((480, 320), 35).convert("RGB").save(stream, format="JPEG")
    data = stream.getvalue()
    expected = analyze_image(data, load_model(doubao_damage.REVIEW_MODEL_PATH))

    def unexpected(*args, **kwargs):
        pytest.fail("破损回合应只使用本地噪声模型")

    monkeypatch.setattr(pipeline, "analyze", unexpected)
    monkeypatch.setattr(pipeline, "run_tool", unexpected)
    _, result = turn(client, "审核退款破损", role="reviewer", llm="0",
                     file=(io.BytesIO(data), "item.jpg"), roi="[0.2,0.2,0.8,0.8]")
    assert result["route"] == "damage" and result["errors"] == {}
    report = result["reports"]["damage"]
    assert report["model"] == expected["model"]
    assert report["device"] == "cpu" and report["evaluation_scope"] == "original_files"
    assert report["whole"]["score"] == expected["score"]
    assert report["whole"]["threshold"] == expected["threshold"]
    assert report["region"]["status"] == "insufficient" and report["region"]["score"] is None
    assert "整图噪声分布" in report["region"]["reason"]
    assert report["preview"].startswith("data:image/jpeg;") and report["edge_preview_png"]


@pytest.mark.parametrize("request_text", ["检查售后破损照片", "同时检查宣传和退款破损"])
def test_consumer_turn_does_not_dispatch_damage(client, request_text):
    _, result = turn(client, request_text, role="consumer", file=photo())
    assert result["type"] == "clarify"
    assert "消费者入口支持美妆宣传核验" in result["message"]


def test_consumer_photo_defaults_to_publicity(client, monkeypatch):
    monkeypatch.setattr(pipeline, "run_tool", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=""))
    _, result = turn(client, role="consumer", file=photo())
    assert result["route"] == "publicity"
    assert set(result["reports"]) == {"publicity"}


def test_consumer_cannot_resume_reviewer_turn(client):
    job, result = turn(client, "审核退款破损", role="reviewer")
    assert result["type"] == "clarify"
    response = client.post("/api/agent/turn", data={
        "message": "为什么？", "previous_job_id": job, "role": "consumer",
    })
    assert response.status_code == 409


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

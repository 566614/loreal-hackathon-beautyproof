"""Consumer evidence contracts, multi-image context and unsupported model output."""
import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from werkzeug.datastructures import MultiDict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from web import app as web
import seed_review
import review_agents


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (80, 100), color).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def case_client(monkeypatch):
    images = [png("red"), png("blue")]
    case = copy.deepcopy(seed_review.reviewed_case())
    for asset, data in zip(case["assets"], images):
        asset["sha256"] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(seed_review, "reviewed_case", lambda: case)
    monkeypatch.setattr(web, "JOBS", {})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return web.app.test_client(), images


def send(client, data):
    reply = client.post("/api/agent/turn", data=MultiDict(data))
    assert reply.status_code == 200, reply.json
    job = reply.json["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get("/api/job/" + job).json
        if result["status"] != "running":
            assert result["status"] == "done", result
            return job, result["result"]
        time.sleep(.01)
    pytest.fail("种草回合超时")


def test_two_assets_keep_comment_sources_and_grounded_followup(case_client):
    client, images = case_client
    job, result = send(client, [("role", "consumer"), ("message", "核验种草"),
                               *(('file', (io.BytesIO(data), f'{i}.png')) for i, data in enumerate(images))])
    report = result["reports"]["publicity"]
    assert len(report["assets"]) == 2
    assert report["source_counts"]["comment"] >= 7
    assert {e["asset_id"] for e in report["evidence"]} == {"image-1", "image-2"}
    assert not report["scope"]["image_origin_verified"]
    assert any(f["category"] == "credible" for f in report["findings"])
    _, response = send(client, [("role", "consumer"), ("previous_job_id", job), ("message", "这些评论都是水军吗？")])
    assert response["type"] == "answer"
    assert response["evidence"]
    comment_quotes = [e["text"] for e in report["evidence"] if e["source"] == "comment"]
    assert all(any(quote in e["quote"] for quote in comment_quotes) for e in response["evidence"])
    assert any("无法" in e["detail"] or "不足" in e["detail"] for e in response["evidence"])


def test_replacing_only_second_image_discards_old_report(case_client):
    client, images = case_client
    fields = [("role", "consumer"), ("message", "核验种草")]
    job, _ = send(client, fields + [("file", (io.BytesIO(data), f"{i}.png")) for i, data in enumerate(images)])
    _, result = send(client, [("role", "consumer"), ("previous_job_id", job), ("message", "为什么？"),
                             ("file", (io.BytesIO(images[0]), "first.png")), ("file", (io.BytesIO(png("green")), "changed.png"))])
    assert result["type"] != "answer"


def test_comments_only_are_reviewable_and_reviewer_rejects_multiple(case_client):
    client, images = case_client
    _, result = send(client, [("role", "consumer"), ("comments", "作者说私信购买，评论都是求链接")])
    report = result["reports"]["publicity"]
    assert report["source_counts"]["comment"] == 1
    assert any(f["category"] == "comments" for f in report["findings"])
    response = client.post("/api/agent/turn", data=MultiDict([
        ("role", "reviewer"), *(('file', (io.BytesIO(data), f'{i}.png')) for i, data in enumerate(images))]))
    assert response.status_code == 400


def test_reviewer_image_defaults_to_local_damage(case_client):
    client, images = case_client
    _, result = send(client, [("role", "reviewer"), ("file", (io.BytesIO(images[0]), "item.png"))])
    assert set(result["reports"]) == {"damage"}
    assert result["reports"]["damage"]["device"] == "cpu"


def test_consumer_plain_text_does_not_need_a_routing_keyword(case_client):
    client, _ = case_client
    _, result = send(client, [("role", "consumer"), ("message", "姐妹闭眼入，这个真的超级好用")])
    report = result["reports"]["publicity"]
    assert report["source_counts"]["body"] == 1
    assert any(f["category"] == "promotion" for f in report["findings"])


def test_deepseek_refinement_requires_real_evidence_and_reference_ids(tmp_path, monkeypatch):
    path = tmp_path / "test.png"
    path.write_bytes(png("white"))
    calls = []
    def request(prompt, system):
        calls.append(json.loads(prompt.split("\n", 1)[1]))
        return {"conclusion_confidence": .87, "findings": [
            {"category": "promotion", "title": "原文中的广告披露", "analysis": "正文写明品牌合作。",
             "action": "按商业推广阅读。", "evidence_ids": ["body-0"], "reference_ids": []},
            {"category": "comments", "title": "虚构评论", "analysis": "水军", "action": "删除",
             "evidence_ids": ["unknown"], "reference_ids": []},
            {"category": "efficacy", "title": "虚构报告", "analysis": "已证实", "action": "购买",
             "evidence_ids": ["body-0"], "reference_ids": ["fake-study"]},
        ]}
    result = seed_review.analyze([path], [path.name], "本篇为品牌合作", "", "油皮", request, lambda *a: None)
    assert result["engine"] == "deepseek"
    assert result["conclusion"]["confidence"] == .87
    assert result["conclusion"]["confidence_kind"] == "agent_estimate"
    assert result["analysis_process"]["engine"] == "deepseek"
    assert len(result["analysis_process"]["steps"]) >= 3
    assert len(result["findings"]) == 1
    assert calls[0]["personal_conditions"] == "油皮"
    assert all(r["kind"] != "efficacy_study" for r in calls[0]["references"])
    assert "image" not in calls[0]


def test_damage_explanation_keeps_detector_result_and_uses_known_evidence_only():
    result = {"whole": {"status": "similar", "score": .2, "threshold": .14, "reason": "达到复核阈值。"},
              "image_size": [1000, 800], "roi": None, "region": None, "model": "test-local"}
    original = copy.deepcopy(result)
    class Model:
        remaining = 1
        def request(self, prompt, system, timeout):
            assert "未看原图" in prompt
            return {"analysis": [
                {"title": "比较阈值", "detail": "本地分数达到复核阈值，追加来源核查。", "evidence_ids": ["whole"]},
                {"title": "虚构观察", "detail": "裂缝呈现明显错误。", "evidence_ids": ["unknown-pixels"]},
            ], "reasoning_content": "not-for-display"}
    process = review_agents.damage_process(result, Model(), lambda *a: None)
    assert result == original
    assert process["engine"] == "deepseek"
    assert any(step["title"] == "比较阈值" for step in process["steps"])
    assert not any(step["title"] == "虚构观察" for step in process["steps"])
    assert "reasoning_content" not in json.dumps(process)

"""Regression cases for the shared claim audit and isolated image runs."""

import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import content_check
import claim_review
import crossmodal_tool
import llm_explainer
import pipeline
import rule_engine
from web import app as image_app
from web import content_app


def image_evidence(ocr="", model="beautyproof_aigen_v4", score=0.9154):
    return {
        "ocr": {"evidence": [{"text": ocr}]},
        "aigc": {"evidence": [{"model": model, "aigc_score": score, "available": True}]},
        "trufor": {"evidence": [{"available": False, "trufor_score": None}]},
    }


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "knowledge.json"))
    return tmp_path


def test_text_findings_survive_both_entry_points(isolated):
    report = content_check.check_one("七天治愈痘痘，效果最佳。")
    assert report["verdict"] == "suspicious"
    assert report["claims_high"][0]["word"] == "治愈"
    assert report["claims_medium"][0]["word"] == "最佳"
    assert report["audit"]["claims"][0]["status"] == "证据不足"
    report = content_check.check_one("本品不是药品，不能治疗皮炎。评论宣称「七天治愈」。")
    assert report["verdict"] == "inconclusive"
    assert not report["claims_high"]


def test_questions_and_explicit_disavowal_do_not_become_asserted_claims(isolated):
    import text_tool
    question = text_tool.build_evidence("这款精华能治愈痘痘吗？不能。")
    result = question["evidence"][0]
    assert result["claim_high_count"] == 0
    assert any(hit["word"] == "治愈" and hit["severity"] == "review" for hit in result["claim_violations"])
    denied = text_tool.build_evidence("所谓这款精华七天治愈痘痘的说法不属实。")
    assert denied["evidence"][0]["claim_high_count"] == 0
    asserted = text_tool.build_evidence("这款精华七天治愈痘痘。")
    assert asserted["evidence"][0]["claim_high_count"] > 0
    hits = text_tool.scan_claims("根治痘痘。")
    assert sum(hit["word"] == "根治" and hit["category"] == "medical" for hit in hits) == 1
    clinical = text_tool.build_evidence("本品保湿功效经过临床验证。")
    assert clinical["evidence"][0]["claim_high_count"] == 0
    assert "不等同于医疗功效" in clinical["observed"]
    assert any(hit["word"] == "临床验证" and hit["severity"] == "review" for hit in clinical["evidence"][0]["claim_violations"])


def test_numeric_claims_must_match_subject_and_conditions():
    assert crossmodal_tool.extract_dim_numbers("水分提升20.5%，94%为天然来源") == {
        "百分比": [20.5, 94.0],
    }
    ocr = image_evidence("【青蓝精华】7天水分提升22%。94%天然来源。正装30ml。")
    findings, _ = crossmodal_tool.check("【青蓝精华】7天水分提升20%。赠品10ml。", ocr)
    assert len(findings) == 1
    assert "20%" in findings[0]["text_side"] and "22%" in findings[0]["image_side"]
    for text in ("【青蓝精华】28天水分提升20%。", "【橙白精华】7天水分提升20%。",
                 "【青蓝精华】7天敏感肌水分提升20%。"):
        assert not crossmodal_tool.check(text, ocr)[0]


def test_context_conflict_uses_same_product_metric_and_conditions(isolated):
    text = "【青蓝精华】14天水分提升20%。【青蓝精华】14天水分提升40%。"
    result = pipeline.analyze(text=text, verbose=False, publish=False)
    conflicts = result["audit"]["context_conflicts"]
    assert len(conflicts) == 1
    assert "20%" in conflicts[0]["first"]["quote"]
    assert "40%" in conflicts[0]["second"]["quote"]
    assert result["verdict"]["risk_level"] == "suspicious"
    for other in ("【青蓝精华】28天水分提升40%。", "【橙白精华】14天水分提升40%。",
                  "【青蓝精华】94%天然来源。", "【青蓝精华】14天水分不能提升40%。"):
        claims = claim_review.extract("【青蓝精华】14天水分提升20%。" + other)
        assert not claim_review.context_conflicts(claims)


def test_image_shade_matches_only_same_product():
    image = image_evidence("【青蓝遮瑕】色号 325")
    conflicts, _ = crossmodal_tool.check("【青蓝遮瑕】色号 330", image, product="青蓝遮瑕")
    assert len(conflicts) == 1
    assert "330" in conflicts[0]["text_side"]
    assert "325" in conflicts[0]["image_side"]
    conflicts, _ = crossmodal_tool.check("【青蓝遮瑕】色号 325", image, product="青蓝遮瑕")
    assert not conflicts
    assert not crossmodal_tool.check("色号 330", image)[0]


def test_other_product_or_missing_name_does_not_inherit_selected_product():
    body = "青蓝精华14天水分提升20%。"
    for other in ("橙白精华14天水分提升40%", "14天水分提升40%", "青蓝精华霜14天水分提升40%"):
        conflicts, _ = crossmodal_tool.check(body, image_evidence(other), product="青蓝精华")
        assert not conflicts
    conflicts, _ = crossmodal_tool.check(body, image_evidence("青蓝精华14天水分提升40%"), product="青蓝精华")
    assert len(conflicts) == 1
    claims = claim_review.extract(body + "橙白精华14天水分提升40%。", product="青蓝精华")
    assert not claim_review.context_conflicts(claims)


def test_product_versions_do_not_get_compared_as_same_sku():
    body = "【青蓝精华】2024版14天水分提升20%。"
    different = image_evidence("【青蓝精华】2025版14天水分提升40%")
    assert not crossmodal_tool.check(body, different, product="青蓝精华")[0]
    same = image_evidence("【青蓝精华】2024版14天水分提升40%")
    assert len(crossmodal_tool.check(body, same, product="青蓝精华")[0]) == 1


def test_low_confidence_ocr_keeps_region_for_review(isolated, monkeypatch):
    image = isolated / "sample.png"
    Image.new("RGB", (40, 40), "white").save(image)
    line = {"text": "色号 325", "bbox": [1, 2, 30, 12], "confidence": .4}
    monkeypatch.setattr(pipeline, "collect_evidence", lambda *args: {"ocr": {"evidence": [line]}})
    result = pipeline.analyze(image, text="色号 330", product="青蓝遮瑕", verbose=False,
                              publish=False, skip=[tool for tool, _, _ in pipeline.TOOL_LIST])
    assert result["audit"]["uncertain_ocr"] == [line]
    assert not result["audit"]["crossmodal"].get("contradictions")


def test_disclosed_illustration_and_gray_band_are_different_review_paths():
    evidence = image_evidence("AI示意图", score=0.999)
    evidence["ai_label"] = {"evidence": [{"ai_label_detected": True}]}
    for text, expected in [("以下为配色示意。", "inconclusive"),
                           ("真人使用前后对比，展示实拍功效。", "high_risk")]:
        conflicts, ignored = crossmodal_tool.check(text, evidence)
        evidence["crossmodal"] = crossmodal_tool.build_evidence(text, conflicts, evidence, "sample", ignored)
        assert rule_engine.judge(evidence)[0] == expected
    gray = image_evidence()
    text = "这张图是原相机实拍。"
    conflicts, ignored = crossmodal_tool.check(text, gray)
    assert [c["severity"] for c in conflicts] == ["medium"]
    gray["crossmodal"] = crossmodal_tool.build_evidence(text, conflicts, gray, "sample", ignored)
    assert rule_engine.judge(gray)[0] == "suspicious"
    fallback = image_evidence(model="capcheck", score=0.999)
    assert rule_engine.judge(fallback)[0] == "inconclusive"
    assert rule_engine.image_signals(fallback)["origin"]["decision_score"] is None


def test_claim_model_is_source_anchored_and_degrades(isolated, monkeypatch):
    text = "青蓝精华有助改善肤色不均。"
    monkeypatch.setattr(llm_explainer, "model_ready", lambda: True)
    monkeypatch.setattr(llm_explainer, "generate_for_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_explainer, "_chat", lambda prompt: json.dumps([
        {"quote": text, "product": "青蓝精华", "metric_text": "改善肤色不均", "stance": "asserted"},
        {"quote": "伪造的百分比99%", "metric_text": "伪造的百分比", "stance": "asserted"},
    ]))
    audit = pipeline.analyze(text=text, llm=True, verbose=False)["audit"]
    assert audit["checks"]["claim_model"]["status"] == "completed"
    assert [(c["quote"], c["status"]) for c in audit["claims"]] == [(text, "证据不足")]
    monkeypatch.setattr(llm_explainer, "model_ready", lambda: False)
    audit = pipeline.analyze(text="七天治愈痘痘。", llm=True, verbose=False)["audit"]
    assert audit["checks"]["claim_model"]["status"] == "model_unavailable"
    assert audit["claims"][0]["metric"] == "治愈"


def test_model_claim_is_not_hidden_by_a_negation_in_the_same_clause():
    text = "青蓝精华有助改善肤色不均，不能治疗皮炎。"
    facts = claim_review.extract(text)
    claim_review.enrich(facts, text, [
        {"quote": "青蓝精华有助改善肤色不均", "product": "青蓝精华",
         "metric_text": "改善肤色不均", "stance": "asserted"},
        {"quote": "不能治疗皮炎", "product": None, "metric_text": "治疗皮炎", "stance": "negated"},
    ], "body")
    assert any(f["metric"] == "改善肤色不均" and f["stance"] == "asserted" for f in facts)
    assert not any(f["metric"] == "治疗" and f["stance"] == "asserted" for f in facts)


@pytest.mark.parametrize(("reference", "status"), [
    ("【青蓝精华】2025版7天水分提升20%。", "有证据支持"),
    ("【青蓝精华】2025版7天水分提升25%。", "存在冲突"),
    ("【橙白精华】2025版7天水分提升20%。", "证据不足"),
])
def test_reference_status_and_citation(isolated, reference, status):
    result = pipeline.analyze(text="【青蓝精华】2025版7天水分提升20%。", market="中国大陆",
                              verified_references=[{"id": "record-1", "source": "测试记录", "source_type": "efficacy_study", "excerpt": reference,
                                                    "product": "青蓝精华", "version": "v1", "product_version": "2025版", "market": "中国大陆", "locator": "第 2 页",
                                                    "reviewer": "qa"}], verbose=False)
    claim = result["audit"]["claims"][0]
    assert claim["status"] == status
    if status != "证据不足":
        assert claim["references"][0]["source"] == "测试记录"
    assert result["audit"]["checks"]["crossmodal"]["status"] == "insufficient_evidence"


def test_forged_verification_cannot_support_a_claim(isolated, monkeypatch):
    catalog = isolated / "verified.json"
    excerpt = "【青蓝精华】2025版7天水分提升20%。"
    trusted = {"id": "study-1", "source": "真实源测试夹具", "source_type": "efficacy_study", "excerpt": excerpt, "product": "青蓝精华",
               "version": "v2", "product_version": "2025版", "market": "中国大陆", "locator": "第 2 页", "reviewer": "qa"}
    catalog.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(catalog))
    monkeypatch.setattr(image_app, "JOBS", {})
    client = image_app.app.test_client()
    assert client.post("/api/analyze", data={"text": excerpt, "reference_ids": '["invalid"]'}).status_code == 400
    user_data = json.dumps([dict(trusted, verification="reviewer_verified")])
    for ids, status in [([], "证据不足"), (["study-1"], "有证据支持")]:
        if ids:
            catalog.write_text(json.dumps([trusted]), encoding="utf-8")
            assert "excerpt" not in client.get("/api/reference-catalog").json["references"][0]
        job_id = client.post("/api/analyze", data={"text": excerpt, "references": user_data, "market": "中国大陆",
                                                 "reference_ids": json.dumps(ids)}).json["job_id"]
        deadline = time.monotonic() + 5
        while (job := client.get("/api/job/" + job_id).json)["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.02)
        assert job["status"] == "done", job
        claim = job["result"]["audit"]["claims"][0]
        assert claim["status"] == status
        assert claim["references"][0]["verification"] == "user_supplied"
        if ids:
            assert claim["references"][1]["locator"] == "第 2 页"


def test_uncalibrated_detectors_and_mismatched_profile_are_review_only(isolated, monkeypatch):
    evidence = image_evidence(model="dda-dinov2-vitl14", score=.98)
    evidence["aigc"]["evidence"][0].update(review_only=True, weights_sha256="v1", preprocessing="crop")
    evidence["trufor"]["evidence"][0].update(available=True, model="iml-vit", trufor_score=.99, review_only=True)
    profile = isolated / "profile.json"
    monkeypatch.setenv("BP_ORIGIN_PROFILE", str(profile))
    row = {"model": "dda-dinov2-vitl14", "preprocessing": "crop", "low": .3, "high": .8, "approved": True}
    profile.write_text(json.dumps(dict(row, weights_sha256="wrong")), encoding="utf-8")
    assert rule_engine.image_signals(evidence)["origin"]["status"] == "未校准，供复核"
    assert rule_engine.judge(evidence)[0] == "inconclusive"
    profile.write_text(json.dumps(dict(row, weights_sha256="v1")), encoding="utf-8")
    assert rule_engine.image_signals(evidence)["origin"]["status"] == "倾向生成"
    assert rule_engine.judge(evidence)[0] == "inconclusive"


def test_concurrent_same_named_images_have_distinct_evidence(isolated, monkeypatch):
    files = []
    for name, color in (("first", "red"), ("second", "blue")):
        f = isolated / name / "shared.png"
        f.parent.mkdir()
        Image.new("RGB", (6, 6), color).save(f)
        files.append(f)
    stale = isolated / "outputs" / "aigc_shared.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"tool": "aigc", "evidence": [{"aigc_score": 1}]}))

    def detector(script, path, timeout=900, output_dir=None):
        assert script == "ocr_tool.py"
        pixel = Image.open(path).getpixel((0, 0))
        text = "红色" if pixel[0] else "蓝色"
        (output_dir / "ocr_shared.json").write_text(
            json.dumps({"tool": "ocr", "evidence": [{"text": text}]}))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(pipeline, "run_tool", detector)
    skip = [tool for tool, _, _ in pipeline.TOOL_LIST if tool != "ocr"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda f: pipeline.analyze(f, skip=skip, verbose=False), files))
    assert len({r["run_id"] for r in results}) == 2
    for source, result, text in zip(files, results, ("红色", "蓝色")):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert result["image"]["sha256"] == digest
        assert result["evidence"]["ocr"]["image_sha256"] == digest
        assert result["evidence"]["ocr"]["evidence"][0]["text"] == text
        assert "aigc" not in result["evidence"]
        assert Path(result["_output_file"]).parent.name == result["run_id"]


def test_web_text_and_image_entry_points(isolated, monkeypatch):
    uploads = isolated / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(image_app, "UPLOADS", uploads)
    monkeypatch.setattr(image_app, "JOBS", {})
    from web.content_app import app as content
    assert content.test_client().post("/check", json={"text": "不能治疗皮炎。"}).json["verdict"] == "inconclusive"
    real_analyze = pipeline.analyze
    monkeypatch.setattr(pipeline, "analyze", lambda *args, **kwargs:
                        real_analyze(*args, **kwargs, skip=tuple(k for k, _, _ in pipeline.TOOL_LIST if k != "hash")))
    with image_app.app.test_client() as client:
        data = {"text": "七天治愈痘痘。"}
        img = io.BytesIO()
        Image.new("RGB", (4, 4)).save(img, format="PNG")
        digest = hashlib.sha256(img.getvalue()).hexdigest()
        img.seek(0)
        for payload in (data, dict(data, file=(img, "sample.png"))):
            submitted = client.post("/api/analyze", data=payload)
            assert submitted.status_code == 200
            endpoint = "/api/job/" + submitted.json["job_id"]
            deadline = time.monotonic() + 5
            while (job := client.get(endpoint).json)["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
            assert job["status"] == "done", job.get("error")
            assert job["result"]["verdict"]["risk_level"] == "suspicious"
            if "file" in payload:
                assert job["result"]["evidence"]["hash"]["image_sha256"] == digest


def test_web_packaging_reference(isolated, monkeypatch):
    uploads = isolated / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(image_app, "UPLOADS", uploads)
    monkeypatch.setattr(image_app, "JOBS", {})
    real_analyze = pipeline.analyze
    monkeypatch.setattr(pipeline, "analyze", lambda *args, **kwargs:
                        real_analyze(*args, **kwargs, skip=tuple(k for k, _, _ in pipeline.TOOL_LIST)))

    def image_bytes():
        stream = io.BytesIO()
        Image.new("RGB", (50, 50), "white").save(stream, format="PNG")
        stream.seek(0)
        return stream

    with image_app.app.test_client() as client:
        assert client.post("/api/analyze", data={"text": "正文", "reference_image": (image_bytes(), "ref.png")}).status_code == 400
        response = client.post("/api/analyze", data={"text": "正文", "file": (image_bytes(), "ad.png"),
                                                       "reference_image": (image_bytes(), "ref.png")})
        assert response.status_code == 200
        endpoint = "/api/job/" + response.json["job_id"]
        deadline = time.monotonic() + 5
        while (job := client.get(endpoint).json)["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert job["status"] == "done", job.get("error")
        assert job["result"]["audit"]["crossmodal"]["package_reference"]["status"] == "无法判断"

"""Exercise remote text extraction and local evidence through real entry points."""
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import llm_explainer
import pipeline
import import_brand_claims
from web.app import app


def test_brand_import_is_repeatable_and_cannot_create_study_records(tmp_path, monkeypatch):
    from claim_review import verified_catalog
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "verified.json"))
    import_brand_claims.main()
    first = verified_catalog()
    assert first and all(row["source_type"] == "brand_claim" for row in first.values())
    import_brand_claims.main()
    assert verified_catalog() == first


def test_deepseek_receives_text_only_and_rejects_fabricated_quotes(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("BP_DEEPSEEK_BASE_URL", "https://example.invalid")
    text = "青蓝精华有助改善肤色不均。"

    def respond(request, timeout):
        assert request.full_url == "https://example.invalid/chat/completions"
        payload = json.loads(request.data)
        assert payload["model"] == "deepseek-flash"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][1]["content"].endswith(json.dumps({"text": text, "product": None}, ensure_ascii=False))
        assert "image" not in payload["messages"][1]
        content = {"claims": [
            {"quote": text[:-1], "product": "青蓝精华", "metric_text": "改善肤色不均", "stance": "asserted"},
            {"quote": "原文没有的99%", "product": None, "metric_text": "99%", "stance": "asserted"},
        ]}
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode())

    monkeypatch.setattr(llm_explainer.urllib.request, "urlopen", respond)
    rows = llm_explainer.extract_claim_fields(text)
    assert len(rows) == 1
    assert rows[0]["quote"] == text[:-1]


def test_deepseek_does_not_assign_an_ambiguous_quote_to_first_occurrence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    content = {"claims": [{"quote": "保湿20%", "metric_text": "保湿", "stance": "asserted"}]}
    monkeypatch.setattr(llm_explainer.urllib.request, "urlopen", lambda *args, **kwargs:
                        io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()))
    assert llm_explainer.extract_claim_fields("保湿20%。再次写保湿20%。") is None


def test_semantic_review_requires_real_unambiguous_quotes(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    text = "精华立即见效。详细说明需要连续使用28天。"
    reference = {"id": "ref-1", "source": "报告", "excerpt": "连续使用28天。",
                 "population": "42名受试者", "period": "连续使用28天"}
    findings = [
        {"quote": "精华立即见效", "other_quote": "连续使用28天", "reference_id": "ref-1", "reason": "时间不同"},
        {"quote": "伪造原句", "other_quote": "连续使用28天", "reference_id": "ref-1", "reason": "伪造"},
        {"quote": "精华立即见效", "other_quote": "连续使用28天", "reference_id": "unknown", "reason": "未知资料"},
    ]
    def respond(request, timeout):
        payload = json.loads(request.data)
        assert "image" not in payload["messages"][0]
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps({"findings": findings})}}]}).encode())
    monkeypatch.setattr(llm_explainer.urllib.request, "urlopen", respond)
    result = llm_explainer.semantic_review(text, [reference])
    assert len(result) == 1
    assert result[0]["reference_id"] == "ref-1"
    assert result[0]["reference_conditions"] == {"population": "42名受试者", "period": "连续使用28天"}


def test_semantic_candidates_are_scoped_to_product_and_market(tmp_path, monkeypatch):
    catalog = tmp_path / "verified.json"
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(catalog))
    monkeypatch.setenv("BP_TEXT_API_ONLY", "1")
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    rows = [{"id": "cn", "source": "品牌", "source_type": "brand_claim", "product": "青蓝精华",
             "market": "中国大陆", "version": "v1", "locator": "正文", "reviewer": "qa",
             "excerpt": "连续使用28天", "population": "成年人"},
            {"id": "us", "source": "品牌", "source_type": "brand_claim", "product": "青蓝精华",
             "market": "美国", "version": "v1", "locator": "正文", "reviewer": "qa",
             "excerpt": "立即见效"}]
    catalog.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(llm_explainer, "extract_claim_fields", lambda *args: [])
    def inspect(text, references):
        assert [ref["id"] for ref in references] == ["cn"]
        assert references[0]["population"] == "成年人"
        return []
    monkeypatch.setattr(llm_explainer, "semantic_review", inspect)
    result = pipeline.analyze(text="青蓝精华立即见效。", product="青蓝精华", market="中国大陆",
                              llm=True, verbose=False, publish=False)
    assert result["audit"]["checks"]["semantic_review"]["status"] == "completed"
    assert result["audit"]["checks"]["semantic_review"]["reference_count"] == 1


def test_semantic_review_marks_missing_references(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "missing.json"))
    monkeypatch.setenv("BP_TEXT_API_ONLY", "1")
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    monkeypatch.setattr(llm_explainer, "extract_claim_fields", lambda *args: [])
    monkeypatch.setattr(llm_explainer, "semantic_review", lambda text, references: [])
    result = pipeline.analyze(text="【青蓝精华】7天保湿。", llm=True, verbose=False, publish=False)
    status = result["audit"]["checks"]["semantic_review"]
    assert status["status"] == "completed" and status["reference_count"] == 0
    assert "仅检查正文上下文" in status["reason"]


def test_semantic_review_does_not_use_market_unspecified_reference(tmp_path, monkeypatch):
    catalog = tmp_path / "verified.json"
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(catalog))
    monkeypatch.setenv("BP_TEXT_API_ONLY", "1")
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    catalog.write_text(json.dumps([{"id": "ref", "source": "品牌", "product": "青蓝精华",
                                    "version": "v1", "locator": "正文", "reviewer": "qa", "excerpt": "14天"}]), encoding="utf-8")
    monkeypatch.setattr(llm_explainer, "extract_claim_fields", lambda *args: [])
    monkeypatch.setattr(llm_explainer, "semantic_review", lambda text, refs: [] if not refs else None)
    audit = pipeline.analyze(text="青蓝精华14天保湿。", product="青蓝精华", market="中国大陆", llm=True,
                             verbose=False, publish=False)["audit"]
    assert audit["checks"]["semantic_review"]["reference_count"] == 0


def test_semantic_review_excludes_explicitly_different_product_version(tmp_path, monkeypatch):
    catalog = tmp_path / "verified.json"
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(catalog))
    monkeypatch.setenv("BP_TEXT_API_ONLY", "1")
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    rows = [{"id": str(year), "source": "品牌", "source_type": "brand_claim", "product": "青蓝精华",
             "market": "中国大陆", "product_version": f"{year}版", "version": "v1", "locator": "正文",
             "reviewer": "qa", "excerpt": "14天"} for year in (2024, 2025)]
    catalog.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(llm_explainer, "extract_claim_fields", lambda *args: [])
    def inspect(text, refs):
        assert [ref["id"] for ref in refs] == ["2025"]
        return []
    monkeypatch.setattr(llm_explainer, "semantic_review", inspect)
    audit = pipeline.analyze(text="青蓝精华2025版14天保湿。", product="青蓝精华", market="中国大陆",
                             llm=True, verbose=False, publish=False)["audit"]
    assert audit["checks"]["semantic_review"]["reference_count"] == 1


def test_semantic_review_prioritizes_selected_reference_and_reports_truncation(tmp_path, monkeypatch):
    catalog = tmp_path / "verified.json"
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(catalog))
    monkeypatch.setenv("BP_TEXT_API_ONLY", "1")
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    rows = [{"id": str(i), "source": "品牌", "source_type": "brand_claim", "product": "青蓝精华",
             "market": "中国大陆", "version": "v1", "locator": "正文", "reviewer": "qa", "excerpt": f"资料{i}"}
            for i in range(9)]
    catalog.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(llm_explainer, "extract_claim_fields", lambda *args: [])
    def inspect(text, refs):
        assert len(refs) == 8 and refs[0]["id"] == "8"
        return []
    monkeypatch.setattr(llm_explainer, "semantic_review", inspect)
    audit = pipeline.analyze(text="青蓝精华保湿。", product="青蓝精华", market="中国大陆", llm=True,
                             verified_references=[rows[8]], verbose=False, publish=False)["audit"]
    status = audit["checks"]["semantic_review"]
    assert (status["reference_count"], status["reference_total"]) == (8, 9)
    assert "手选资料优先" in status["reason"]


def test_reviewed_knowledge_is_saved_and_matched_automatically(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "knowledge" / "verified.json"))
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    ref = {"source": "试验报告", "source_type": "efficacy_study", "product": "青蓝精华", "version": "v1", "locator": "第 2 页",
           "market": "中国大陆", "product_version": "2025版", "reviewer": "审核员", "excerpt": "【青蓝精华】2025版7天水分提升20%。"}
    client = app.test_client()
    assert client.post("/api/reference-catalog", json=ref).status_code == 400
    response = client.post("/api/reference-catalog", json={**ref, "reviewer_confirmed": True})
    assert response.status_code == 201
    assert len(client.get("/api/reference-catalog").json["references"]) == 1
    result = pipeline.analyze(text=ref["excerpt"], market="中国大陆", verbose=False, publish=False)
    claim = result["audit"]["claims"][0]
    assert claim["status"] == "有证据支持"
    assert claim["references"][0]["locator"] == "第 2 页"
    result = pipeline.analyze(text="【橙白精华】7天水分提升20%。", verbose=False, publish=False)
    assert result["audit"]["claims"][0]["status"] == "证据不足"


def test_legacy_and_untyped_references_cannot_support_claims(tmp_path, monkeypatch):
    catalog = tmp_path / "verified.json"
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(catalog))
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    ref = {"source": "品牌页面", "product": "青蓝精华", "version": "v1", "locator": "页面",
           "reviewer": "审核员", "excerpt": "【青蓝精华】7天水分提升20%。"}
    client = app.test_client()
    assert client.post("/api/reference-catalog", json={**ref, "reviewer_confirmed": True}).status_code == 400
    catalog.write_text(json.dumps([{**ref, "id": "legacy"}]), encoding="utf-8")
    claim = pipeline.analyze(text=ref["excerpt"], verbose=False, publish=False)["audit"]["claims"][0]
    assert claim["status"] == "证据不足"
    assert claim["references"][0]["id"] == "legacy"


def test_brand_claim_cannot_become_efficacy_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "verified.json"))
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    ref = {"source": "品牌页面", "source_type": "brand_claim", "market": "美国",
           "product": "青蓝精华", "version": "v1", "locator": "页面正文", "reviewer": "审核员",
           "excerpt": "【青蓝精华】7天水分提升20%。", "reviewer_confirmed": True}
    client = app.test_client()
    assert client.post("/api/reference-catalog", json=ref).status_code == 201
    result = pipeline.analyze(text=ref["excerpt"], verbose=False, publish=False)
    claim = result["audit"]["claims"][0]
    assert claim["status"] == "证据不足"
    assert claim["references"][0]["source_type"] == "brand_claim"


def test_efficacy_evidence_requires_matching_declared_market(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "verified.json"))
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    ref = {"source": "报告", "source_type": "efficacy_study", "market": "美国", "product_version": "2025版",
           "product": "青蓝精华", "version": "v1", "locator": "第2页", "reviewer": "审核员",
           "excerpt": "【青蓝精华】2025版7天水分提升20%。", "reviewer_confirmed": True}
    assert app.test_client().post("/api/reference-catalog", json=ref).status_code == 201
    for market, expected in [(None, "证据不足"), ("中国", "证据不足"), ("美国", "有证据支持")]:
        result = pipeline.analyze(text=ref["excerpt"], market=market, verbose=False, publish=False)
        assert result["audit"]["claims"][0]["status"] == expected


def test_efficacy_support_requires_known_matching_product_version_and_market(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "verified.json"))
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    from claim_review import add_verified_reference
    ref = {"source": "报告", "source_type": "efficacy_study", "market": "中国大陆",
           "product": "青蓝精华", "version": "报告第1版", "product_version": "2025版",
           "locator": "第2页", "reviewer": "审核员", "excerpt": "【青蓝精华】7天水分提升20%。",
           "reviewer_confirmed": True}
    add_verified_reference(ref)
    for text, expected in [("【青蓝精华】2024版7天水分提升20%。", "证据不足"),
                           ("【青蓝精华】7天水分提升20%。", "证据不足"),
                           ("【青蓝精华】2025版7天水分提升20%。", "有证据支持")]:
        result = pipeline.analyze(text=text, market="中国大陆", verbose=False, publish=False)
        assert result["audit"]["claims"][0]["status"] == expected
    ref.pop("market")
    Path(tmp_path / "verified.json").unlink()
    add_verified_reference(ref)
    result = pipeline.analyze(text="【青蓝精华】2025版7天水分提升20%。", market="中国大陆", verbose=False, publish=False)
    assert result["audit"]["claims"][0]["status"] == "证据不足"


def test_study_conditions_cannot_be_silently_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_VERIFIED_REFERENCES", str(tmp_path / "verified.json"))
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    ref = {"source": "报告", "source_type": "efficacy_study", "conditions": "实验室",
           "product": "青蓝精华", "version": "v1", "locator": "第2页", "reviewer": "审核员",
           "excerpt": "【青蓝精华】7天水分提升20%。", "reviewer_confirmed": True}
    assert app.test_client().post("/api/reference-catalog", json=ref).status_code == 201
    result = pipeline.analyze(text=ref["excerpt"], verbose=False, publish=False)
    assert result["audit"]["claims"][0]["status"] == "证据不足"

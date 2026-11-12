"""Regression cases for the blind-set boundary and release metrics."""
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from beauty_benchmark import calibrate, evaluate, origin_report, release_checks, run, sha256, validate


def test_derivative_and_generator_cannot_leak_into_blind(tmp_path):
    image = tmp_path / "original.png"
    Image.new("RGB", (24, 24), "red").save(image)
    manifest = tmp_path / "manifest.json"
    row = {"id": "a", "group": "a", "authorization": "consent-1", "image": image.name,
           "sha256": sha256(image), "kind": "generated", "split": "train",
           "generator": "family-a", "generator_version": "v1"}
    blind = dict(row, id="b", group="b", split="blind", reviewers=["x", "y"])
    manifest.write_text(json.dumps([row, blind]), encoding="utf-8")
    with pytest.raises(ValueError, match="generator"):
        validate(manifest)
    blind.update(kind="edited", edit="retouch", original_sha256=row["sha256"])
    blind.pop("generator")
    manifest.write_text(json.dumps([row, blind]), encoding="utf-8")
    with pytest.raises(ValueError, match="original_sha256"):
        validate(manifest)


def test_same_image_or_identity_cannot_inflate_groups(tmp_path):
    image = tmp_path / "original.png"
    Image.new("RGB", (24, 24), "red").save(image)
    manifest = tmp_path / "manifest.json"
    row = {"id": "a", "group": "a", "authorization": "consent-1", "image": image.name,
           "sha256": sha256(image), "kind": "real", "split": "train"}
    duplicate = dict(row, id="b", group="b")
    manifest.write_text(json.dumps([row, duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="图片跨分区或来源组重复"):
        validate(manifest)

    other = tmp_path / "other.png"
    Image.new("RGB", (24, 24), "blue").save(other)
    row["identity"] = duplicate["identity"] = "person-1"
    duplicate.update(image=other.name, sha256=sha256(other))
    manifest.write_text(json.dumps([row, duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        validate(manifest)


def test_edit_requires_original_record_in_same_group(tmp_path):
    original = tmp_path / "original.png"
    edited = tmp_path / "edited.png"
    Image.new("RGB", (24, 24), "red").save(original)
    Image.new("RGB", (24, 24), "blue").save(edited)
    manifest = tmp_path / "manifest.json"
    source = {"id": "original", "group": "a", "authorization": "consent-1", "image": original.name,
              "sha256": sha256(original), "kind": "real", "split": "train"}
    derivative = dict(source, id="edited", image=edited.name, sha256=sha256(edited),
                      kind="edited", original_sha256=source["sha256"], edit="retouch")
    manifest.write_text(json.dumps([derivative]), encoding="utf-8")
    with pytest.raises(ValueError, match="original_sha256 找不到"):
        validate(manifest)
    manifest.write_text(json.dumps([derivative, dict(source, group="b")]), encoding="utf-8")
    with pytest.raises(ValueError, match="original_sha256"):
        validate(manifest)
    manifest.write_text(json.dumps([derivative, source]), encoding="utf-8")
    assert len(validate(manifest)) == 2


def test_group_metrics_count_variants_once():
    rows = [({"group": "a", "kind": "real"}, score) for score in [.1, .95]]
    rows += [({"group": "b", "kind": "generated"}, score) for score in [.9, .7]]
    metrics = origin_report(rows, .4, .8)
    assert (metrics["false_positive"]["count"], metrics["false_positive"]["n"]) == (1, 1)
    assert (metrics["recall"]["count"], metrics["recall"]["n"]) == (0, 1)
    assert metrics["uncertain"]["count"] == 1


def test_missing_origin_scores_are_excluded_and_block_release(tmp_path):
    profile = {"low": .4, "high": .8, "model": "dda", "weights_sha256": "sha", "preprocessing": "v1"}
    rows = []
    for i in range(120):
        score = None if i < 20 else .95 if i < 25 else .1
        signal = {"score": score, "model": "dda", "weights_sha256": "sha", "preprocessing": "v1"}
        rows.append(({"kind": "real", "group": f"real-{i}"},
                     {"signals": {"origin": signal, "editing": {}}, "checks": {}, "seconds": 0}))
    for i in range(100):
        signal = {"score": .9, "model": "dda", "weights_sha256": "sha", "preprocessing": "v1"}
        rows.append(({"kind": "generated", "group": f"fake-{i}", "generator": f"family-{i // 50}"},
                     {"signals": {"origin": signal, "editing": {}}, "checks": {}, "seconds": 0}))
    metrics = evaluate(rows, profile, tmp_path)
    assert metrics["origin"]["false_positive"]["count"] == 5
    assert metrics["origin"]["false_positive"]["n"] == 100
    assert metrics["origin"]["missing_scores"] == {"real": 20, "generated": 0}
    assert release_checks(metrics)["origin"] is False


def test_blind_run_rejects_visible_or_unreviewed_ai_labels(tmp_path):
    row = {"id": "example", "kind": "generated", "split": "blind",
           "visible_ai_label_reviewed": True, "visible_ai_label_present": True}
    with pytest.raises(ValueError, match="可见 AI 标识"):
        run(tmp_path / "manifest.json", [row], False)
    row.pop("visible_ai_label_present")
    with pytest.raises(ValueError, match="可见 AI 标识"):
        run(tmp_path / "manifest.json", [row], False)


def test_missing_edit_score_is_not_counted_as_clean_and_blocks_release(tmp_path):
    row = {"id": "a", "group": "a", "kind": "real"}
    report = {"signals": {"origin": {}, "editing": {"score": None}}, "checks": {}, "seconds": 0}
    metrics = evaluate([(row, report)], None, tmp_path)
    assert metrics["localization"]["clean_false_positive"]["n"] == 0
    assert metrics["localization"]["missing_scores"] == 1
    metrics["localization"].update(mask_groups=60, mean_iou=.5,
                                   clean_false_positive={"n": 100, "rate": 0}, models=[{}])
    assert release_checks(metrics)["editing"] is False


def test_detected_edit_without_mask_blocks_release(tmp_path):
    row = {"id": "a", "group": "a", "kind": "edited", "mask": "truth.png"}
    report = {"signals": {"editing": {"score": .9, "model": "iml", "weights_sha256": "hash",
                                     "preprocessing": "v1"}}, "checks": {}, "seconds": 0}
    metrics = evaluate([(row, report)], None, tmp_path)
    assert metrics["localization"]["mean_iou"] == 0
    assert metrics["localization"]["missing_masks"] == 1
    metrics["localization"].update(mask_groups=60, mean_iou=.5,
                                   clean_false_positive={"n": 100, "rate": 0})
    assert release_checks(metrics)["editing"] is False


def test_context_metrics_count_real_labels_and_missing_model(tmp_path):
    rows = []
    for index, (label, predicted) in enumerate([(True, True), (False, True), (True, False)]):
        row = {"id": str(index), "group": str(index), "kind": "claim", "claims": [],
               "crossmodal_conflict": False, "context_conflict": label}
        report = {"signals": {"editing": {}}, "claims": [], "crossmodal": {}, "seconds": 0,
                  "context_conflicts": [{}] if predicted else [], "semantic_findings": [],
                  "checks": {"semantic_review": {"status": "model_unavailable" if index == 2 else "completed"}}}
        rows.append((row, report))
    metrics = evaluate(rows, None, tmp_path)["context"]
    assert metrics["precision"]["rate"] == .5
    assert metrics["recall"]["rate"] == .5
    assert metrics["semantic_unavailable"] == 1
    del rows[0][1]["semantic_findings"]
    with pytest.raises(ValueError, match="缺少正文上下文结果"):
        evaluate(rows, None, tmp_path)


def test_context_release_requires_enough_cases_quality_and_available_model(tmp_path):
    metrics = evaluate([], None, tmp_path)
    context = metrics["context"]
    context.update(cases=60, precision={"rate": .9}, recall={"rate": .8}, semantic_unavailable=0)
    assert release_checks(metrics)["context"] is True
    context["semantic_unavailable"] = 1
    assert release_checks(metrics)["context"] is False
    context.update(semantic_unavailable=0, cases=59)
    assert release_checks(metrics)["context"] is False


def test_brand_claim_cannot_pass_support_audit(tmp_path):
    row = {"id": "a", "group": "a", "kind": "claim", "market": "中国大陆", "claims": [], "crossmodal_conflict": False}
    ref = {"verification": "reviewer_verified", "source": "brand", "locator": "page",
           "source_type": "brand_claim", "matches": True, "market": "中国大陆", "product_version": "2025版"}
    claim = {"status": "有证据支持", "quote": "test", "product_version": "2025版", "references": [ref]}
    report = {"signals": {"editing": {}}, "claims": [claim], "crossmodal": {}, "seconds": 0, "checks": {}}
    assert evaluate([(row, report)], None, tmp_path)["claims"]["unsupported"] == 1
    ref["source_type"] = "efficacy_study"
    assert evaluate([(row, report)], None, tmp_path)["claims"]["unsupported"] == 0
    ref["product_version"] = "2024版"
    assert evaluate([(row, report)], None, tmp_path)["claims"]["unsupported"] == 1


def test_mixed_or_untracked_edit_models_cannot_pass_release(tmp_path):
    rows = []
    for idx, weights in enumerate(("hash-a", "hash-b", None)):
        row = {"id": str(idx), "group": str(idx), "kind": "real"}
        signal = {"score": .1, "model": "iml", "weights_sha256": weights, "preprocessing": "v1"}
        rows.append((row, {"signals": {"origin": {}, "editing": signal}, "checks": {}, "seconds": 0}))
    metrics = evaluate(rows, None, tmp_path)
    assert len(metrics["localization"]["models"]) == 2
    assert metrics["localization"]["untracked_scores"] == 1
    metrics["localization"].update(mask_groups=60, mean_iou=.5,
                                   clean_false_positive={"n": 100, "rate": 0})
    assert release_checks(metrics)["editing"] is False


def test_calibration_never_approves_a_profile():
    rows = []
    for kind, values in [("real", [.1, .2]), ("generated", [.8, .9])]:
        for idx, score in enumerate(values):
            signal = {"model": "candidate", "weights_sha256": "sha", "preprocessing": "crop", "score": score}
            rows.append(({"kind": kind, "group": f"{kind}-{idx}"}, {"signals": {"origin": signal}}))
    profile = calibrate(rows, "manifest")
    assert profile["high"] == .8
    assert profile["approved"] is False
    assert profile["metrics"]["recall"]["rate"] == 1

"""Bounded evidence decisions shared by live review and offline policy replay."""

import time


POLICIES = ("v0", "v1")
ACTIONS = {
    "request_metadata": "补充产品、版本或市场",
    "request_original": "补充清晰原图",
    "request_study": "补充已核验功效试验",
    "human_review": "转人工复核",
    "finish": "结束取证",
}


def observe(report, market, *, followup_used=False):
    """Export decision inputs without excerpts, filenames, or product identities."""
    audit = report["audit"]
    claims = []
    for index, claim in enumerate(audit["claims"]):
        claims.append({
            "id": f"claim:{claim['source']}:{index}",
            "source": claim["source"],
            "quantified": isinstance(claim.get("value"), (int, float))
            and claim["metric"] in ("水分提升", "斑点减少", "保湿时长"),
            "product_known": bool(claim.get("product")),
            "version_known": bool(claim.get("product_version"))
            and claim["product_version"] not in ("新版", "旧版", "ambiguous"),
            "status": claim["status"],
        })
    return {
        "market_known": bool(market),
        "has_image": bool(report.get("image")),
        "uncertain_ocr_count": len(audit["uncertain_ocr"]),
        "ocr_status": audit["checks"].get("ocr", {}).get("status", "insufficient_evidence"),
        "context_conflict": bool(audit["context_conflicts"]),
        "followup_used": followup_used,
        "claims": claims,
    }


def decide(observation, policy_version="v0"):
    """Change only study/original priority between v0 and v1."""
    if policy_version not in POLICIES:
        raise ValueError("未知取证策略版本")
    claims = observation["claims"]
    unresolved = [c for c in claims if c["status"] == "证据不足"]
    quantified = [c for c in unresolved if c["quantified"]]
    eligible = [c for c in quantified if c["product_known"] and c["version_known"]
                and observation["market_known"]]
    unclear = observation["has_image"] and (
        observation["uncertain_ocr_count"] > 0 or observation["ocr_status"] != "completed"
    )
    if observation["context_conflict"] or any(c["status"] == "存在冲突" for c in claims):
        action, reason = "human_review", "conflicting_evidence"
    elif observation["followup_used"]:
        action, reason = ("human_review", "followup_still_incomplete") if unresolved or unclear else (
            "finish", "evidence_checked")
    elif quantified and (not observation["market_known"] or any(
            not c["product_known"] or not c["version_known"] for c in quantified)):
        action, reason = "request_metadata", "missing_scope"
    elif eligible and policy_version == "v1" and any(c["source"] == "body" for c in eligible):
        action, reason = "request_study", "body_study_first"
    elif unclear:
        action, reason = "request_original", "uncertain_image_text"
    elif eligible:
        action, reason = "request_study", "missing_efficacy_study"
    elif unresolved or not claims:
        action, reason = "human_review", "insufficient_observations"
    else:
        action, reason = "finish", "evidence_checked"
    evidence_ids = [c["id"] for c in claims]
    if observation["has_image"]:
        evidence_ids.append("tool:ocr")
    if observation["context_conflict"]:
        evidence_ids.append("context:conflicts")
    return {"action": action, "label": ACTIONS[action], "reason_code": reason,
            "evidence_ids": evidence_ids}


def trace(report, market, *, followup_used=False):
    started = time.monotonic()
    observation = observe(report, market, followup_used=followup_used)
    decision = decide(observation)
    state = "completed"
    if decision["action"] == "request_study":
        state = "waiting_evidence"
    elif decision["action"].startswith("request_"):
        state = "waiting_materials"
    elif decision["action"] == "human_review":
        state = "human_review"
    return {
        "schema": "beautyproof/harness@1", "policy_version": "v0",
        "observation": observation, "decision": decision,
        "state": state,
        "followup_used": followup_used,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
    }

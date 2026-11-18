"""Regression gates for the one mutable evidence-priority decision."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from replay_harness import load_cases, replay, validate_cases
from review_harness import decide


def dataset():
    return load_cases([ROOT / "data/harness_replay.json"])


def test_replay_is_deterministic_without_mutating_observations():
    cases = dataset()
    frozen = copy.deepcopy(cases)
    result = replay(cases)
    assert result == replay(list(reversed(cases)))
    assert cases == frozen
    assert result["recommendation"] == "suggest_human_adoption"
    assert result["development"]["v0_hits"] == 5
    assert result["development"]["v1_hits"] == 6
    assert result["holdout"]["v0_hits"] == 4
    assert result["holdout"]["v1_hits"] == 6
    assert result["execution"] == {"tools_executed": 0, "models_called": 0, "verdicts_modified": False}


def test_holdout_regression_rejects_candidate_without_changing_selection():
    cases = dataset()
    cases[6]["feedback"]["correct_action"] = "request_original"
    result = replay(cases)
    assert result["selected_on_development"] == "v1"
    assert result["recommendation"] == "keep_baseline"
    assert result["holdout"]["regressed"] == ["hold-01"]
    assert result["holdout"]["improved"] == ["hold-02"]


def test_candidate_must_be_selected_on_development():
    cases = dataset()
    cases[0]["feedback"]["correct_action"] = "request_original"
    result = replay(cases)
    assert result["selected_on_development"] == "v0"
    assert result["recommendation"] == "keep_baseline"


@pytest.mark.parametrize("change", [
    lambda cases: cases[6].update(group=cases[0]["group"]),
    lambda cases: cases[1].update(sample_id=cases[0]["sample_id"]),
    lambda cases: cases[0]["observation"].update(excerpt="private text"),
    lambda cases: cases[0]["observation"].update(has_image="false"),
    lambda cases: cases[0]["feedback"].update(correct_action="run_arbitrary_tool"),
    lambda cases: cases.pop(),
])
def test_invalid_external_corpus_is_rejected(change):
    cases = dataset()
    change(cases)
    with pytest.raises(ValueError):
        validate_cases(cases)


def test_ui_single_case_exports_can_be_combined(tmp_path):
    paths = []
    for index, case in enumerate(dataset()):
        path = tmp_path / f"feedback-{index}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)
    assert replay(load_cases(paths)) == replay(dataset())


def test_consumed_followup_cannot_request_more_evidence():
    observation = dataset()[0]["observation"]
    assert decide(observation, "v0")["action"] == "request_original"
    assert decide(observation, "v1")["action"] == "request_study"
    assert decide(observation, "v1", followup_used=True)["action"] == "human_review"

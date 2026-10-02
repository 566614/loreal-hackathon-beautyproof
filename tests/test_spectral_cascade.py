# -*- coding: utf-8 -*-
"""
频域工具 + 双检测器级联的回归测试

为什么单独写这个文件：
    2026-10-02 判罚逻辑从「AIGC 单条证据」改成「AIGC × 频域级联」，
    这是全项目最关键的一处改动（自动下架误伤红线 + 跨生成器召回）。
    必须用测试把四条边界钉死，防止后人改阈值时无声破坏红线。
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import rule_engine  # noqa: E402
import spectral_tool  # noqa: E402


def _ev(aigc=None, spectral=None, trufor=None, **extra):
    """构造一份最小证据字典，直接喂 rule_engine.judge()。"""
    ev = {}
    if aigc is not None:
        ev["aigc"] = {"tool": "aigc", "evidence": [{"aigc_score": aigc, "available": True}]}
    if spectral is not None:
        ev["spectral"] = {"tool": "spectral", "evidence": [{"spectral_score": spectral}]}
    if trufor is not None:
        ev["trufor"] = {"tool": "trufor", "evidence": [{"trufor_score": trufor, "available": True}]}
    ev.update(extra)
    return ev


# ── 频域特征与标定 ──────────────────────────────────────────────
def test_extract_features_returns_all_keys():
    img = REPO / "data" / "clean" / "clean_01.png"
    if not img.exists():
        pytest.skip("样本图不存在")
    f = spectral_tool.extract_features(img)
    for k in spectral_tool.FEATURE_KEYS:
        assert k in f, f"缺少特征 {k}"
    assert isinstance(f["hf_energy_ratio"], float)
    assert f["image_size"][0] > 0


def test_score_without_calib_is_zero_and_says_so():
    """没标定时必须返回 0 并在明细里说明，不能悄悄给一个看起来正常的分数。"""
    f = {k: 0.5 for k in spectral_tool.FEATURE_KEYS}
    s, parts = spectral_tool.score_from_features(f, None)
    assert s == 0.0
    assert "_note" in parts and "未标定" in parts["_note"]


def test_score_mismatched_calib_is_rejected():
    """标定文件的特征顺序与标准化数组长度对不上时，必须拒绝出分而不是错位加权。

    注意：特征数少于 FEATURE_KEYS 是**合法**的（spectral_calib.py 会丢掉离随机太近的特征），
    真正要防的是 order 与 mean/std 长度不一致 —— 那种情况会静默算错。
    """
    bad = {"feature_order": ["hf_energy_ratio", "spectral_flatness"],
           "features": {"hf_energy_ratio": {"signed_weight": 1.0},
                        "spectral_flatness": {"signed_weight": 1.0}},
           "normalization": {"mean": [0.0], "std": [1.0], "bias": 0.0}}
    f = {k: 0.5 for k in spectral_tool.FEATURE_KEYS}
    s, parts = spectral_tool.score_from_features(f, bad)
    assert s == 0.0
    assert "不匹配" in parts.get("_note", "")


def test_score_accepts_feature_subset_calib():
    """只标定了部分特征（丢掉离随机太近的）也必须能正常出分。"""
    sub = {"feature_order": ["hf_energy_ratio"],
           "features": {"hf_energy_ratio": {"signed_weight": 1.0}},
           "normalization": {"mean": [0.0], "std": [1.0], "bias": 0.0}}
    f = {k: 0.5 for k in spectral_tool.FEATURE_KEYS}
    s, _ = spectral_tool.score_from_features(f, sub)
    assert 0.0 < s < 1.0


def test_calibration_file_if_exists_is_wellformed():
    p = REPO / "results" / "spectral_calib.json"
    if not p.exists():
        pytest.skip("尚未标定（先跑 tools/spectral_calib.py）")
    c = json.loads(p.read_text(encoding="utf-8"))
    assert c.get("feature_order"), "feature_order 不能为空"
    assert c.get("normalization", {}).get("mean"), "缺标准化参数"
    for k in c["feature_order"]:
        assert k in c["features"], f"特征 {k} 缺权重"
    # 决策线必须落在 (0,1]，否则等于永远不触发或永远触发
    assert 0.0 < float(c["decision_thr"]) <= 1.0


# ── 级联判罚：四条必须钉死的边界 ────────────────────────────────
def test_aigc_hard_high_alone_still_high_risk():
    """AIGC ≥0.99 即使频域不支持，也必须保持 high_risk（不许为了降误报把召回丢了）。"""
    lvl, _ = rule_engine.judge(_ev(aigc=0.999, spectral=0.05))
    assert lvl == "high_risk"


def test_dual_evidence_agreement_escalates_to_high_risk():
    """AIGC 落在高分区 + 频域强信号 → 两条独立证据一致 → 自动下架。"""
    lvl, reasons = rule_engine.judge(_ev(aigc=0.95, spectral=0.85))
    assert lvl == "high_risk"
    assert any("双证据一致" in r for r in reasons)


def test_aigc_gray_without_spectral_stays_suspicious():
    """AIGC 在灰带但频域不支持 → 只能转人工，不能自动下架（这是灰带存在的全部意义）。"""
    lvl, _ = rule_engine.judge(_ev(aigc=0.95, spectral=0.1))
    assert lvl == "suspicious"


def test_cascade_recovers_abstain_band():
    """级联的核心价值：AIGC 弃权区 + 频域强 → 至少要捞到可疑，不能直接放过。"""
    lvl, reasons = rule_engine.judge(_ev(aigc=0.5, spectral=0.9))
    assert lvl == "suspicious"
    assert any("级联捞回" in r for r in reasons)


def test_spectral_gray_alone_is_suspicious():
    lvl, _ = rule_engine.judge(_ev(aigc=0.01, spectral=0.7))
    assert lvl == "suspicious"


def test_both_quiet_stays_inconclusive():
    lvl, _ = rule_engine.judge(_ev(aigc=0.02, spectral=0.05))
    assert lvl == "inconclusive"


def test_missing_spectral_does_not_break_judgement():
    """没有频域证据（老样本、标定缺失）时，判罚必须退化成原来的 AIGC 单条逻辑而不是崩。"""
    lvl, _ = rule_engine.judge(_ev(aigc=0.999))
    assert lvl == "high_risk"


# ── 人话结论层 ─────────────────────────────────────────────────
def test_explain_mentions_spectral_when_it_drives_the_verdict():
    ev = _ev(aigc=0.5, spectral=0.9)
    lvl, _ = rule_engine.judge(ev)
    out = rule_engine.explain(lvl, ev)
    assert out["headline"] and out["summary"] and out["what_to_do"] and out["caveat"]
    assert "频域" in out["summary"], "频域主导时人话结论必须提到频域，否则评委看不懂依据"


def test_ela_reason_carries_its_limitation():
    """ELA 对整图 AI 生成无效，这个局限必须写进给用户看的理由里。"""
    ela_ev = {"ela": {"tool": "ela", "evidence": [
        {"suspicious_regions": [{"bbox": [0, 0, 10, 10], "ela_score": 9.9}]}]}}
    lvl, reasons = rule_engine.judge(ela_ev)
    assert lvl == "suspicious"
    joined = " ".join(reasons)
    assert "不能" in joined and "AI 生成" in joined, "ELA 理由里必须写明它不能判整图 AI 生成"

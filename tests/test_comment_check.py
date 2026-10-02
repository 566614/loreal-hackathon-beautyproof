# -*- coding: utf-8 -*-
"""
评论区真实性核验的回归测试

为什么必须有：
    ① 这个场景是官方三场景之一，判错方向会直接变成"我们乱报"；
    ② 特征阈值是标定出来的，后人调阈值极易把防误报那条线调穿
       （cc_05 全是短评的真实评论必须判 credible，这条最容易被误伤）；
    ③ 「样本不足不出结论」是刻意设计，不能被"改成 3 条也判一下"破坏。
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import comment_check as cc  # noqa: E402


def _cases():
    d = json.loads((REPO / "data" / "comment_cases.json").read_text(encoding="utf-8"))
    return {c["id"]: c for c in d["cases"]}


CASES = _cases()


@pytest.mark.parametrize("cid,expected", [
    ("cc_01_flood_burst", "high_risk"),
    ("cc_02_flood_near_dup", "high_risk"),
    ("cc_03_flood_cross_post", "high_risk"),
    ("cc_04_organic_detailed", "credible"),
    ("cc_05_organic_short", "credible"),
    ("cc_06_too_few", "inconclusive"),
])
def test_case_direction(cid, expected):
    c = CASES[cid]
    r = cc.analyze_comments(c["comments"], c.get("timestamps"), c.get("cross_post_index"))
    assert r["verdict"] == expected, f"{cid} 期望 {expected} 实得 {r['verdict']} 分数 {r.get('score')}"


def test_all_short_comments_is_not_automatically_flood():
    """防误报红线：真实评论区可能全是短评。短评率高**不能**单独定灌水。"""
    r = cc.analyze_comments(["好看", "买了", "谢谢", "多少钱", "已下单", "求链接"])
    f = r["features"]
    assert f["short_rate"] == 1.0
    assert f["near_dup_rate"] == 0.0
    assert f["dup_exact_rate"] == 0.0
    assert r["verdict"] == "credible"


def test_too_few_comments_gives_no_score():
    r = cc.analyze_comments(["好用", "不错"])
    assert r["verdict"] == "inconclusive"
    assert r["score"] is None, "样本不足时必须不给分数，不能给一个看起来正常的数字"


def test_punctuation_and_emoji_do_not_hide_duplication():
    """水军靠加标点/emoji 伪装复读，归一化必须吃掉这些差异。"""
    base = ["姐妹们冲啊真的绝了", "姐妹们冲啊真的绝了!!", "姐妹们冲啊真的绝了👀", "姐妹们冲啊真的绝了～"]
    f = cc.analyze_comments(base + base)["features"]
    assert f["near_dup_rate"] == 1.0, "加标点/emoji 不应降低近似重复率"


def test_matrix_anomaly_alone_is_high_risk():
    """矩阵级异常是结构性证据，强度足以单独进高风险。"""
    idx = {f"评论{i}": 5 for i in range(8)}
    r = cc.analyze_comments([f"评论{i}" for i in range(8)], cross_post_index=idx)
    assert r["matrix_anomaly"] is True
    assert r["verdict"] == "high_risk"


def test_strong_signal_counting_not_average():
    """多条独立证据同时命中应提级，即使其中一条（如近似重复）为 0。"""
    comments = ["太好用了推荐", "推荐太好用了", "推荐太好用了啊", "推荐太好用了吧",
                "推荐太好用了呀", "推荐太好用了哦", "推荐太好用了嗯", "推荐太好用了呗"]
    r = cc.analyze_comments(comments)
    assert r["features"]["near_dup_rate"] < 0.3, "这条样本刻意让近似重复率低"
    assert r["n_strong_signals"] >= 3, "其余特征应构成多条独立强信号"
    assert r["verdict"] in ("suspicious", "high_risk")


def test_evidence_always_carries_cannot_prove():
    rep = cc.build_evidence(["好用", "不错", "推荐", "回购", "喜欢", "一般般"], "t")
    assert "不能证明" in rep["cannot_prove"] or "不是" in rep["cannot_prove"]
    assert "人工" in rep["cannot_prove"], "必须写明结论只到人工复核，不自动封号"
    ev = rep["evidence"][0]
    assert ev["thresholds"] and ev["weights"] and ev["strong_signal_thresholds"]


def test_account_profile_flags_template_reuse():
    posts = [
        {"text": "这支粉底液真的太好用了推荐给大家", "published_at": 1758000000},
        {"text": "这支粉底液真的太好用了推荐给大家呀", "published_at": 1758000600},
        {"text": "这支粉底液真的太好用了推荐给大家哦", "published_at": 1758001200},
    ]
    a = cc.analyze_account(posts)
    assert a["text_reuse_rate"] >= 0.5
    assert a["verdict"] == "suspicious"
    assert "不能证明" in a["cannot_prove"]


def test_account_profile_needs_two_posts():
    a = cc.analyze_account([{"text": "只有一条"}])
    assert a["verdict"] == "inconclusive"

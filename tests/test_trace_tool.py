# -*- coding: utf-8 -*-
"""
跨平台溯源工具的回归测试

这个工具最容易越界的地方是**把「发现第三方水印」说成「盗图已证实」**，
以及**把「图上有品牌名」当成风险**（产品图印品牌本来就正常）。
测试主要盯这两条。
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import trace_tool as tt  # noqa: E402


def _cases():
    d = json.loads((REPO / "data" / "trace_cases.json").read_text(encoding="utf-8"))
    return {c["id"]: c for c in d["cases"]}


CASES = _cases()


@pytest.mark.parametrize("cid,expected", [
    ("tr_01_watermark_vs_official", "high_risk"),
    ("tr_02_brand_not_on_image", "suspicious"),
    ("tr_03_consistent_own_post", "credible"),
    ("tr_04_no_watermark_no_claim", "credible"),
    ("tr_05_own_shot_with_watermark", "suspicious"),
])
def test_case_direction(cid, expected):
    c = CASES[cid]
    r = tt.trace_one(c["text"], c["ocr_lines"])
    assert r["verdict"] == expected, f"{cid} 期望 {expected} 实得 {r['verdict']}"


def test_watermark_alone_never_says_stolen():
    """铁律：发现第三方水印只能输出「需核实授权」，绝不能给出「盗图已证实」的结论。

    注意断言方式：cannot_prove 里**本身**含有「绝不输出『盗图已证实』」这句反面表述，
    所以不能对全文做子串排除（第一版就是这么写的，误报自己）。
    正确做法是检查**每条 finding 的文本**里没有这种结论。
    """
    c = CASES["tr_01_watermark_vs_official"]
    rep = tt.build_evidence(c["text"], c["ocr_lines"], c["id"])
    assert "授权" in rep["cannot_prove"]
    # 必须列出正当解释，避免把话说死
    assert "截图自己的笔记" in rep["cannot_prove"]
    for f in rep["evidence"][0]["findings"]:
        blob = json.dumps(f, ensure_ascii=False)
        assert "盗图已证实" not in blob, "单条 finding 不得给出盗图已证实的结论"
        assert "核实授权" in blob or "需注意" in blob or "非必然矛盾" in blob, \
            f"每条 finding 都必须带边界说明，实际：{f.get('类型')}"


def test_brand_on_product_photo_is_not_risk():
    """防误报红线：产品图上印着品牌名、文案没提，不该判风险。"""
    lines = [{"text": "MIST 保湿喷雾", "confidence": 0.9}]
    r = tt.trace_one("这次出差顺手买的，味道不错。", lines)
    assert r["verdict"] == "credible"


def test_category_conflict_is_detected():
    """品类冲突是主判据：文案说面霜、图上是原液。"""
    r = tt.compare_text_ocr("推荐珂润浸润保湿面霜",
                            [{"text": "美即 烟酰胺原液", "confidence": 0.9}])
    assert r["品类冲突"] is True
    assert "面霜" in r["文案品类"] and "原液" in r["图上品类"]


def test_same_category_is_no_conflict():
    r = tt.compare_text_ocr("自购的这支粉底液，02 号色",
                            [{"text": "02 SUPER MATCH 粉底液", "confidence": 0.9}])
    assert r["品类冲突"] is False


def test_official_watermark_is_not_treated_as_third_party():
    """平台/品牌官方图库水印不是矛盾，不能当第三方账号。"""
    lines = [{"text": "品牌官方素材库", "confidence": 0.95}]
    assert tt.find_watermarks(lines) == []


def test_matrix_flags_multi_account_and_template():
    recs = [
        {"content_id": "c1", "text": "这支粉底液真的太好用了推荐给大家",
         "ocr_evidence": [{"text": "@小李_小红书", "confidence": 0.9}]},
        {"content_id": "c2", "text": "这支粉底液真的太好用了推荐给大家呀",
         "ocr_evidence": [{"text": "@阿May_小红书", "confidence": 0.9}]},
        {"content_id": "c3", "text": "这支粉底液真的太好用了推荐给大家哦",
         "ocr_evidence": [{"text": "@小李_小红书", "confidence": 0.9}]},
    ]
    m = tt.trace_matrix(recs)
    assert m["水印账号数"] == 2
    assert m["文案复用率"] >= 0.5
    assert m["verdict"] == "suspicious"
    assert "不能证明" in m["cannot_prove"]


def test_matrix_needs_two_records():
    m = tt.trace_matrix([{"text": "只有一条", "ocr_evidence": []}])
    assert m["verdict"] == "inconclusive"


def test_low_confidence_ocr_is_ignored():
    """低置信度 OCR 不该拿来做溯源判断（认错了字比没认更糟）。"""
    lines = [{"text": "@某人_小红书", "confidence": 0.2}]
    r = tt.trace_one("这张是品牌官方提供的素材图。", lines, ocr_conf_min=0.5)
    assert r["verdict"] != "high_risk"

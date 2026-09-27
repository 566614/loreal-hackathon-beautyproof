# -*- coding: utf-8 -*-
"""Agent 决策闭环（处置 playbook + 可追溯链路 + 核验过程页）—— 不依赖深度学习，秒跑。

覆盖：
    - rule_engine.ACTION_PLAYBOOK 四档齐全且字段完整
    - rule_engine.explain 在 high_risk / suspicious / credible / inconclusive 各分支
      都返回 action_playbook 与 headline（含图文硬矛盾早返回分支，杜绝 UnboundLocalError）
    - pipeline.build_decision_trace 的 json 结构 + contributed_to_risk_level 回溯正确
    - build_trace_view.build_html 生成非空 HTML 且含关键区块；main 能写出文件

运行：
    python -m pytest tests/test_agent_close_loop.py -q
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import rule_engine            # noqa: E402
import pipeline              # noqa: E402
import build_trace_view      # noqa: E402


# ---- 构造各档位证据（与 test_core 口径一致）----
def ev_aigc(score):
    return {"aigc": {"evidence": [{"aigc_score": score, "available": True}]}}


def ev_trufor(score, ratio=0.1):
    return {"trufor": {"evidence": [{"trufor_score": score,
                                    "tampered_area_ratio": ratio,
                                    "available": True}]}}


def ev_c2pa_present():
    return {"c2pa": {"evidence": [{"c2pa_status": "present"}]}}


def ev_cross_hard():
    return {"crossmodal": {"evidence": [{
        "conflict_level": "hard",
        "contradictions": [{"text_side": "文案说原相机实拍",
                            "image_side": "图像被判整图 AI 生成"}],
    }]}}


class TestActionPlaybook(unittest.TestCase):
    def test_four_levels_present(self):
        for level in ("high_risk", "suspicious", "credible", "inconclusive"):
            self.assertIn(level, rule_engine.ACTION_PLAYBOOK,
                          f"ACTION_PLAYBOOK 缺少档位 {level}")

    def test_each_level_has_required_fields(self):
        for level, pb in rule_engine.ACTION_PLAYBOOK.items():
            for k in ("disposition", "time_limit", "owner_role", "must_human_review"):
                self.assertIn(k, pb, f"{level} 的 playbook 缺字段 {k}")
            self.assertIsInstance(pb["must_human_review"], bool,
                                  f"{level}.must_human_review 必须是布尔")

    def test_high_risk_forces_human(self):
        self.assertTrue(rule_engine.ACTION_PLAYBOOK["high_risk"]["must_human_review"])


class TestExplainActionPlaybook(unittest.TestCase):
    def _check(self, risk_level, evidence):
        ex = rule_engine.explain(risk_level, evidence)
        self.assertIn("action_playbook", ex, "explain 返回缺少 action_playbook")
        self.assertIn("headline", ex, "explain 返回缺少 headline")
        self.assertTrue(ex["headline"], "headline 不应为空")
        self.assertEqual(ex["action_playbook"],
                         rule_engine.ACTION_PLAYBOOK[risk_level])
        return ex

    def test_high_risk_trufor(self):
        self._check("high_risk", ev_trufor(0.95))

    def test_high_risk_cross_hard_early_return(self):
        # 图文硬矛盾走早返回分支，也必须带 action_playbook
        self._check("high_risk", ev_cross_hard())

    def test_suspicious_aigc(self):
        self._check("suspicious", ev_aigc(0.85))

    def test_credible_c2pa(self):
        self._check("credible", ev_c2pa_present())

    def test_inconclusive_empty(self):
        self._check("inconclusive", {})

    def test_inconclusive_trufor_low(self):
        self._check("inconclusive", ev_trufor(0.1))


class TestDecisionTrace(unittest.TestCase):
    def _trace(self, evidence, reasons, risk_level="high_risk"):
        return pipeline.build_decision_trace(
            "demo_img", "demo_img.png", risk_level, reasons, evidence)

    def test_structure(self):
        evidence = ev_trufor(0.95)
        evidence["hash"] = {"observed": "指纹未命中已知原图库"}
        reasons = ["TruFor：篡改分数 0.95（≥0.9）→ 篡改痕迹非常明显"]
        tr = self._trace(evidence, reasons, "high_risk")

        for key in ("schema", "risk_level", "reasons", "reason",
                    "action_playbook", "trace"):
            self.assertIn(key, tr, f"decision_trace 缺字段 {key}")
        self.assertEqual(tr["risk_level"], "high_risk")
        self.assertEqual(tr["action_playbook"],
                         rule_engine.ACTION_PLAYBOOK["high_risk"])
        self.assertIsInstance(tr["trace"], list)
        self.assertTrue(tr["trace"])
        for item in tr["trace"]:
            for k in ("tool", "observed", "evidence_ref",
                      "contributed_to_risk_level"):
                self.assertIn(k, item, f"trace 项缺 {k}")
            self.assertTrue(item["evidence_ref"].endswith(".json"))

    def test_contributed_backtrack(self):
        evidence = ev_trufor(0.95)
        evidence["hash"] = {"observed": "指纹未命中已知原图库"}
        reasons = ["TruFor：篡改分数 0.95（≥0.9）→ 篡改痕迹非常明显"]
        tr = self._trace(evidence, reasons, "high_risk")
        by_tool = {i["tool"]: i for i in tr["trace"]}
        self.assertTrue(by_tool["trufor"]["contributed_to_risk_level"])
        self.assertFalse(by_tool["hash"]["contributed_to_risk_level"])

    def test_reason_join(self):
        reasons = ["理由一", "理由二"]
        tr = self._trace(ev_trufor(0.9), reasons, "high_risk")
        self.assertEqual(tr["reason"], "理由一\n理由二")
        self.assertEqual(tr["reasons"], reasons)

    def test_negative_signal_not_contributing(self):
        # aigc 给出 0.0（<0.2，否定信号）虽在 reasons 里，不应记成推动 high_risk 的元凶
        evidence = ev_trufor(0.95)
        evidence["aigc"] = {"evidence": [{"aigc_score": 0.0, "available": True}]}
        reasons = [
            "AI 生成检测：AI 生成概率 0.0（<0.2）→ 看起来像真实拍摄/人工制作的图",
            "TruFor：篡改分数 0.95（≥0.9）→ 篡改痕迹非常明显",
        ]
        tr = self._trace(evidence, reasons, "high_risk")
        by_tool = {i["tool"]: i for i in tr["trace"]}
        self.assertTrue(by_tool["trufor"]["contributed_to_risk_level"])
        self.assertFalse(by_tool["aigc"]["contributed_to_risk_level"])


class TestBuildTraceView(unittest.TestCase):
    def _sample_trace(self):
        return {
            "schema": "beautyproof/decision-trace@1",
            "image_stem": "demo_img",
            "image_name": "demo_img.png",
            "generated_at": "2026-09-27T00:00:00",
            "risk_level": "high_risk",
            "reasons": ["TruFor：篡改分数 0.95 → 篡改痕迹明显"],
            "reason": "TruFor：篡改分数 0.95 → 篡改痕迹明显",
            "action_playbook": rule_engine.ACTION_PLAYBOOK["high_risk"],
            "trace": [
                {"tool": "trufor", "observed": "篡改分数 0.95",
                 "evidence_ref": "outputs/trufor_demo_img.json",
                 "contributed_to_risk_level": True},
                {"tool": "hash", "observed": "指纹未命中",
                 "evidence_ref": "outputs/hash_demo_img.json",
                 "contributed_to_risk_level": False},
            ],
        }

    def test_html_nonempty_with_key_sections(self):
        html_text = build_trace_view.build_html(self._sample_trace())
        self.assertTrue(html_text.strip())
        self.assertIn("处置建议", html_text)
        self.assertIn("核验时间线", html_text)
        self.assertIn("高风险", html_text)  # 档位中文标签
        self.assertIn("强制人工", html_text)
        self.assertIn("outputs/trufor_demo_img.json", html_text)  # 证据可追溯

    def test_main_writes_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            # 把输出目录重定向到临时目录，避免污染仓库 reports/
            build_trace_view.REP_DIR = Path(td)
            trace_file = Path(td) / "decision_trace_demo_img.json"
            trace_file.write_text(json.dumps(self._sample_trace()),
                                  encoding="utf-8")
            rc = build_trace_view.main([str(trace_file)])
            self.assertEqual(rc, 0)
            out = Path(td) / "trace_demo_img.html"
            self.assertTrue(out.exists())
            self.assertTrue(out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

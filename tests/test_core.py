# -*- coding: utf-8 -*-
"""BeautyProof 核心模块单测 —— 不依赖任何深度学习模型，秒跑。

覆盖：
    - rule_engine.judge 的定级逻辑（高/可疑/可信/无法判定）
    - planner.decide_next 的跳过规则（R1 非 JPEG / R2 命中原图库 / R3 无文案）
    - validator 的禁用措辞关（越界表述会被拦下）

运行：
    python tests/test_core.py
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import rule_engine  # noqa: E402
import planner     # noqa: E402
import validator   # noqa: E402


def ev_aigc(score):
    return {"aigc": {"evidence": [{"aigc_score": score, "available": True}]}}


def ev_trufor(score, ratio=0.1):
    return {"trufor": {"evidence": [{"trufor_score": score,
                                     "tampered_area_ratio": ratio,
                                     "available": True}]}}


class TestRuleEngine(unittest.TestCase):
    def test_high_risk_aigc(self):
        risk, reasons = rule_engine.judge(ev_aigc(0.999))
        self.assertEqual(risk, "high_risk")
        self.assertTrue(any("AI 生成" in r for r in reasons))

    def test_gray_band_aigc_suspicious_not_high_risk(self):
        # 灰带 [0.9, 0.99)：疑似但不自动下架 —— 高滤镜真实自拍常落此区间，
        # 只能升 suspicious（人工复核），绝不能 high_risk（自动下架）
        risk, _ = rule_engine.judge(ev_aigc(0.95))
        self.assertEqual(risk, "suspicious")
        self.assertNotEqual(risk, "high_risk")
        # 边界紧贴 hard_high 之下仍属灰带
        risk, _ = rule_engine.judge(ev_aigc(0.9899))
        self.assertEqual(risk, "suspicious")

    def test_abstain_aigc(self):
        # 弃权区 [0.2, 0.9)：不硬判，也不因 AIGC 单独升 suspicious
        risk, reasons = rule_engine.judge(ev_aigc(0.5))
        self.assertEqual(risk, "inconclusive")
        risk, _ = rule_engine.judge(ev_aigc(0.85))
        self.assertEqual(risk, "inconclusive")

    def test_hard_high_boundary(self):
        # ≥ 0.99（hard_high）才允许 AIGC 单独触发 high_risk
        risk, _ = rule_engine.judge(ev_aigc(0.99))
        self.assertEqual(risk, "high_risk")

    def test_gray_band_plus_trufor_corroboration_is_high_risk(self):
        # 灰带 + TruFor 强信号 corroborate → 仍应 high_risk（两个信号都硬）
        ev = {"aigc": {"evidence": [{"aigc_score": 0.95, "available": True}]},
              "trufor": {"evidence": [{"trufor_score": 0.95,
                                       "tampered_area_ratio": 0.1,
                                       "available": True}]}}
        risk, _ = rule_engine.judge(ev)
        self.assertEqual(risk, "high_risk")

    def test_high_risk_trufor(self):
        risk, _ = rule_engine.judge(ev_trufor(0.95))
        self.assertEqual(risk, "high_risk")

    def test_credible_c2pa_only_when_verified(self):
        # reviewer P0：present（未验证）不得直接升 credible
        ev_unverified = {"c2pa": {"evidence": [{"c2pa_status": "present_unverified",
                                                "c2pa_verified": False,
                                                "c2pa_declares_ai_generated": False}]}}
        risk, reasons = rule_engine.judge(ev_unverified)
        self.assertEqual(risk, "inconclusive")
        # 只有 verified 才升 credible
        ev_verified = {"c2pa": {"evidence": [{"c2pa_status": "present_verified",
                                             "c2pa_verified": True,
                                             "c2pa_declares_ai_generated": False}]}}
        risk, _ = rule_engine.judge(ev_verified)
        self.assertEqual(risk, "credible")

    def test_c2pa_declares_ai_is_suspicious_not_credible(self):
        # 生成器自声明 AI 生成：转可疑，绝不判可信
        ev = {"c2pa": {"evidence": [{"c2pa_status": "present_verified",
                                     "c2pa_verified": True,
                                     "c2pa_declares_ai_generated": True}]}}
        risk, reasons = rule_engine.judge(ev)
        self.assertEqual(risk, "suspicious")
        self.assertTrue(any("AI 生成" in r for r in reasons))

    def test_inconclusive_empty(self):
        risk, reasons = rule_engine.judge({})
        self.assertEqual(risk, "inconclusive")
        self.assertTrue(reasons)


class TestPlanner(unittest.TestCase):
    def test_r1_skip_ela_on_png(self):
        ctx = {"image_format": "PNG", "has_text": True, "image_pixels": 1000}
        skipped, _ = planner.decide_next({}, ctx, done=set())
        self.assertIn("ela", skipped)

    def test_r1_keep_ela_on_jpeg(self):
        ctx = {"image_format": "JPEG", "has_text": True, "image_pixels": 1000}
        skipped, _ = planner.decide_next({}, ctx, done=set())
        self.assertNotIn("ela", skipped)

    def test_r2_skip_on_known_original(self):
        ev = {"hash": {"evidence": [{"known_original_hit": "brand_x"}]}}
        skipped, dec = planner.decide_next(ev, {"has_text": False}, done={"hash"})
        self.assertIn("ela", skipped)
        self.assertIn("trufor", skipped)
        self.assertTrue(any(d["rule"] == "R2" for d in dec))

    def test_r3_skip_text_without_copy(self):
        ctx = {"image_format": "JPEG", "has_text": False, "image_pixels": 1000}
        skipped, _ = planner.decide_next({}, ctx, done=set())
        self.assertIn("text", skipped)
        self.assertIn("crossmodal", skipped)


class TestValidator(unittest.TestCase):
    def test_forbidden_phrase_caught(self):
        # 构造一份含越界措辞的报告 + 对应证据，验证关6 能拦下
        stem = "test_core_tmp"
        out_dir = REPO / "outputs"
        rep_dir = REPO / "reports"
        out_dir.mkdir(exist_ok=True)
        rep_dir.mkdir(exist_ok=True)
        try:
            (out_dir / f"verdict_{stem}.json").write_text(
                json.dumps({"risk_level": "high_risk", "reasons": ["x"]}),
                encoding="utf-8")
            (out_dir / f"aigc_{stem}.json").write_text(
                json.dumps({"tool": "aigc",
                            "evidence": [{"aigc_score": 0.99}],
                            "cannot_prove": "边界声明"}),
                encoding="utf-8")
            # 报告里写禁用表述 + 缺 cannot_prove 引用
            (rep_dir / f"report_{stem}.md").write_text(
                "# 报告\n## 一、结论\n确认为伪造\n## 二、检测依据\n"
                "### 1. AI 生成检测（AIGC）\n- 观察到：x\n"
                "## 三、这些工具说不清什么\n## 四、建议\n改一下\n",
                encoding="utf-8")
            results = validator.check(stem)
            names = {r["check"] for r in results}
            self.assertIn("无越界措辞", names)
            no_overreach = next(r for r in results if r["check"] == "无越界措辞")
            self.assertFalse(no_overreach["passed"])
            # 边界声明完整也该失败（没把 cannot_prove 写进报告）
            boundary = next(r for r in results if r["check"] == "边界声明完整")
            self.assertFalse(boundary["passed"])
        finally:
            for f in (out_dir / f"verdict_{stem}.json",
                      out_dir / f"aigc_{stem}.json",
                      out_dir / f"validation_{stem}.json",
                      rep_dir / f"report_{stem}.md"):
                if f.exists():
                    f.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)

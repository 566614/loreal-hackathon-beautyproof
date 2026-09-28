# -*- coding: utf-8 -*-
"""C2PA 凭证判定单测 —— 针对 reviewer P0「非空即可信」误判的回归与覆盖。

覆盖 judge() 的五类关键场景：
    1. 无凭证（returncode!=0 / 输出为空 / 非空但不可解析）
    2. 工具不可用（stderr 报找不到 c2patool）→ error
    3. 有 manifest 但本环境无法验证签名 → present_unverified（不升 credible）
    4. 有 manifest 且验证通过、但声明 AI 生成 → present_verified + declares_ai（不升 credible）
    5. 有 manifest 且验证通过、来源清晰 → present_verified（可升 credible）

这些用例不依赖 c2patool 真实安装，直接喂 judge() 的入参，秒跑。
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import c2pa_tool  # noqa: E402


class TestC2paJudge(unittest.TestCase):
    def test_no_credential_rc_nonzero(self):
        status, note, m = c2pa_tool.judge("", "no claim found", 1)
        self.assertEqual(status, "missing")
        self.assertFalse(m["verified"])

    def test_no_credential_empty_stdout(self):
        status, note, m = c2pa_tool.judge("", "", 0)
        self.assertEqual(status, "missing")
        self.assertFalse(m["verified"])

    def test_nonempty_but_unparseable_not_present(self):
        # reviewer 核心场景：一段普通非空字符串此前会被判 present，现在必须 missing
        status, note, m = c2pa_tool.judge("some random non-claim text output", "", 0)
        self.assertEqual(status, "missing")
        self.assertFalse(m["verified"])
        self.assertFalse(m["declares_ai"])

    def test_tool_unavailable_is_error(self):
        status, note, m = c2pa_tool.judge("", "找不到 c2patool：请设置环境变量 C2PA_EXE", 127)
        self.assertEqual(status, "error")

    def test_present_unverified_no_signature(self):
        # c2patool 给出 manifest，但本环境未验证签名 → present_unverified
        out = '{"active_manifest": {"label": "x", "assertions": {}}}'
        status, note, m = c2pa_tool.judge(out, "", 0)
        self.assertEqual(status, "present_unverified")
        self.assertFalse(m["verified"])
        self.assertFalse(m["declares_ai"])

    def test_present_verified_and_clear_source(self):
        # manifest + validation_status=valid + 非 AI → present_verified，可升 credible
        out = '{"active_manifest": {"label": "x", "assertions": {"c2pa.actions": []}}, ' \
              '"validation_status": "valid"}'
        status, note, m = c2pa_tool.judge(out, "", 0)
        self.assertEqual(status, "present_verified")
        self.assertTrue(m["verified"])
        self.assertFalse(m["declares_ai"])

    def test_declares_ai_generated_not_credible(self):
        # manifest 声明 AI 生成 → declares_ai=True，绝不升 credible
        out = '{"active_manifest": {"label": "x", "assertions": {"ai-generated": true}}, ' \
              '"validation_status": "valid"}'
        status, note, m = c2pa_tool.judge(out, "", 0)
        self.assertTrue(m["declares_ai"])
        # 即便是 present_verified，规则引擎也不得判 credible
        from rule_engine import judge as re_judge  # noqa: E402
        risk, reasons = re_judge({"c2pa": {"evidence": [{
            "c2pa_status": status, "c2pa_verified": m["verified"],
            "c2pa_declares_ai_generated": m["declares_ai"]}]}})
        self.assertEqual(risk, "suspicious")


if __name__ == "__main__":
    unittest.main(verbosity=2)

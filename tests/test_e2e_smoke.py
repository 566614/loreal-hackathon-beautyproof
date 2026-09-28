# -*- coding: utf-8 -*-
"""端到端冒烟测试（reviewer P1：干净环境可复现 + CI 至少一个小 e2e）。

不依赖任何重型依赖（torch / paddle / PIL 仅在函数内可选引用），在 CI 默认环境也能跑：
    1) 用一个普通文件当「图」，真实调用 c2pa_tool 脚本 → 写出证据 JSON（无 c2patool 时优雅降级为 error）
    2) 真实调用 rule_engine 脚本 → 汇总证据给出 verdict（无报警时应为 inconclusive，绝不误判可信）
    3) 直接用 rule_engine 校验 reviewer P0 的凭证门禁：
       - present_unverified 不得升 credible
       - present_verified 且声明 AI 生成 不得升 credible（应 suspicious）
       - present_verified 且非 AI 声明 才升 credible

这证明：从干净克隆开始，不装重型模型、不依赖开发者私人路径，核心决策链路能跑通且结论正确。
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
PY = sys.executable


def run_tool(script, *args):
    return subprocess.run([PY, str(TOOLS / script), *args],
                          cwd=str(REPO), capture_output=True, text=True,
                          encoding="utf-8", errors="ignore")


class TestE2ESmoke(unittest.TestCase):
    stem = "e2e_smoke_tmp"

    def setUp(self):
        # 用一个普通文本文件冒充「图片」：c2pa_tool 在找不到 c2patool 时应优雅降级为 error
        self.img = REPO / "outputs" / f"{self.stem}.bin"
        self.img.write_text("not a real image, just a smoke-test placeholder", encoding="utf-8")

    def tearDown(self):
        for f in (REPO / "outputs" / f"c2pa_{self.stem}.json",
                  REPO / "outputs" / f"hash_{self.stem}.json",
                  REPO / "outputs" / f"verdict_{self.stem}.json",
                  REPO / "reports" / f"report_{self.stem}.md",
                  self.img):
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

    def test_c2pa_tool_runs_and_does_not_crash(self):
        # 真实调用脚本；无 c2patool 时返回 error 而非崩溃
        r = run_tool("c2pa_tool.py", str(self.img))
        self.assertEqual(r.returncode, 0, msg=r.stderr[-500:])
        ev = json.loads((REPO / "outputs" / f"c2pa_{self.stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(ev["tool"], "c2pa")
        self.assertIn(ev["evidence"][0]["c2pa_status"], ("error", "missing"))

    def test_rule_engine_end_to_end_not_credible_without_proof(self):
        # 只有一条 error/missing 的 c2pa 证据 → 规则引擎应判 inconclusive，绝不 credible
        (REPO / "outputs" / f"c2pa_{self.stem}.json").write_text(json.dumps({
            "tool": "c2pa",
            "evidence": [{"c2pa_status": "error", "c2pa_verified": False,
                          "c2pa_declares_ai_generated": False}]
        }), encoding="utf-8")
        r = run_tool("rule_engine.py", self.stem)
        self.assertEqual(r.returncode, 0, msg=r.stderr[-500:])
        verdict = json.loads((REPO / "outputs" / f"verdict_{self.stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["risk_level"], "inconclusive")

    def test_credential_gate_present_unverified_not_credible(self):
        # reviewer P0：present_unverified 不得升 credible
        (REPO / "outputs" / f"c2pa_{self.stem}.json").write_text(json.dumps({
            "tool": "c2pa",
            "evidence": [{"c2pa_status": "present_unverified", "c2pa_verified": False,
                          "c2pa_declares_ai_generated": False}]
        }), encoding="utf-8")
        r = run_tool("rule_engine.py", self.stem)
        verdict = json.loads((REPO / "outputs" / f"verdict_{self.stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["risk_level"], "inconclusive")

    def test_credential_gate_verified_ai_declared_is_suspicious(self):
        # reviewer P0：验证通过但声明 AI 生成 → suspicious，绝不 credible
        (REPO / "outputs" / f"c2pa_{self.stem}.json").write_text(json.dumps({
            "tool": "c2pa",
            "evidence": [{"c2pa_status": "present_verified", "c2pa_verified": True,
                          "c2pa_declares_ai_generated": True}]
        }), encoding="utf-8")
        r = run_tool("rule_engine.py", self.stem)
        verdict = json.loads((REPO / "outputs" / f"verdict_{self.stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["risk_level"], "suspicious")

    def test_credential_gate_verified_clear_source_is_credible(self):
        # 验证通过且非 AI 声明 → 才升 credible
        (REPO / "outputs" / f"c2pa_{self.stem}.json").write_text(json.dumps({
            "tool": "c2pa",
            "evidence": [{"c2pa_status": "present_verified", "c2pa_verified": True,
                          "c2pa_declares_ai_generated": False}]
        }), encoding="utf-8")
        r = run_tool("rule_engine.py", self.stem)
        verdict = json.loads((REPO / "outputs" / f"verdict_{self.stem}.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["risk_level"], "credible")


if __name__ == "__main__":
    unittest.main(verbosity=2)

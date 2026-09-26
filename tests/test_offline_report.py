# -*- coding: utf-8 -*-
"""离线降级测试：无 Ollama、无网络环境下，--llm 路径必须仍产出完整四段式报告。

做法：
  - 用 monkeypatch 把 llm_explainer 的 Ollama 调用模拟成「不可达 / 抛异常」，
    验证 explain() 始终返回 None、绝不冒泡到 pipeline。
  - 把 pipeline.run_tool 替成即时写入最小合法 evidence JSON 的桩（仅针对 *_tool.py；
    report_generator / export_pdf 仍走真实实现），避免真的去拉深度学习模型，
    让测试快且确定。重点验证的是「LLM 解释层降级分支」，而非工具链本身。
  - 跑 pipeline.analyze(..., llm=True) —— 即 --llm 代码路径。
  - 断言：①不抛异常；②report JSON/MD 含四段式人话结论字段（一/二/三/四）。
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import pipeline  # noqa: E402
import llm_explainer  # noqa: E402

# 真实工具调用入口（钉在 import 时，供桩里「非 *_tool.py」的脚本继续走真实实现）
REAL_RUN_TOOL = pipeline.run_tool

# 取一张真实存在的测试图（data/clean/ 或 data/ai_aug/ 下）
SAMPLE_IMAGE = next(
    (p for p in [
        REPO / "data" / "clean" / "clean_01.png",
        REPO / "data" / "ai_aug" / "3D_rendered_floating_cosmetics_2026-09-26T11-46-24.png",
        REPO / "data" / "clean" / "clean_02_recompressed.png",
    ] if p.exists()),
    None,
)
assert SAMPLE_IMAGE is not None, "需要一个真实存在的测试图（data/clean/ 或 data/ai_aug/ 下）"

# 四段式报告章节（report_generator.build_report 的固定结构）
FOUR_SECTIONS = [
    "## 一、结论",
    "## 二、检测依据",
    "## 三、这些工具说不清什么",
    "## 四、建议",
]

# 每个 stem 会产生的临时文件，统一清理
_TOOL_NAMES = ("hash", "c2pa", "ela", "ocr", "aigc", "trufor", "text", "crossmodal")
_AUX_NAMES = ("analysis", "llm_explain", "verdict", "roundtable")


def _fake_run_tool(script, image_path, timeout=900):
    """桩：仅拦截 *_tool.py，按脚本名即时写一份最小但合法的 evidence JSON。

    其余脚本（report_generator.py / export_pdf.py）走真实实现，确保 Markdown 报告
    是真的由 report_generator 拼出来的。
    """
    name = Path(script).stem
    if name.endswith("_tool"):
        tool = name[:-5]  # 'hash_tool' -> 'hash'
        stem = Path(image_path).stem
        out = REPO / "outputs" / f"{tool}_{stem}.json"
        payload = {
            "tool": tool,
            "source_asset_id": stem,
            "observed": f"[{tool}] 离线桩 evidence（测试用）",
            "cannot_prove": f"[{tool}] 测试桩未声明能力边界",
            "evidence": [{}],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    # 其余脚本走真实实现
    return REAL_RUN_TOOL(script, image_path, timeout=timeout)


@pytest.fixture
def fake_tools(monkeypatch):
    monkeypatch.setattr(pipeline, "run_tool", _fake_run_tool)
    stem = SAMPLE_IMAGE.stem
    for t in _TOOL_NAMES + _AUX_NAMES:
        f = REPO / "outputs" / f"{t}_{stem}.json"
        if f.exists():
            f.unlink()
    yield
    # 清理本测试产生的证据 / 报告 / 结果，保持仓库干净
    for t in _TOOL_NAMES + _AUX_NAMES:
        f = REPO / "outputs" / f"{t}_{stem}.json"
        if f.exists():
            f.unlink()
    rf = REPO / "reports" / f"report_{stem}.md"
    if rf.exists():
        rf.unlink()


def _run_llm_path(monkeypatch, ollama_patch, verbose=False):
    """把 Ollama 调用模拟成不可达/抛异常，跑 pipeline 的 --llm 路径（fast 跳过慢模型）。"""
    monkeypatch.setattr(llm_explainer, "ollama_available", ollama_patch)
    result = pipeline.analyze(
        str(SAMPLE_IMAGE), fast=True, llm=True, verbose=verbose)
    # 主流程只落 analysis JSON；Markdown 报告由 write_markdown 生成（与 CLI 同口径）
    md = pipeline.write_markdown(result)
    return result, md


def _assert_report_ok(result, md):
    # ① 能跑到这里说明没有抛异常；结论字段存在
    assert isinstance(result["verdict"]["risk_level"], str)
    # ② 降级：不应带大模型解读段
    assert "llm_explanation" not in result, "离线分支不应产生 llm_explanation"
    # 报告 JSON 落盘
    json_path = Path(result["_output_file"])
    assert json_path.exists(), "未生成 analysis JSON"
    # 文字报告含四段式
    md_path = REPO / "reports" / f"report_{result['image']['stem']}.md"
    assert md_path.exists(), "未生成 Markdown 报告"
    md_text = md_path.read_text(encoding="utf-8")
    for sec in FOUR_SECTIONS:
        assert sec in md_text, f"报告缺少四段式章节：{sec}"


def test_offline_ollama_unreachable(fake_tools, monkeypatch):
    """场景1：Ollama 服务不可达（返回 False）→ 直接降级。"""
    result, _md = _run_llm_path(monkeypatch, lambda: False)
    _assert_report_ok(result, _md)


def test_offline_ollama_raises(fake_tools, monkeypatch):
    """场景2：Ollama 调用抛任何意外异常（连接被拒/超时/JSON 错）→ 仍降级不冒泡。"""
    def _boom():
        raise ConnectionError("connection refused / offline")
    result, _md = _run_llm_path(monkeypatch, _boom)
    _assert_report_ok(result, _md)


def test_offline_log_line_present(fake_tools, monkeypatch, capsys):
    """降级时必须打一行明确日志，便于答辩现场确认走了离线分支。"""
    monkeypatch.setattr(llm_explainer, "ollama_available", lambda: False)
    pipeline.analyze(str(SAMPLE_IMAGE), fast=True, llm=True, verbose=True)
    out = capsys.readouterr().out
    assert "[llm] offline/degraded" in out

# -*- coding: utf-8 -*-
"""报告生成器 —— 把五个工具的证据 JSON 拼成一份中文鉴定报告。

用法:
    python tools/report_generator.py <图片名不带后缀>
    例: python tools/report_generator.py post

产出: reports/report_<stem>.md

为什么不用大模型写报告（重要）:
    取证项目的命门是「结论必须可追溯到工具输出」。大模型会脑补、
    会把"可疑"说成"确认造假"，这在鉴定场景是致命的。
    所以这里用**模板拼装**：报告里每一个字都能对应到某个工具的实际输出，
    没有任何一句是模型自己编的。可复现、可审计、答辩时讲得清。
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"
REPORT_DIR = REPO / "reports"

TOOL_NAMES = {
    "hash": "内容指纹（哈希）",
    "c2pa": "内容凭证（C2PA）",
    "ela": "压缩痕迹分析（ELA）",
    "ocr": "图上文字识别（OCR）",
    "aigc": "AI 生成检测（AIGC）",
}

TOOL_ORDER = ["hash", "c2pa", "ela", "ocr", "aigc"]

RISK_TEXT = {
    "high_risk": ("高风险", "检测结果显示该图很可能有问题，**不建议直接采信**，必须人工复核。"),
    "suspicious": ("可疑", "发现了局部篡改迹象，**建议人工重点查看报告标注的可疑区域**。"),
    "credible": ("可信", "该图带有内容凭证，编辑历史可追溯，可信度较高。"),
    "inconclusive": ("无法判定", "现有工具没有发现明确的篡改信号，但**也不能证明它一定真实**——"
                                "多数网络图片本来就没有可查凭证。"),
    "NO_VERDICT": ("未出结论", "规则引擎没有产出结论，请检查证据文件是否齐全。"),
}

ADVICE = {
    "high_risk": "1) 不要直接采信该图；2) 人工核验原始出处；3) 如用于举证，保留本报告与原始证据 JSON。",
    "suspicious": "1) 人工放大查看可疑区域；2) 尽量找到原始未压缩版本比对；3) 结合图片来源（谁发的、何时发的）综合判断。",
    "credible": "1) 可正常采信，凭证可查；2) 如需更强证据，可查看 C2PA 记录的完整编辑历史。",
    "inconclusive": "1) 不能据此认定造假，也不能据此认定真实；2) 如需定论，补充原始出处、拍摄设备信息等外部证据。",
    "NO_VERDICT": "先补齐证据文件后重新生成报告。",
}


def load_evidence(stem):
    """读 outputs/ 下所有 <tool>_<stem>.json"""
    evidence = {}
    for f in OUT_DIR.glob(f"*_{stem}.json"):
        if f.name.startswith("verdict_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        tool = data.get("tool")
        if tool:
            evidence[tool] = data
    return evidence


def load_verdict(stem):
    vf = OUT_DIR / f"verdict_{stem}.json"
    if not vf.exists():
        return {}
    try:
        return json.loads(vf.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def key_facts(tool, ev_list):
    """从 evidence 里挑出关键数字，让报告有据可查"""
    if not ev_list:
        return []
    e = ev_list[0]
    facts = []
    if tool == "hash":
        if "sha256" in e:
            facts.append(f"SHA-256：`{str(e['sha256'])[:32]}…`")
        if "size_bytes" in e:
            facts.append(f"文件大小：{e['size_bytes']} 字节")
    elif tool == "c2pa":
        if "c2pa_status" in e:
            status = e["c2pa_status"]
            zh = {"present": "有凭证", "missing": "无凭证"}.get(status, status)
            facts.append(f"凭证状态：**{status}（{zh}）**")
    elif tool == "ela":
        if "mean_ela_score" in e:
            facts.append(f"整图平均误差：{e['mean_ela_score']}")
        regions = e.get("suspicious_regions") or []
        if regions:
            top = regions[0]
            facts.append(f"最可疑区域：bbox={top.get('bbox')}，误差 **{top.get('ela_score')}**"
                         f"（触发阈值 2.0）")
            facts.append(f"可疑区域总数：{len(regions)}")
    elif tool == "ocr":
        for k in ("line_count", "text_count", "count"):
            if k in e:
                facts.append(f"识别文字条数：{e[k]}")
                break
        txt = e.get("text") or e.get("joined_text")
        if txt:
            facts.append(f"识别内容：{str(txt)[:80]}")
    elif tool == "aigc":
        score = e.get("aigc_score")
        if score is None:
            facts.append("AI 生成概率：**未获取**（模型未就绪，本项跳过）")
        else:
            facts.append(f"AI 生成概率：**{score}**（触发阈值 0.7）")
        if "label" in e:
            facts.append(f"模型标签：{e['label']}")
    return facts


def build_report(stem, evidence, verdict):
    risk = verdict.get("risk_level", "NO_VERDICT")
    zh_level, zh_desc = RISK_TEXT.get(risk, RISK_TEXT["NO_VERDICT"])
    reasons = verdict.get("reasons", [])
    asset = ""
    for t in TOOL_ORDER:
        if t in evidence and evidence[t].get("source_asset_id"):
            asset = evidence[t]["source_asset_id"]
            break

    lines = []
    lines.append(f"# 图片鉴定报告：{stem}")
    lines.append("")
    if asset:
        lines.append(f"- 检测对象：`{asset}`")
    sha = ""
    if "hash" in evidence:
        ev = evidence["hash"].get("evidence") or [{}]
        sha = str(ev[0].get("sha256", ""))[:16]
    if sha:
        lines.append(f"- 内容指纹（前 16 位）：`{sha}`")
    lines.append(f"- 参与检测的工具：{len(evidence)} 个")
    lines.append(f"- **鉴定结论：{zh_level}（{risk}）**")
    lines.append("")

    lines.append("## 一、结论")
    lines.append("")
    lines.append(zh_desc)
    if reasons:
        lines.append("")
        lines.append("判定依据（规则引擎给出）：")
        for r in reasons:
            lines.append(f"- {r}")
    lines.append("")

    lines.append("## 二、检测依据（逐工具）")
    lines.append("")
    ordered = [t for t in TOOL_ORDER if t in evidence] + \
              [t for t in sorted(evidence) if t not in TOOL_ORDER]
    for i, tool in enumerate(ordered, 1):
        data = evidence[tool]
        name = TOOL_NAMES.get(tool, tool)
        lines.append(f"### {i}. {name}")
        lines.append("")
        obs = data.get("observed", "（无观察结果）")
        lines.append(f"- 观察到：{obs}")
        for fact in key_facts(tool, data.get("evidence", [])):
            lines.append(f"- {fact}")
        lines.append("")

    lines.append("## 三、这些工具说不清什么（必读）")
    lines.append("")
    lines.append("以下每一条都是对应工具**明确声明的能力边界**，超出这些范围的结论本报告不作数：")
    lines.append("")
    for tool in ordered:
        cp = evidence[tool].get("cannot_prove")
        if cp:
            name = TOOL_NAMES.get(tool, tool)
            lines.append(f"- **{name}**：{cp}")
    lines.append("")

    lines.append("## 四、建议")
    lines.append("")
    lines.append(ADVICE.get(risk, ADVICE["NO_VERDICT"]))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由模板从工具证据 JSON 直接拼装，未使用大模型生成，"
                 "所有结论均可追溯到 `outputs/` 下的原始证据文件。*")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/report_generator.py <图片名不带后缀>")
        return 1
    stem = Path(sys.argv[1]).stem
    evidence = load_evidence(stem)
    if not evidence:
        print(f"outputs/ 下没找到 {stem} 的证据文件，先跑 batch_run.py")
        return 1
    verdict = load_verdict(stem)
    report = build_report(stem, evidence, verdict)

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f"report_{stem}.md"
    out.write_text(report, encoding="utf-8")
    print(f"报告已生成: {out}")
    print(f"结论: {verdict.get('risk_level', 'NO_VERDICT')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

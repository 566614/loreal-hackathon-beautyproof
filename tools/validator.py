# -*- coding: utf-8 -*-
"""Validator —— 校验鉴定报告：字段齐不齐、结论跟证据矛盾不矛盾、有没有越界措辞。

用法:
    python tools/validator.py <图片名不带后缀>
    例: python tools/validator.py post

产出: reports/validation_<stem>.json

为什么需要它:
    报告是给人看、给评委看的最后一道关口。生成器拼错了、证据缺了、
    或者结论说得比证据能支撑的更满，都会让整个鉴定失去可信度。
    Validator 就是**出报告前的自检**，六道关全过才算合格。
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"
REPORT_DIR = REPO / "reports"

try:
    from rule_engine import THRESHOLDS
except Exception:  # noqa: BLE001
    THRESHOLDS = {"aigc_ai_score": 0.7, "ela_region_score": 2.0}

try:
    from report_generator import TOOL_NAMES
except Exception:  # noqa: BLE001
    TOOL_NAMES = {}

REQUIRED_SECTIONS = [
    "## 一、结论",
    "## 二、检测依据",
    "## 三、这些工具说不清什么",
    "## 四、建议",
]

VALID_RISK = {"high_risk", "suspicious", "credible", "inconclusive"}

# 取证红线：工具只能说"可疑/无法判定"，不能下"确认造假"这种断定
FORBIDDEN_PHRASES = [
    "确认为伪造",
    "确认造假",
    "一定是假的",
    "已经被篡改",
    "证实为",
]


def load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def load_evidence(stem):
    ev = {}
    for f in OUT_DIR.glob(f"*_{stem}.json"):
        if f.name.startswith("verdict_"):
            continue
        data = load_json(f)
        if data.get("tool"):
            ev[data["tool"]] = data
    return ev


def check(stem):
    results = []

    def add(name, ok, detail):
        results.append({"check": name, "passed": bool(ok), "detail": detail})

    report_path = REPORT_DIR / f"report_{stem}.md"
    if not report_path.exists():
        add("报告存在", False, f"找不到 {report_path}，先跑 report_generator.py")
        return results
    text = report_path.read_text(encoding="utf-8")

    # 关1：必需段落
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    add("段落齐全", not missing, "缺少段落: " + ", ".join(missing) if missing else "四个必需段落都在")

    # 关2：风险等级合法
    verdict = load_json(OUT_DIR / f"verdict_{stem}.json")
    risk = verdict.get("risk_level", "NO_VERDICT")
    add("风险等级合法", risk in VALID_RISK, f"risk_level={risk}")

    # 关3：报告引用的工具都有证据文件
    # 注意：报告里写的是中文名（如"压缩痕迹分析（ELA）"）和大写缩写，
    # 不能直接拿小写的英文 tool key 去匹配，否则会误判成"没引用"。
    def cited(tool):
        name = TOOL_NAMES.get(tool, "")
        return (name and name in text) or (tool.upper() in text)

    evidence = load_evidence(stem)
    cited_tools = [t for t in evidence if cited(t)]
    uncited = [t for t in evidence if not cited(t)]
    add("工具有据可查", not uncited and len(evidence) > 0,
        f"证据工具 {len(evidence)} 个，报告中引用 {len(cited_tools)} 个"
        + (f"，未被引用: {', '.join(uncited)}" if uncited else ""))

    # 关4：结论与证据一致性
    if risk == "suspicious":
        ela = evidence.get("ela", {}).get("evidence", [{}])
        regions = ela[0].get("suspicious_regions") or [] if ela else []
        top = regions[0].get("ela_score", 0) if regions else 0
        ok = top >= THRESHOLDS["ela_region_score"]
        add("可疑结论有 ELA 支撑", ok,
            f"ELA 最高区域 {top}，阈值 {THRESHOLDS['ela_region_score']}")
    elif risk == "high_risk":
        aigc = evidence.get("aigc", {}).get("evidence", [{}])
        score = aigc[0].get("aigc_score") if aigc else None
        ok = score is not None and score >= THRESHOLDS["aigc_ai_score"]
        add("高风险结论有 AIGC 支撑", ok,
            f"AIGC 分数 {score}，阈值 {THRESHOLDS['aigc_ai_score']}")
    else:
        add("结论一致性", True, f"风险等级 {risk}，无需额外证据支撑")

    # 关5：能力边界（cannot_prove）没有遗漏
    lost = []
    for tool, data in evidence.items():
        cp = data.get("cannot_prove")
        if cp and cp[:20] not in text:
            lost.append(tool)
    add("边界声明完整", not lost, "遗漏 cannot_prove 的工具: " + ", ".join(lost) if lost else "全部如实写入")

    # 关6：没有越界措辞（取证红线）
    hits = [p for p in FORBIDDEN_PHRASES if p in text]
    add("无越界措辞", not hits, "出现禁用表述: " + ", ".join(hits) if hits else "未发现断定式表述")

    return results


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/validator.py <图片名不带后缀>")
        return 1
    stem = Path(sys.argv[1]).stem
    results = check(stem)
    passed = sum(1 for r in results if r["passed"])

    print(f"=== 报告校验：{stem} ===")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['check']}: {r['detail']}")
    print(f"\n  通过 {passed}/{len(results)}")

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f"validation_{stem}.json"
    out.write_text(json.dumps(
        {"stem": stem, "passed": passed, "total": len(results), "checks": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细: {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

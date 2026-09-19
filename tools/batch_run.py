# -*- coding: utf-8 -*-
"""批量跑完整检测链路 + 规则引擎，验证「裁判」在数据集上是否判得对。

用法:
    python tools/batch_run.py

流程（每个样本）:
    1. hash / c2pa / ela      —— 永远跑
    2. ocr                    —— 跑，但失败不致命（PaddleOCR 较重）
    3. aigc                   —— 模型没下下来会优雅降级（写 available=False）
    4. rule_engine            —— 汇总定级
最后把每个样本的真值 vs 实际定级写进 results/dataset_eval.json，并打印汇总表。

大白话:
    我们造了几张「真图」和「P 过的图」，
    这里让五个工具逐个扫一遍，再看「裁判」给的分级对不对 ——
    P 过的应该至少判「可疑」，原图应该是「无法判定」。
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
MANIFEST = REPO / "data" / "manifest.json"
RESULTS = REPO / "results"

PY = "C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"


def run_tool(script, img_path):
    """跑单个工具脚本；失败返回 False 但不抛出（保证批量不中断）"""
    try:
        subprocess.run(
            [PY, str(script), str(img_path)],
            check=True, capture_output=True, text=True,
            encoding="utf-8", errors="ignore", timeout=420,
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [工具失败·跳过] {script.name}: {type(e).__name__}")
        return False


def read_verdict(stem):
    vf = REPO / "outputs" / f"verdict_{stem}.json"
    if not vf.exists():
        return {"risk_level": "NO_VERDICT", "reasons": ["rule_engine 未产出"]}
    try:
        return json.loads(vf.read_text(encoding="utf-8"))
    except Exception:
        return {"risk_level": "PARSE_ERROR"}


def main():
    if not MANIFEST.exists():
        print("先跑 tools/make_dataset.py 造数据集")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    RESULTS.mkdir(exist_ok=True)

    always = ["hash_tool.py", "c2pa_tool.py", "ela_tool.py"]
    optional = ["ocr_tool.py", "aigc_tool.py"]

    rows = []
    for s in manifest["samples"]:
        p = Path(s["path"])
        stem = p.stem
        print(f"\n=== 处理 {s['id']} （真值: {s['ground_truth']}）===")
        for name in always:
            run_tool(TOOLS / name, p)
        for name in optional:
            run_tool(TOOLS / name, p)
        # 规则引擎
        try:
            subprocess.run(
                [PY, str(TOOLS / "rule_engine.py"), stem],
                check=True, capture_output=True, text=True,
                encoding="utf-8", errors="ignore", timeout=120,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    [裁判失败] {type(e).__name__}")
        verdict = read_verdict(stem)
        rows.append({
            "id": s["id"],
            "ground_truth": s["ground_truth"],
            "expected_risk": s.get("expected_risk"),
            "actual_risk": verdict.get("risk_level"),
            "reasons": verdict.get("reasons", []),
            "tools_used": verdict.get("tools_used", []),
        })

    out = {"results": rows}
    (RESULTS / "dataset_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总表
    print("\n===== 数据集评测汇总 =====")
    ok = 0
    for row in rows:
        mark = "OK" if row["expected_risk"] == row["actual_risk"] else "??"
        if row["expected_risk"] == row["actual_risk"]:
            ok += 1
        print(f"  [{mark}] {row['id']:28s} 真值={row['ground_truth']:10s} "
              f"预期={str(row['expected_risk']):12s} 实际={row['actual_risk']}")
    print(f"\n  命中 {ok}/{len(rows)}  （预期=实际 即视为判对）")
    print(f"  明细: {RESULTS / 'dataset_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""批量跑完整检测链路 + 规则引擎，验证「裁判」在数据集上是否判得对。

用法:
    python tools/batch_run.py

流程:
    0. trufor（深度学习篡改检测）—— 先把所有图一次性喂给它（模型只加载一次，省时间），
       结果缓存进 outputs/trufor_cache.json
    1. hash / c2pa / ela      —— 每个样本都跑
    2. ocr                    —— 跑，但失败不致命（PaddleOCR 较重）
    3. aigc / trufor          —— 读上面的缓存 / 优雅降级
    4. rule_engine            —— 汇总定级
最后把每个样本的真值 vs 实际定级写进 results/dataset_eval.json，并打印汇总表。

大白话:
    我们造了几张「真图」和「P 过的图」，
    这里让六个工具逐个扫一遍，再看「裁判」给的分级对不对 ——
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


def prewarm_trufor(image_paths):
    """把所有图一次性交给 TruFor（模型只加载一次），结果写进缓存

    这样后面逐样本跑 trufor_tool.py 时是读缓存，不会每张图都重载模型。
    """
    print("\n=== TruFor 预热：一次性处理全部样本（模型只加载一次）===")
    try:
        sys.path.insert(0, str(TOOLS))
        import trufor_tool
        cache, err = trufor_tool.run_trufor_batch(image_paths)
        if err:
            print(f"  [提示] TruFor 未跑成（会自动降级为「无法判断」）: {err[:300]}")
            return False
        ok = sum(1 for v in cache.values() if v.get("available"))
        print(f"  TruFor 完成：{ok}/{len(image_paths)} 张拿到分数")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [提示] TruFor 跳过（{type(e).__name__}），其余工具照常跑")
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
    optional = ["ocr_tool.py", "aigc_tool.py", "trufor_tool.py"]

    all_paths = [Path(s["path"]) for s in manifest["samples"]]
    prewarm_trufor(all_paths)

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

    # ---------------------------------------------------------------- 两套命中口径
    # exact = 档位和清单逐字一致（严格口径）
    # hit   = 方向抓对了没（篡改图有没有被报警 / 干净图有没有被误报）
    #         判得比预期更重（suspicious 变 high_risk）是"偏严"，不是漏判，算 hit 不算 exact
    RISK_ALARM = {"suspicious", "high_risk"}    # 判为"有问题"
    RISK_CLEAN = {"credible", "inconclusive"}   # 判为"没抓到问题"

    n_exact = 0
    n_hit = 0
    for row in rows:
        gt = row["ground_truth"]
        exp, act = row["expected_risk"], row["actual_risk"]
        is_exact = (exp == act)
        is_hit = (act in RISK_CLEAN) if gt == "clean" else (act in RISK_ALARM)
        row["hit"] = is_hit
        row["exact"] = is_exact
        n_exact += int(is_exact)
        n_hit += int(is_hit)

    # 把命中标记回写进明细，方便后面看报告
    (RESULTS / "dataset_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 数据集评测汇总 =====")
    for row in rows:
        mark = "OK" if row["hit"] else "!!"   # !! = 真漏判/真误报，必须处理
        note = "" if row["exact"] else \
            f"   ← 档位偏{'严' if row['actual_risk'] == 'high_risk' else '松'}（清单预期 {row['expected_risk']}）"
        print(f"  [{mark}] {row['id']:28s} 真值={row['ground_truth']:10s} "
              f"预期={str(row['expected_risk']):12s} 实际={row['actual_risk']}{note}")

    print(f"\n  命中 {n_hit}/{len(rows)}  （方向判对：篡改图被报警 / 干净图不误报）")
    print(f"  档位严丝合缝 {n_exact}/{len(rows)}  （预期=实际，逐字一致）")
    print(f"  明细: {RESULTS / 'dataset_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

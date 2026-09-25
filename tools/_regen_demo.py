# -*- coding: utf-8 -*-
"""
一次性把 data/manifest.json 里登记的全部正式样本，用「完整（非 --fast）」流水线重跑一遍，
确保 outputs/analysis_*.json 与 reports/report_*.md 都反映 v3 模型（AIGC v3 + TruFor）。

用法（用 venv 的 python）：
    python tools/_regen_demo.py

注意：pipeline.analyze 内部每个工具都是独立子进程，AIGC/TruFor 每次都会重新加载模型，
所以对 6 张图跑全套会比较慢（CPU 上约 5~8 分钟），建议后台跑。
"""
import json
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import pipeline  # noqa: E402

REPO = TOOLS.parent


def main():
    m = json.loads((REPO / "data" / "manifest.json").read_text(encoding="utf-8"))
    samples = m["samples"]
    t0 = time.time()
    for s in samples:
        p = s["path"]
        print(f"\n===== {s['id']} : {p} =====", flush=True)
        if not Path(p).exists():
            print("  图片不存在，跳过", flush=True)
            continue
        try:
            res = pipeline.analyze(p, fast=False, verbose=True)
        except Exception as e:  # noqa: BLE001
            print(f"  analyze 异常: {type(e).__name__}: {e}", flush=True)
            continue
        v = res["verdict"]
        print(f"  结论={v['risk_level']} 工具={sorted(res['evidence'].keys())}", flush=True)
        try:
            md = pipeline.write_markdown(res)
            print(f"  报告={md}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  write_markdown 异常: {type(e).__name__}: {e}", flush=True)
    print(f"\n全部完成 用时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
批量判「新来的素材」—— 只跑对判真伪有用的快工具，跳过 OCR（太慢）

用法：
    python tools/judge_incoming.py

为什么单独写这个：
    pipeline 全套要跑 OCR（约 60 秒/张），30 张就是半小时。
    而判断「整图是不是 AI 生成」主要靠 aigc + trufor，
    OCR 和 C2PA 对这一问基本没贡献，所以这里跳过。

⚠️ 铁律：跑之前不看标签。先看模型说什么，再跟人工标签对质。
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
TOOLS = REPO / "tools"

# 只跑这几个：hash（身份+原图库）、ela（压缩）、aigc（本域微调）、trufor（篡改定位）
FAST_TOOLS = ["hash", "ela", "aigc", "trufor"]


def run_one(img):
    stem = img.stem
    out = {"file": img.name}
    for tool in FAST_TOOLS:
        proc = subprocess.run(
            [PY, str(TOOLS / f"{tool}_tool.py"), str(img)],
            cwd=str(REPO), capture_output=True, text=True,
            encoding="utf-8", errors="ignore", timeout=600,
        )
        f = REPO / "outputs" / f"{tool}_{stem}.json"
        if f.exists():
            try:
                out[tool] = json.loads(f.read_text(encoding="utf-8"))["evidence"][0]
            except Exception:  # noqa: BLE001
                out[tool] = None
        else:
            out[tool] = None
    a = out.get("aigc") or {}
    t = out.get("trufor") or {}
    h = out.get("hash") or {}
    e = out.get("ela") or {}
    out["_summary"] = {
        "aigc_score": a.get("aigc_score"),
        "trufor_score": t.get("trufor_score"),
        "format": None,
        "size_bytes": h.get("size_bytes"),
        "known_original_hit": h.get("known_original_hit"),
        "mean_ela": e.get("mean_ela_score"),
    }
    return out


def main():
    groups = [("AI(待验)", "data/incoming_ai"), ("真实(待验)", "data/incoming_real")]
    all_rows = []
    for tag, d in groups:
        p = REPO / d
        imgs = sorted([x for x in p.iterdir()
                       if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")])
        print(f"\n===== {tag} · {len(imgs)} 张 =====")
        for img in imgs:
            r = run_one(img)
            s = r["_summary"]
            # 格式单独读（hash 证据里没有）
            try:
                from PIL import Image
                with Image.open(img) as im:
                    s["format"] = im.format
                    s["width"], s["height"] = im.size
            except Exception:  # noqa: BLE001
                pass
            r["_group"] = tag
            all_rows.append(r)
            print(f"  {img.name[:38]:40s} "
                  f"AIGC={str(s['aigc_score']):>7s} "
                  f"TruFor={str(s['trufor_score']):>7s} "
                  f"{str(s['format']):4s} {str(s['width'])}x{str(s['height'])}")

    (REPO / "results" / "incoming_judge.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总：两组的分数分布是否分开
    print("\n===== 分布对比 =====")
    for tag, _ in groups:
        scores = [r["_summary"]["aigc_score"] for r in all_rows
                  if r["_group"] == tag and r["_summary"]["aigc_score"] is not None]
        if scores:
            print(f"  {tag}: AIGC {min(scores):.4f} ~ {max(scores):.4f}  (n={len(scores)})")
    print("\n已保存到 results/incoming_judge.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

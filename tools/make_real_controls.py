# -*- coding: utf-8 -*-
"""
造「阴性对照」样本 —— 用真实图做一次很重的压缩，真值仍然是"干净"

为什么要它：
    评委一定会问：压缩这么狠的图，你们会不会误报？
    这张图就是答案的物证：它被压到 JPEG 质量 30（模拟微信传了三手），
    真值依然是 clean —— 系统如果把它判成可疑，说明误报，要回头调阈值。

用法：
    ./run.sh tools/make_real_controls.py
产出：
    data/clean/clean_02_recompressed.png
    并把真值登记进 data/manifest.json（幂等：重复跑不会重复登记）
"""
import io
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "manifest.json"
SRC = REPO / "data" / "raw" / "base.png"
DST = REPO / "data" / "clean" / "clean_02_recompressed.png"


def main():
    if not SRC.exists():
        print(f"找不到真实原图: {SRC}")
        return 1

    # 读原图 → JPEG 质量 30 重压 → 再存回 PNG（格式还是图，但压缩损伤已留在像素里）
    img = Image.open(SRC).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30, optimize=True)
    buf.seek(0)
    recompressed = Image.open(buf)
    recompressed.save(DST, format="PNG")
    print(f"已生成重压缩阴性对照: {DST}（{DST.stat().st_size // 1024} KB，JPEG q30 损伤）")

    # 登记真值（幂等）
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "id": "clean_02_recompressed",
        "path": str(DST),
        "ground_truth": "clean",
        "expected_risk": "inconclusive",
        "note": "真实图经 JPEG q30 重压缩（模拟多次转发），真值仍是 clean，用于验证不误报",
    }
    if not any(s.get("id") == entry["id"] for s in manifest["samples"]):
        manifest["samples"].append(entry)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已登记进 data/manifest.json")
    else:
        print("manifest 里已有这条，跳过登记")
    return 0


if __name__ == "__main__":
    sys.exit(main())

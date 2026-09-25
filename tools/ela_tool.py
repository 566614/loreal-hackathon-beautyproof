# -*- coding: utf-8 -*-
# 来源：经典算法自研实现 —— ELA 为公共方法论，本文件实现与统一证据封装为 BeautyProof 团队原创
"""
ELA 工具 —— 误差水平分析（TruFor 的降级替代方案），输出统一格式的证据 JSON

用法：
    python tools/ela_tool.py <图片路径>

大白话：
    JPEG 图片每保存一次都会损失一点画质。一张被 P 过的图，
    被改的那块区域通常是「新贴上去的」，压缩次数跟周围不一样，
    所以重新压一遍之后，那块区域的误差会比周围大。
    ELA 就是把图重新压一遍，看哪块「跟周围对不上」。

能证明什么：
    这张图里哪些区域的压缩痕迹跟其他区域不一致（可疑区域）。

不能证明什么（铁律）：
    不能证明那里一定被改过 —— 反复保存、截图、平台二次压缩都会造成同样的现象。
    ELA 是「提示哪里值得人工看一眼」，不是「判案」。

⚠️ 重要限制：
    ELA 只对 JPEG 有效。如果原图是 PNG，PNG→JPEG 这一步本身就会
    产生巨大差异，结果不可信（脚本会在这种情况下给出提示）。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BLOCKS = 8
JPEG_QUALITY = 90


def run_ela(image_path, blocks=BLOCKS, quality=JPEG_QUALITY):
    """重新压缩一遍，算出每个区块的误差水平。返回 (整图平均误差, 前3个最可疑区块, 图片真实格式)"""
    original = Image.open(image_path)
    actual_format = (original.format or "").upper()
    img = original.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    again = Image.open(buf).convert("RGB")

    a = np.asarray(img).astype(np.int16)
    b = np.asarray(again).astype(np.int16)
    diff = np.abs(a - b).mean(axis=2)

    h, w = diff.shape
    bh, bw = h // blocks, w // blocks

    regions = []
    for r in range(blocks):
        for c in range(blocks):
            block = diff[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            regions.append({
                "bbox": [int(c * bw), int(r * bh), int((c + 1) * bw), int((r + 1) * bh)],
                "ela_score": round(float(block.mean()), 3),
            })

    regions.sort(key=lambda x: -x["ela_score"])
    return round(float(diff.mean()), 3), regions[:3], actual_format


def build_evidence(image_path, mean_score, top_regions, actual_format):
    if actual_format != "JPEG":
        warn = f"注意：这张图的真实格式是 {actual_format}，不是 JPEG。非 JPEG 转 JPEG 本身就会产生很大差异，本结果仅作演示、不可信。"
    else:
        warn = ""

    top = top_regions[0] if top_regions else {"bbox": None, "ela_score": 0.0}
    observed = (
        f"整图平均误差水平 {mean_score}；"
        f"误差最高的区域 bbox={top['bbox']}，该区域误差 {top['ela_score']}。"
        f"{warn}"
    )

    return {
        "tool": "ela",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "cannot_prove": "ELA 只是压缩痕迹分析，不能证明某处一定被篡改；截图、反复保存、平台压缩都会造成同样现象",
        "evidence": [
            {"mean_ela_score": mean_score, "suspicious_regions": top_regions},
        ],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/ela_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    mean_score, top_regions, actual_format = run_ela(image_path)
    report = build_evidence(image_path, mean_score, top_regions, actual_format)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"ela_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

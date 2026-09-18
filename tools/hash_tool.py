# -*- coding: utf-8 -*-
"""
哈希工具 —— 给图片算一个「内容指纹」，输出统一格式的证据 JSON

用法：
    python tools/hash_tool.py <图片路径>

大白话：
    哈希就像给文件按指纹。同一个文件（一个字节都没改）按出来的指纹永远一样；
    只要改了一个像素，指纹就完全变了。

能证明什么：
    这两份文件是不是「一模一样」——可以用来查这张图是不是网上已有的原图。

不能证明什么（铁律）：
    不能证明图片真伪，不能证明有没有被 P 过。改过和没改过，指纹都长那样。
"""
import hashlib
import json
import sys
from pathlib import Path


def compute_asset_id(image_path):
    """读出文件的全部字节，算 sha256 指纹。返回 (指纹, 文件字节数)"""
    with open(image_path, "rb") as f:
        data = f.read()
    return "sha256:" + hashlib.sha256(data).hexdigest(), len(data)


def build_evidence(image_path, asset_id, size_bytes):
    """打包成统一证据格式（和 OCR 工具一模一样的结构）"""
    return {
        "tool": "hash",
        "source_asset_id": Path(image_path).name,
        "observed": f"这张图的内容指纹是 {asset_id}，文件大小 {size_bytes} 字节",
        "cannot_prove": "哈希只证明文件的身份，不能证明图片真伪、不能判断是否被篡改",
        "evidence": [
            {"asset_id": asset_id, "size_bytes": size_bytes},
        ],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/hash_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    asset_id, size_bytes = compute_asset_id(image_path)
    report = build_evidence(image_path, asset_id, size_bytes)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"hash_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

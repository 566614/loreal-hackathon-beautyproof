# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队哈希指纹工具与统一证据封装
"""
哈希工具 —— 给图片算一个「内容指纹」，输出统一格式的证据 JSON

用法：
    python tools/hash_tool.py <图片路径>

大白话：
    哈希就像给文件按指纹。同一个文件（一个字节都没改）按出来的指纹永远一样；
    只要改了一个像素，指纹就完全变了。

能证明什么：
    这两份文件是不是「一模一样」——可以用来查这张图是不是网上已有的原图。
    如果它的指纹能在「品牌方已知原图库」（data/known_originals/）里对上，
    说明这张图一个字节都没被改过，就是品牌方给的原图本身。

不能证明什么（铁律）：
    不能证明图片真伪，不能证明有没有被 P 过。改过和没改过，指纹都长那样。
    ⚠️ 特别提醒：没在原图库里对上，也不代表这张图是假的 ——
       很可能只是品牌方还没把这张原图交给我们。库里没有 ≠ 图有问题。
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


KNOWN_DIR = Path(__file__).resolve().parent.parent / "data" / "known_originals"


def match_known_original(image_path, asset_id):
    """拿指纹去「品牌方已知原图库」比对

    命中意味着：这张图和品牌方给的原图一个字节都不差 —— 这是最强的一种清白证据，
    Agent 可以据此跳过压缩痕迹 / 篡改检测（都没被改过，查了也是白查）。

    没命中什么都不说明，可能只是库里还没这张，绝不能反过来当成可疑信号。
    """
    if not KNOWN_DIR.is_dir():
        return None
    try:
        for p in sorted(KNOWN_DIR.iterdir()):
            if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            if p.resolve() == Path(image_path).resolve():
                # 传进来的就是库里那张本身，不算「比对命中」，避免自证
                continue
            try:
                if "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() == asset_id:
                    return p.name
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return None
    return None


def build_evidence(image_path, asset_id, size_bytes):
    """打包成统一证据格式（和 OCR 工具一模一样的结构）"""
    hit = match_known_original(image_path, asset_id)
    if hit:
        observed = (f"这张图的内容指纹是 {asset_id}，文件大小 {size_bytes} 字节。"
                    f"该指纹在品牌方已知原图库里命中「{hit}」——"
                    f"说明这张图与品牌方提供的原图一个字节都不差。")
    else:
        observed = (f"这张图的内容指纹是 {asset_id}，文件大小 {size_bytes} 字节。"
                    f"该指纹未在品牌方已知原图库中命中"
                    f"（库里没有 ≠ 图有问题，很可能只是还没收录这张原图）。")

    return {
        "tool": "hash",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "cannot_prove": "哈希只证明文件的身份，不能证明图片真伪、不能判断是否被篡改",
        "evidence": [
            {
                "asset_id": asset_id,
                "size_bytes": size_bytes,
                "known_original_hit": hit,
                "known_originals_dir": str(KNOWN_DIR.name) if KNOWN_DIR.is_dir() else None,
            },
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

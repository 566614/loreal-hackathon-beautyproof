# -*- coding: utf-8 -*-
"""自动造一份「篡改 vs 干净」数据集，用于验证检测链路 + 规则引擎裁判是否有效。

用法:
    python tools/make_dataset.py

产出:
    data/raw/base.png            底图（从素材 post.jpg 复制，git 不跟踪）
    data/clean/*.png             干净对照样本
    data/tampered/*.png          篡改样本（copy_move / splice / text_edit）
    data/manifest.json           真值清单：每个样本的预期篡改类型 + 预期风险档

大白话:
    我们只有一张真实美妆图。为了证明「裁判」能分辨真假，
    我拿这张图动手脚造出几张「被 P 过」的图，
    再让五个工具去扫它们 —— 被 P 过的应该触发「可疑/高风险」，原图应该「无法判定」。
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
# reviewer P1：旧逻辑写死仓库外的私人路径 ../my-beautyproof-practice/assets/post.jpg，
# 新用户克隆本仓库后跑不通。改为三级回退：
#   ① 仓库内 data/raw/base.png（优先，干净克隆若已自带则直接用）
#   ② 开发者机器上的私人 post.jpg（兼容旧环境，存在才用）
#   ③ 都没有 → 生成一张合成底图（仅用于链路/CI 跑通，非真实评测数据），并打印告警
BASE_SRC = REPO.parent / "my-beautyproof-practice" / "assets" / "post.jpg"
REPO_LOCAL_BASE = REPO / "data" / "raw" / "base.png"
RAW_DIR = REPO / "data" / "raw"
CLEAN_DIR = REPO / "data" / "clean"
TAMP_DIR = REPO / "data" / "tampered"
MANIFEST = REPO / "data" / "manifest.json"


def load_font(size=28):
    """尽量用系统中文字体，找不到就退回默认（英文/数字仍能画）"""
    for cand in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _make_synthetic_base(size=(900, 1200)):
    """生成合成底图（仅用于干净环境/CI 跑通链路；不是真实评测样本）。

    一张柔和渐变 + 居中随机色块，保证 copy_move/splice/text_edit 能造出可检测的篡改痕迹。
    """
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        t = y / size[1]
        arr[y, :, 0] = int(210 - 120 * t)   # R
        arr[y, :, 1] = int(180 - 60 * t)     # G
        arr[y, :, 2] = int(160 + 40 * t)     # B
    # 随机几个柔和色块，模拟「皮肤/背景」区域，供 copy_move 取样
    rng = np.random.default_rng(20260928)
    for _ in range(6):
        cx, cy = rng.integers(0, size[0]), rng.integers(0, size[1])
        s = rng.integers(60, 160)
        col = rng.integers(120, 240, size=3)
        y0, y1 = max(0, cy - s // 2), min(size[1], cy + s // 2)
        x0, x1 = max(0, cx - s // 2), min(size[0], cx + s // 2)
        arr[y0:y1, x0:x1] = col
    return Image.fromarray(arr).convert("RGB")


def make_base():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = REPO_LOCAL_BASE
    if dst.exists():
        return Image.open(dst).convert("RGB")
    if BASE_SRC.exists():
        shutil.copyfile(BASE_SRC, dst)
        print(f"  (使用开发者机器底图 {BASE_SRC} → {dst})")
        return Image.open(dst).convert("RGB")
    # 干净克隆且无可读底图：生成合成底图，保证整条链路可跑通（非真实评测数据）
    base = _make_synthetic_base()
    base.save(dst)
    print("  ⚠ 未找到真实底图，已生成「合成底图」data/raw/base.png 仅供链路/CI 跑通；"
          "真实评测请用团队自产美妆原图替换它。")
    return base


def add_copy_move(img):
    """把图里一块区域复制后贴到别处 —— 模拟「仿制图章」遮盖/复制"""
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    s = 90  # 补丁大小
    # 源：左上角附近一块皮肤区域
    sx, sy = 60, 200
    # 目标：贴到右下角空处
    dx, dy = w - s - 40, h - s - 40
    patch = arr[sy:sy + s, sx:sx + s].copy()
    arr[dy:dy + s, dx:dx + s] = patch
    out = Image.fromarray(arr)
    # 画个框标注篡改区（仅用于人类看，不影响检测）
    d = ImageDraw.Draw(out)
    d.rectangle([dx, dy, dx + s, dy + s], outline=(255, 0, 0), width=2)
    return out, {"tamper_type": "copy_move", "box": [dx, dy, dx + s, dy + s]}


def add_splice(img):
    """从图里抠一块，挪到另一处并轻微缩放 —— 模拟「把别处的内容拼进来」"""
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    s = 110
    sx, sy = 40, 120
    dx, dy = 240, 360
    patch = arr[sy:sy + s, sx:sx + s].copy()
    # 缩放一点点制造接缝
    patch_img = Image.fromarray(patch).resize((s + 14, s + 14))
    arr[dy:dy + s + 14, dx:dx + s + 14] = np.asarray(patch_img)
    out = Image.fromarray(arr)
    d = ImageDraw.Draw(out)
    d.rectangle([dx, dy, dx + s + 14, dy + s + 14], outline=(255, 165, 0), width=2)
    return out, {"tamper_type": "splice", "box": [dx, dy, dx + s + 14, dy + s + 14]}


def add_text_edit(img):
    """在图上覆盖一段新文字 —— 模拟「改了图上文案/水印」"""
    out = img.copy()
    d = ImageDraw.Draw(out)
    font = load_font(30)
    # 在原水印附近覆盖新文字（红色，明显覆盖）
    d.text((20, 600), "@美妆鉴定局_OFFICIAL", fill=(220, 20, 60), font=font)
    # 再画一条横线假装改了原水印
    d.line([(10, 595), (330, 595)], fill=(220, 20, 60), width=3)
    return out, {"tamper_type": "text_edit", "box": [10, 595, 330, 640]}


def main():
    base = make_base()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    TAMP_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"base": str(RAW_DIR / "base.png"), "samples": []}

    # 干净对照
    clean_path = CLEAN_DIR / "clean_01.png"
    base.save(clean_path)
    manifest["samples"].append({
        "id": "clean_01",
        "path": str(clean_path),
        "ground_truth": "clean",
        "expected_risk": "inconclusive",
    })

    # 篡改样本
    makers = [add_copy_move, add_splice, add_text_edit]
    for i, maker in enumerate(makers, 1):
        out, meta = maker(base)
        name = f"tampered_{i:02d}_{meta['tamper_type']}"
        p = TAMP_DIR / f"{name}.png"
        out.save(p)
        manifest["samples"].append({
            "id": name,
            "path": str(p),
            "ground_truth": meta["tamper_type"],
            "tamper_box": meta["box"],
            "expected_risk": "suspicious",  # 篡改样本预期至少触发「可疑」
        })

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据集已生成：{len(manifest['samples'])} 个样本")
    print(f"  - 干净对照: {CLEAN_DIR}")
    print(f"  - 篡改样本: {TAMP_DIR}")
    print(f"  - 真值清单: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

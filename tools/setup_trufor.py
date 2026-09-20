# -*- coding: utf-8 -*-
"""
TruFor 部署脚本 —— 把官方代码 + 权重放到项目里能被 trufor_tool.py 找到的位置

用法：
    python tools/setup_trufor.py

它做三件事：
    1. 找我下载好的官方源码 zip / 已解压目录，部署到 models/TruFor/TruFor_train_test
    2. 找官方权重 zip（TruFor_weights.zip），把 trufor.pth.tar 解压到
       models/TruFor/TruFor_train_test/pretrained_models/
    3. 逐项体检：代码 / 基础权重 / 最终权重 是否齐全，缺什么明确告诉你

大白话：
    TruFor 模型由三块拼起来：程序代码 + 两个基础权重 + 一个最终训练权重。
    这个脚本就是把它们各就各位摆好，摆完告诉你"齐了没"。
"""
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTER = REPO.parent  # 黑客松/ 目录（我下载的东西临时放这）
DEST = REPO / "models" / "TruFor" / "TruFor_train_test"

SRC_ZIP_CANDS = [
    OUTER / "_TruFor_src.zip",
    REPO / "models" / "_TruFor_src.zip",
]
WEIGHT_ZIP_CANDS = [
    OUTER / "_TruFor_weights.zip",
    REPO / "models" / "_TruFor_weights.zip",
]


def find_first(cands):
    for c in cands:
        if c.exists() and c.stat().st_size > 1024:
            return c
    return None


def deploy_source():
    """把官方源码里的 TruFor_train_test 部署到 models/TruFor/ 下"""
    if (DEST / "test.py").exists():
        return True, f"代码已就位: {DEST}"

    # 优先用已解压的目录
    for base in [OUTER / "_trufor_src", OUTER / "_trufor_prep", REPO / "models" / "_trufor_src"]:
        if base.exists():
            hits = [p for p in base.rglob("test.py") if p.parent.name == "TruFor_train_test"]
            if hits:
                src_dir = hits[0].parent
                DEST.parent.mkdir(parents=True, exist_ok=True)
                if DEST.exists():
                    shutil.rmtree(DEST, ignore_errors=True)
                shutil.copytree(src_dir, DEST)
                return True, f"已从解压目录部署: {src_dir} -> {DEST}"

    # 否则解压 zip
    z = find_first(SRC_ZIP_CANDS)
    if z is None:
        return False, ("没找到 TruFor 源码。请先下载："
                       "curl -L -x http://127.0.0.1:7897 -o 黑客松/_TruFor_src.zip "
                       "https://codeload.github.com/grip-unina/TruFor/zip/refs/heads/main")

    tmp = OUTER / "_trufor_unzip"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    hits = [p for p in tmp.rglob("test.py") if p.parent.name == "TruFor_train_test"]
    if not hits:
        return False, f"源码 zip 里没找到 TruFor_train_test/test.py：{z}"
    src_dir = hits[0].parent
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        shutil.rmtree(DEST, ignore_errors=True)
    shutil.copytree(src_dir, DEST)
    return True, f"已从 zip 部署: {z.name} -> {DEST}"


def deploy_weights():
    """把最终训练权重 trufor.pth.tar 放进 pretrained_models/"""
    pm = DEST / "pretrained_models"
    pm.mkdir(parents=True, exist_ok=True)

    for cand in [pm / "trufor.pth.tar", pm / "TruFor_weights" / "trufor.pth.tar"]:
        if cand.exists():
            return True, f"最终权重已就位: {cand}"

    z = find_first(WEIGHT_ZIP_CANDS)
    if z is None:
        return False, ("没找到 TruFor 权重 zip。请下载："
                       "curl -L -x http://127.0.0.1:7897 -o 黑客松/_TruFor_weights.zip "
                       "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip")

    tmp = OUTER / "_trufor_w_unzip"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
    except zipfile.BadZipFile:
        return False, f"权重 zip 损坏或不完整（可能还在下载中）: {z} ({z.stat().st_size} bytes)"

    hits = sorted(tmp.rglob("trufor*.pth.tar")) or sorted(tmp.rglob("*.pth.tar"))
    if not hits:
        return False, f"权重 zip 里没找到 .pth.tar：{z}"
    shutil.copy2(hits[0], pm / "trufor.pth.tar")
    return True, f"已部署最终权重: {hits[0].name} -> {pm / 'trufor.pth.tar'}"


def check():
    """体检：列清楚还差什么"""
    items = [
        ("官方代码 test.py", DEST / "test.py"),
        ("基础权重 noiseprint++", DEST / "pretrained_models" / "noiseprint++" / "noiseprint++.th"),
        ("基础权重 SegFormer", DEST / "pretrained_models" / "segformers" / "mit_b2.pth"),
        ("最终权重 trufor.pth.tar", DEST / "pretrained_models" / "trufor.pth.tar"),
    ]
    ok = True
    print("\n===== TruFor 部署体检 =====")
    for name, p in items:
        mark = "OK  " if p.exists() else "缺失"
        if not p.exists():
            ok = False
        size = f"  ({round(p.stat().st_size/1024/1024,1)}MB)" if p.exists() else ""
        print(f"  [{mark}] {name}{size}")
    return ok


def main():
    print("== 1/2 部署官方代码 ==")
    ok1, msg1 = deploy_source()
    print(f"  {msg1}")

    print("\n== 2/2 部署官方权重 ==")
    ok2, msg2 = deploy_weights()
    print(f"  {msg2}")

    all_ok = check()
    print()
    if all_ok:
        print("全部就位，可以跑检测了：")
        print("  python tools/trufor_tool.py assets/post.jpg")
    else:
        print("还有东西没齐（见上面「缺失」项），补齐后重跑本脚本即可。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

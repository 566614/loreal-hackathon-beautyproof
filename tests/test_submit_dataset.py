# -*- coding: utf-8 -*-
"""
初赛三交付物之一「测试数据包」的官方格式回归测试

为什么必须有：
    ① 「测试数据是否满足提交格式」是官方明列的评分维度（/information 子页），
       格式不齐 = 硬扣分，跟模型准不准无关；
    ② 2026-10-02 实测踩过：7 个样本里有 4 个**没有** 文案.txt（当时理由是
       「纯视觉样本不伪造配文」），格式检查一票否决，所以这个坑必须钉死；
    ③ 本测试的断言对象是**落盘的 zip**，不是源码 —— 保证评委拿到的那个包是齐的。
"""
import re
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ZIP = REPO / "outputs" / "BeautyProof_submit_dataset.zip"

# 官方格式：每个 sample_XXX/ 必须齐这三样
REQUIRED = ["README.txt", "种草正文.txt", "image_1.jpg"]
SAMPLE_DIR_RE = re.compile(r"^sample_\d{3,}$")


def _names():
    if not ZIP.exists():
        pytest.skip(f"提交包不存在：{ZIP}（先跑 python tools/make_submit_dataset.py）")
    with zipfile.ZipFile(ZIP) as z:
        return z.namelist()


NAMES = _names()
SAMPLE_DIRS = sorted({n.split("/")[0] for n in NAMES if SAMPLE_DIR_RE.match(n.rstrip("/"))})


def _files_in(sample):
    return {n.split("/", 1)[1] for n in NAMES
            if n.startswith(sample + "/") and not n.endswith("/")}


def test_提交包内至少有一个样本文件夹():
    assert SAMPLE_DIRS, "zip 里找不到任何 sample_XXX/ 目录"


def test_每个样本的文件清单与官方格式完全一致():
    """每样本必须含 README.txt + 文案txt + image_1.jpg —— 不允许缺项。"""
    for d in SAMPLE_DIRS:
        files = _files_in(d)
        missing = [f for f in REQUIRED if f not in files]
        assert not missing, f"{d} 缺文件：{missing}（官方格式硬要求，不能靠'说明一下'蒙混）"


def test_样本编号连续且无重复():
    nums = [int(re.match(r"sample_(\d+)$", d).group(1)) for d in SAMPLE_DIRS]
    assert nums == sorted(nums), "样本编号不是升序"
    assert len(set(nums)) == len(nums), "样本编号有重复（会导致覆盖）"


def test_顶层有总说明():
    roots = {n.split("/")[0] for n in NAMES if n.endswith(".txt") and "/" not in n}
    assert any("README" in r for r in roots), "顶层缺少总说明 README"


def test_每个样本的README覆盖官方要求的四个字段():
    """数据来源平台 / 样本类型 / 是否伪造 / 伪造方式 —— 缺一项评委没法核。"""
    with zipfile.ZipFile(ZIP) as z:
        for d in SAMPLE_DIRS:
            blob = z.read(f"{d}/README.txt").decode("utf-8", errors="replace")
            for key in ("数据来源平台", "样本类型", "是否为伪造样本", "伪造方式"):
                assert key in blob, f"{d}/README.txt 缺少字段「{key}」"


def test_每个样本的文案文件都有实际内容():
    """纯视觉样本写的是'本样本无配套文案'的显式声明，不能是个空文件。"""
    with zipfile.ZipFile(ZIP) as z:
        for d in SAMPLE_DIRS:
            body = z.read(f"{d}/种草正文.txt").decode("utf-8", errors="replace").strip()
            assert len(body) >= 10, f"{d}/种草正文.txt 内容过短，等于空壳（格式齐但内容空）"

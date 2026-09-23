# -*- coding: utf-8 -*-
"""下载 CLIP-ViT-L/14 作为「整图 AI 生成」鉴伪的冻结特征骨干。

用法: python tools/download_clip.py

为什么换它（forensics-scout 调研结论，2026-09-23）：
    我们的旧 AIGC 检测头是 MobileNetV3（ImageNet 预训练）。而 2023–2025 所有
    该领域 SOTA（UnivFD / NPR / FatFormer / C2P-CLIP / SAFE / Effort）的共同形态
    都是「冻结的 CLIP 特征 + 轻量头」：
      · UnivFD 当年就是靠「冻结 CLIP + 一个线性探针」打赢全监督 ResNet
      · Effort (ICML'25) 把可训参数压到 0.19M
    关键：CLIP 是 4 亿图文对预训练的表征，对"生成器指纹"比 ImageNet 特征敏感得多。
    论文说 CLIP 系"过时"是针对**跨生成器泛化**；而我们是**单一垂直域 + 本域适配**，
    恰好落在 CLIP 系最擅长的区间。

体积与许可：openai/clip-vit-large-patch14，Apache-2.0，权重约 1.7GB（fp32）。
CPU 可行性：单图特征提取约 1~3 秒（纯前向，无解码），比 4B VLM 快两个数量级。

网络：huggingface.co 被屏蔽，走镜像 hf-mirror.com（HF_ENDPOINT）。
已知坑：Windows 下 HF 的 snapshot 是失效 symlink，会读到 0 字节 →
        所以这里用 hf_hub_download(local_dir=...) 下成**真实文件**，不用缓存软链。
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "models" / "clip-vit-large-patch14"
REPO_ID = "openai/clip-vit-large-patch14"
FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def setup_env():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")


def main():
    setup_env()
    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        target = DEST / fn
        if target.exists() and target.stat().st_size > 0:
            print(f"[已有] {fn}  {target.stat().st_size / 2**20:.1f}MB", flush=True)
            continue
        print(f"\n=== 下载 {fn} ===", flush=True)
        ok = False
        for attempt in range(1, 9):
            try:
                t0 = time.time()
                p = hf_hub_download(repo_id=REPO_ID, filename=fn, local_dir=str(DEST))
                sz = Path(p).stat().st_size if Path(p).exists() else -1
                print(f"[成功] {fn}  {sz / 2**20:.1f}MB  用时 {time.time() - t0:.0f}s", flush=True)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"[失败 {attempt}] {fn}: {type(e).__name__} {str(e)[:150]}", flush=True)
                if attempt < 8:
                    time.sleep(min(2 ** (attempt - 1), 20))
        if not ok:
            print(f"[跳过] {fn}（非必需文件可容忍缺失）", flush=True)

    print("\n=== 结果 ===")
    need = ["config.json", "preprocessor_config.json", "model.safetensors"]
    for fn in need:
        f = DEST / fn
        print(("OK  " if f.exists() and f.stat().st_size > 0 else "缺  ") + fn)
    print(f"\nCLIP 目录: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""下载「能看懂美妆图」的开源视觉语言大模型（VLM）—— Qwen3-VL-4B-Instruct。

用法: python tools/download_vlm.py

为什么选它（调研结论，2026-09-23）：
    · 唯一同时满足「中文强 + 官方自带 GGUF 量化 + Apache-2.0 可商用 + 纯 CPU 能跑」的选项
    · 4B 参数、Q4_K_M 量化后约 2.5GB；官方 mmproj 负责把图像编码进语言模型
    · 本机没有独立显卡（只有 Intel 核显），所以必须走 GGUF + llama.cpp 的 CPU 路线，
      不能用 transformers(torch) 那条路（实测 0.5~2 tok/s，慢到不可用）

运行环境（本机已具备，无需额外安装）：
    · Ollama 0.32.6 已装在 C:/Users/Lenovo/AppData/Local/Programs/Ollama
    · 它内部自带完整 llama.cpp：lib/ollama/llama-server.exe + libmtmd.dll（多模态）
      → 不需要去 GitHub 下 llama.cpp，直接复用现成的

网络路由：huggingface.co 被屏蔽，必须走镜像 hf-mirror.com（HF_ENDPOINT）。
只下推理必需的两个文件：主模型 Q4_K_M + 视觉投影 mmproj。已存在则跳过，可反复跑。
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "models" / "qwen3vl-4b"
REPO_ID = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
FILES = [
    "Qwen3VL-4B-Instruct-Q4_K_M.gguf",   # 主模型（语言+视觉塔），约 2.5GB
    "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",  # 视觉投影层，把图像接进语言模型
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
            print(f"[已有] {fn}  {target.stat().st_size / 2**30:.2f}GB", flush=True)
            continue
        print(f"\n=== 下载 {fn} ===", flush=True)
        for attempt in range(1, 9):
            try:
                t0 = time.time()
                p = hf_hub_download(repo_id=REPO_ID, filename=fn, local_dir=str(DEST))
                sz = Path(p).stat().st_size if Path(p).exists() else -1
                print(f"[成功] {fn}  {sz / 2**30:.2f}GB  用时 {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[失败 {attempt}] {fn}: {type(e).__name__} {str(e)[:150]}", flush=True)
                if attempt < 8:
                    time.sleep(min(2 ** (attempt - 1), 20))
        else:
            print(f"[放弃] {fn} 多次失败", flush=True)

    print("\n=== 结果 ===")
    for fn in FILES:
        f = DEST / fn
        print(("OK  " if f.exists() and f.stat().st_size > 0 else "缺  ") + fn)
    print(f"\n模型目录: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

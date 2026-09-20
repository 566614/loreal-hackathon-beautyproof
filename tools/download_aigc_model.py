# -*- coding: utf-8 -*-
"""下载 AIGC 检测模型（Organika/sdxl-detector）——只下推理必需文件。

用法: python tools/download_aigc_model.py

为什么这么写:
    这个仓库共 17 个文件，但**推理只需要根目录 3 个**：
        config.json / preprocessor_config.json / model.safetensors
    其余 14 个是训练检查点（checkpoint-801/optimizer.pt 等）和 onnx 导出，
    又大又用不上，之前几轮整包下载就是卡在这些无用大文件上反复超时。

    所以这里逐文件下载到 models/sdxl-detector/：
      - 已存在且非空的文件自动跳过 → 多跑几轮能累积凑齐
      - 每个文件独立重试 → 一个文件失败不会清掉其他文件的进度
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_ID = "Organika/sdxl-detector"
LOCAL_DIR = REPO / "models" / "sdxl-detector"
ESSENTIAL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
]
MAX_RETRY = 8


def setup_env():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def download_one(filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=MODEL_ID,
        filename=filename,
        local_dir=str(LOCAL_DIR),
    )


def main():
    setup_env()
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    for fn in ESSENTIAL_FILES:
        dest = LOCAL_DIR / fn
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[已有] {fn}  {dest.stat().st_size} bytes，跳过")
            continue
        for attempt in range(1, MAX_RETRY + 1):
            try:
                print(f"[下载 {attempt}/{MAX_RETRY}] {fn} ...", flush=True)
                path = download_one(fn)
                size = Path(path).stat().st_size if Path(path).exists() else -1
                print(f"[成功] {fn}  {size} bytes -> {path}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[失败] {fn}: {type(e).__name__} {str(e)[:160]}", flush=True)
                if attempt < MAX_RETRY:
                    wait = min(2 ** (attempt - 1), 20)
                    print(f"[等待 {wait}s 重试]", flush=True)
                    time.sleep(wait)

    print("\n=== 结果 ===")
    missing = []
    for fn in ESSENTIAL_FILES:
        p = LOCAL_DIR / fn
        if p.exists() and p.stat().st_size > 0:
            print(f"  OK   {fn}  {p.stat().st_size} bytes")
        else:
            print(f"  缺   {fn}")
            missing.append(fn)
    if missing:
        print(f"\n还缺 {len(missing)} 个文件，再跑一次本脚本即可续下（已下的不会重下）。")
        return 1
    print(f"\n模型就绪: {LOCAL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

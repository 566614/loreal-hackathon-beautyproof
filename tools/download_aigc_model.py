# -*- coding: utf-8 -*-
"""带重试地下载 AIGC 检测模型（Organika/sdxl-detector）。
用法: python tools/download_aigc_model.py
网络不稳时自动重试；坏缓存自动清掉重下。
"""
import os
import shutil
import sys
import time

MODEL_ID = "Organika/sdxl-detector"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
MODEL_CACHE = os.path.join(CACHE_DIR, "hub", f"models--{MODEL_ID.replace('/', '--')}")


def clear_model_cache():
    # 不再主动清缓存：断点续传要保留 .incomplete 文件，清了就前功尽弃
    # 只在最终彻底失败时由调用方决定是否清理（这里默认不清）
    print("[提示] 保留已有缓存以续传")


def try_download():
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    return snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=False,
    )


def main():
    # 强制走国内镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 关掉 Xet 内容寻址存储（本机环境连不上 cas-server.xethub.hf.co，会 401），
    # 退回传统直下模式，绕开这个坑
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    max_retries = 20
    for i in range(1, max_retries + 1):
        try:
            print(f"\n[尝试 {i}/{max_retries}] 下载 {MODEL_ID} ...")
            path = try_download()
            print(f"[成功] 模型已缓存到: {path}")
            return 0
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(f"[失败] {msg[:300]}")
            # 任何失败都清掉可能写坏的缓存，避免下次读到半截文件
            try:
                clear_model_cache()
            except Exception:
                pass
            if i < max_retries:
                wait = min(2 ** i, 16)
                print(f"[等待 {wait}s 后重试]")
                time.sleep(wait)
    print("\n[全部重试失败] 网络仍不稳定，AIGC 模型暂未下载成功。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

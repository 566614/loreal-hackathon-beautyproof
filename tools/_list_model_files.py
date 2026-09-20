# -*- coding: utf-8 -*-
"""列出 AIGC 模型仓库的文件结构（决定从哪里加载模型）。"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"

from huggingface_hub import list_repo_files  # noqa: E402

MODEL_ID = "Organika/sdxl-detector"

try:
    files = list_repo_files(MODEL_ID)
    print(f"共 {len(files)} 个文件：")
    for f in files:
        print("  -", f)
except Exception as e:  # noqa: BLE001
    print("LIST_FAIL:", type(e).__name__, str(e)[:300])

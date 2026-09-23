# -*- coding: utf-8 -*-
"""下载「AI 生成图检测」模型（替换旧的零区分度 sdxl-detector）。

用法: python tools/download_aigen_models.py

下载两个候选到 models/airealnet 与 models/capcheck：
  - XenArcAI/AIRealNet   : SwinV2-Tiny，MIT，AI vs Real，轻量 CPU 可跑
  - capcheck/ai-human-generated-image-detection : ViT-B，Apache2.0，human/AI-generated

路由：HF 镜像 hf-mirror.com + 本地代理 7897（HF 官方域名走代理 SSL 报错，镜像可用）。

只下推理必需文件：config.json / preprocessor_config.json / 权重（model.safetensors 或 pytorch_model.bin）。
已存在的文件自动跳过，可反复跑续下。
"""
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = {
    "airealnet": "XenArcAI/AIRealNet",
    "capcheck": "capcheck/ai-human-generated-image-detection",
}
ESSENTIAL = ["config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"]
MAX_RETRY = 8


def setup_env():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")


def download_one(repo_id, filename, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))


def main():
    setup_env()
    from huggingface_hub import hf_hub_download  # noqa: F401

    for local_name, repo_id in MODELS.items():
        d = REPO / "models" / local_name
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {local_name}  ({repo_id}) ===", flush=True)
        # 配置文件
        for fn in ["config.json", "preprocessor_config.json"]:
            t = d / fn
            if t.exists() and t.stat().st_size > 0:
                print(f"  [已有] {fn}", flush=True)
                continue
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    download_one(repo_id, fn, d)
                    print(f"  [成功] {fn}", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  [失败 {attempt}] {fn}: {type(e).__name__} {str(e)[:120]}", flush=True)
                    if attempt < MAX_RETRY:
                        time.sleep(min(2 ** (attempt - 1), 20))
        # 权重：safetensors 优先，没有再试 pytorch_model.bin
        weight_done = False
        for w in ["model.safetensors", "pytorch_model.bin"]:
            t = d / w
            if t.exists() and t.stat().st_size > 0:
                print(f"  [已有] {w}  {t.stat().st_size} bytes", flush=True)
                weight_done = True
                break
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    p = download_one(repo_id, w, d)
                    sz = Path(p).stat().st_size if Path(p).exists() else -1
                    print(f"  [成功] {w}  {sz} bytes", flush=True)
                    weight_done = True
                    break
                except Exception as e:  # noqa: BLE001
                    # 这个权重文件不存在（如只有 safetensors 没有 bin）就试下一个
                    print(f"  [失败 {attempt}] {w}: {type(e).__name__} {str(e)[:120]}", flush=True)
                    if attempt < MAX_RETRY:
                        time.sleep(min(2 ** (attempt - 1), 20))
            if weight_done:
                break

    print("\n=== 结果 ===")
    for local_name, repo_id in MODELS.items():
        d = REPO / "models" / local_name
        have = [f.name for f in d.glob("*") if f.is_file()] if d.exists() else []
        ok = ("model.safetensors" in have) or ("pytorch_model.bin" in have)
        print(f"  {'OK ' if ok else '缺 '} {local_name}: {have}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

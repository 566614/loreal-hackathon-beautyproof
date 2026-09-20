# -*- coding: utf-8 -*-
"""探测新网络环境：HuggingFace 直连 / 代理 7897 分别能不能通。"""
import json
import os
import time
import urllib.request

URL = "https://huggingface.co/Organika/sdxl-detector/resolve/main/config.json"
MIRROR = "https://hf-mirror.com/Organika/sdxl-detector/resolve/main/config.json"


def probe(label, url, proxy=None):
    opener = urllib.request.build_opener()
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    t0 = time.time()
    try:
        with opener.open(url, timeout=15) as r:
            body = r.read(200)
            dt = time.time() - t0
            print(f"[{label}] HTTP {r.status} in {dt:.2f}s  head={body[:60]!r}")
            return r.status == 200
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        print(f"[{label}] FAIL in {dt:.2f}s -> {type(e).__name__}: {str(e)[:120]}")
        return False


if __name__ == "__main__":
    print("=== 新网络探测 ===")
    probe("直连 huggingface", URL)
    probe("直连 hf-mirror", MIRROR)
    probe("代理7897 huggingface", URL, "http://127.0.0.1:7897")
    probe("代理7897 hf-mirror", MIRROR, "http://127.0.0.1:7897")

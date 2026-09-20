# -*- coding: utf-8 -*-
"""诊断 AIGC pipeline 为什么抛 TypeError —— 打印完整 traceback。"""
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "models" / "sdxl-detector"
IMG = REPO / "data" / "raw" / "base.png"

from transformers import pipeline  # noqa: E402

print("model dir:", LOCAL)
print("files:", [p.name for p in LOCAL.iterdir()])

try:
    print("\n--- 尝试构造 pipeline ---")
    det = pipeline("image-classification", model=str(LOCAL), local_files_only=True)
    print("pipeline 构造成功:", type(det))
    print("model class:", type(det.model).__name__)
    print("\n--- 尝试推理 ---")
    res = det(str(IMG))
    print("结果:", res)
except Exception:  # noqa: BLE001
    print("\n!!! 异常完整信息 !!!")
    traceback.print_exc()

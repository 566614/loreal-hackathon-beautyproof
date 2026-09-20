#!/usr/bin/env bash
# 项目专用启动器 —— 用「装好依赖的隔离 Python」跑任意脚本
#
# 用法（在仓库根目录）：
#   ./run.sh tools/batch_run.py
#   ./run.sh tools/download_aigc_model.py
#   ./run.sh tools/make_dataset.py
#
# 为什么需要它：
#   你本机 `python` 指向系统自带的 3.11，里面没装 PIL / paddleocr /
#   transformers / huggingface_hub，直接跑项目脚本会报 ModuleNotFoundError。
#   项目依赖都装在隔离环境里，所以所有脚本统一用它启动，一次解决全部依赖问题。
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
"$PY" "$@"

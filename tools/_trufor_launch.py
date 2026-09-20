# -*- coding: utf-8 -*-
"""TruFor 官方 test.py 的启动包装器（内部用，不用手动跑它）

为什么需要它（两件事）：

1) 新版 torch（2.14）默认 torch.load(weights_only=True)，只放行"安全"对象。
   TruFor 是 2023 年的官方权重，含 numpy.core.multiarray.scalar 这类旧类型，
   会被拦下报错：Unsupported global: GLOBAL numpy.core.multiarray.scalar
   → 这里把 torch.load 的默认参数放宽成 weights_only=False（权重来自官方可信站点）。

2) 官方 test.py 拼输出路径时只处理了 Linux 的前导斜杠 '/'，
   Windows 下会拼出 'C:\\xxx.png.npz'（C 盘根目录）→ PermissionError，
   推理正常跑完但结果一个都存不下来。
   → 这里拦下 np.savez，把结果强制重定向回 -out 指定的目录，并把
     'xxx.png.npz' 规范成 'xxx.npz'（方便按原图文件名 stem 对回）。

全程不改官方代码，只是在外面"垫一层"。

用法（由 trufor_tool.py 自动调用）：
    python _trufor_launch.py <test.py 路径> <test.py 的参数...>
"""
import os
import runpy
import sys

import numpy as np
import torch

# ---------------------------------------------------------------- 1) 放宽 torch 加载限制
_orig_load = torch.load


def _patched_load(*args, **kwargs):
    """放宽 torch 的安全限制：默认 weights_only=False（仅对官方可信权重）"""
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load


# ---------------------------------------------------------------- 2) 重定向输出路径
def _parse_out_dir(argv):
    """从 test.py 的参数里读出 -out 指定的输出目录"""
    for i, a in enumerate(argv):
        if a in ("-out", "--out") and i + 1 < len(argv):
            return argv[i + 1]
    return None


_out_dir = _parse_out_dir(sys.argv[1:])
_orig_savez = np.savez


def _fix_filename(file):
    """把可能被拼歪的输出路径掰回 -out 目录"""
    try:
        name = os.path.basename(str(file))
        # 'clean_01.png.npz' -> 'clean_01.npz'（去掉原图扩展名，方便按 stem 对回）
        if name.lower().endswith(".png.npz"):
            name = name[: -len(".png.npz")] + ".npz"
        elif name.lower().endswith(".jpg.npz"):
            name = name[: -len(".jpg.npz")] + ".npz"
        if _out_dir:
            os.makedirs(_out_dir, exist_ok=True)
            return os.path.join(_out_dir, name)
        return name
    except Exception:  # noqa: BLE001
        return file


def _patched_savez(file, *args, **kwargs):
    return _orig_savez(_fix_filename(file), *args, **kwargs)


np.savez = _patched_savez

# ---------------------------------------------------------------- 执行官方 test.py
if len(sys.argv) < 2:
    print("用法: python _trufor_launch.py <test.py 路径> [参数...]")
    sys.exit(1)

target = sys.argv[1]
sys.argv = sys.argv[1:]  # 让 test.py 像被直接执行一样解析参数
runpy.run_path(target, run_name="__main__")

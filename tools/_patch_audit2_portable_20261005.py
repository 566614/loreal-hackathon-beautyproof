# -*- coding: utf-8 -*-
"""2026-10-05 · 全面复检 · 第二批：可复现性（去掉写死的本机绝对路径）

问题：7 处把「阮这台机器」的绝对路径写进了代码。官方评分明确看「方案可复现」，
评委换机 / 换账号复现时会直接报错，属于自伤。

改法统一为三级解析：**环境变量 > 标准 PATH > 本机兜底（仅当该路径确实存在时）**。
关键点：本机运行结果**完全不变**（兜底路径就在原处，且 sys.executable 正是当前 venv），
所以这是纯可移植性改进，不是行为变更。

逐处：
1. tools/batch_run.py:31      PY  → sys.executable（用谁跑本脚本就用谁来跑子脚本，最正确）
2. tools/trufor_tool.py:38    PY  → 同上
3. tools/push_via_api.py:26-27 GH/GIT → 环境变量/PATH/本机兜底三选一并给清晰报错
4. tools/make_template.py:7   OUT → 由 __file__ 推导仓库内 data/ 目录
5. tools/video_tool.py:95     报错信息里的 pip 路径 → sys.executable 动态生成
6. tools/_validate_crossgen.py:30 文档字符串里的命令 → 改为 ./run.sh 通用写法
7. tools/download_vlm.py:13   文档字符串里的用户名路径 → 泛化

幂等：逐处断言命中 1 次。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATCHES = [
    # ---------- 1. batch_run.py ----------
    ("tools/batch_run.py",
     'PY = "C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"',
     '''# 子脚本用「跑本脚本的那个解释器」执行，避免写死某台机器的 python 路径。
# 正常用法是 ./run.sh tools/batch_run.py，此时 sys.executable 就是项目 venv 的 python。
PY = os.environ.get("BEAUTYPROOF_PYTHON") or sys.executable'''),

    # ---------- 2. trufor_tool.py ----------
    ("tools/trufor_tool.py",
     'PY = "C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"',
     '''# 复用当前解释器（./run.sh 已指向装了全部依赖的隔离 Python），不写死本机路径。
PY = os.environ.get("BEAUTYPROOF_PYTHON") or sys.executable'''),

    # ---------- 3. push_via_api.py ----------
    ("tools/push_via_api.py",
     '''GH = r"C:/Users/Lenovo/.workbuddy/binaries/gh/bin/gh.exe"
GIT = r"C:/Program Files/Git/cmd/git.exe"''',
     '''def _find_exe(env_var, name, *fallbacks, hint=""):
    """三级定位可执行文件：环境变量 > 标准 PATH > 本机兜底路径（存在才用）。

    写死绝对路径会让脚本换机即废；这里既保证本机行为不变，又让换机的人有明确出路。
    """
    p = os.environ.get(env_var)
    if p and os.path.isfile(p):
        return p
    found = shutil.which(name)
    if found:
        return found
    for fb in fallbacks:
        if os.path.isfile(fb):
            return fb
    raise SystemExit(
        f"找不到 {name} 可执行文件。请任选一种方式配置：\\n"
        f"  1) 设置环境变量 {env_var}=<{name} 的完整路径>\\n"
        f"  2) 把 {name} 加入 PATH 后重试\\n"
        f"{('  ' + hint) if hint else ''}"
    )


GH = _find_exe("BEAUTYPROOF_GH", "gh",
               hint="（GitHub CLI：https://cli.github.com）")
GIT = _find_exe("BEAUTYPROOF_GIT", "git", r"C:/Program Files/Git/cmd/git.exe",
                hint="（Git for Windows 或系统自带 git）")'''),

    # ---------- 4. make_template.py ----------
    ("tools/make_template.py",
     'OUT = r"C:/Users/Lenovo/WorkBuddy/黑客松/loreal-hackathon-beautyproof/data/数据登记表_模板.xlsx"',
     '''# 输出到仓库内 data/，由本文件位置推导，换机 / 换 clone 路径都不会写错地方
OUT = Path(__file__).resolve().parent.parent / "data" / "数据登记表_模板.xlsx"'''),
    ("tools/make_template.py",
     "from openpyxl import Workbook",
     "from pathlib import Path\n\nfrom openpyxl import Workbook"),

    # ---------- 5. video_tool.py ----------
    ("tools/video_tool.py",
     '''        raise RuntimeError(
            "未检测到 opencv（cv2）。请先用项目 venv 的 pip 安装：\\n"
            "  C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/pip.exe "
            "install opencv-python-headless\\n"
            "（绝不要用系统 pip 或全局安装）"
        )''',
     '''        raise RuntimeError(
            "未检测到 opencv（cv2）。请用项目隔离环境装：\\n"
            "  ./run.sh -m pip install opencv-python-headless\\n"
            f"（等价于：\\"{sys.executable}\\" -m pip install opencv-python-headless）\\n"
            "（绝不要用系统 pip 或全局安装）"
        )'''),

    # ---------- 6. _validate_crossgen.py 文档字符串 ----------
    ("tools/_validate_crossgen.py",
     "    C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe tools/_validate_crossgen.py",
     "    ./run.sh tools/_validate_crossgen.py"),

    # ---------- 7. download_vlm.py 文档字符串 ----------
    ("tools/download_vlm.py",
     "    · Ollama 0.32.6 已装在 C:/Users/Lenovo/AppData/Local/Programs/Ollama",
     "    · Ollama 装在默认位置（C:/Users/<你的用户名>/AppData/Local/Programs/Ollama）"),
]


def main():
    by_file = {}
    for rel, old, new in PATCHES:
        by_file.setdefault(rel, []).append((old, new))

    for rel, pairs in by_file.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        for old, new in pairs:
            if new in text and old not in text:
                print(f"[skip] {rel} 已应用")
                continue
            n = text.count(old)
            assert n == 1, f"{rel} 命中 {n} 次：{old[:56]}"
            text = text.replace(old, new)
            print(f"[ok]   {rel}  ← {old[:52]}")
        p.write_text(text, encoding="utf-8")

    # push_via_api.py 新增了 shutil 依赖
    pa = ROOT / "tools" / "push_via_api.py"
    t = pa.read_text(encoding="utf-8")
    if "import shutil" not in t:
        t = t.replace("import os\nimport subprocess", "import os\nimport shutil\nimport subprocess", 1)
        pa.write_text(t, encoding="utf-8")
        print("[ok]   tools/push_via_api.py  补 import shutil")

    # batch_run.py 需要 os
    br = ROOT / "tools" / "batch_run.py"
    t = br.read_text(encoding="utf-8")
    if "\nimport os\n" not in t:
        t = t.replace("import json\nimport subprocess", "import json\nimport os\nimport subprocess", 1)
        br.write_text(t, encoding="utf-8")
        print("[ok]   tools/batch_run.py  补 import os")

    # 复核：项目代码里不再残留指向本机用户名的绝对路径
    bad = []
    for rel in ("tools", "web", "tests"):
        for py in (ROOT / rel).rglob("*.py"):
            s = py.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(s.splitlines(), 1):
                if "C:/Users/Lenovo" in line or "C:\\\\Users\\\\Lenovo" in line:
                    bad.append(f"{py.relative_to(ROOT)}:{i}")
    if bad:
        print(f"[warn] 仍有本机路径残留（需人工确认是否可接受）：{bad}")
    else:
        print("[复核] tools/web/tests 下已无指向本机用户名的绝对路径")


if __name__ == "__main__":
    main()

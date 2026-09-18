# -*- coding: utf-8 -*-
"""
C2PA 工具 —— 查图片有没有「出生证明」（内容凭证），输出统一格式的证据 JSON

用法：
    python tools/c2pa_tool.py <图片路径>

大白话：
    C2PA 是相机或修图软件在图片里留下的一张「出生证明」，
    记录这张图是谁拍的、用什么软件改过。有证明不代表没被改，
    但能看出改动历史；没有证明也不代表是假的，绝大多数网图都没有。

能证明什么：
    这张图有没有带官方的内容凭证，凭证里写了什么编辑历史。

不能证明什么（铁律）：
    没有凭证 ≠ 是假图。目前网上 99% 的图都没凭证。
"""
import json
import subprocess
import sys
from pathlib import Path

C2PA_EXE = r"C:\Users\Lenovo\AppData\Local\Programs\c2patool\c2patool\c2patool.exe"

NO_CLAIM_HINTS = [
    "no claim", "no manifest", "no embedded", "not found",
    "noclaim", "error", "unable", "no c2pa",
]


def run_c2pa(image_path):
    """调已装好的 c2patool，读出图片里的凭证信息"""
    try:
        result = subprocess.run(
            [C2PA_EXE, str(image_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "", "找不到 c2patool，请检查脚本里的 C2PA_EXE 路径", 127
    except OSError as e:
        return "", f"运行 c2patool 失败：{e}", 127

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return stdout, stderr, result.returncode


def judge(stdout, stderr, returncode):
    """判断有没有凭证。返回 (状态, 人类可读的一句话)"""
    if "找不到 c2patool" in stderr or "运行 c2patool" in stderr:
        return "error", stderr

    combined = (stdout + stderr).lower()

    if returncode != 0:
        return "missing", "c2patool 没能读出凭证（绝大多数网图都没有凭证，属正常）"

    if not stdout:
        return "missing", "这张图没有 C2PA 内容凭证（绝大多数网图都没有，属正常）"

    for hint in NO_CLAIM_HINTS:
        if hint in combined:
            return "missing", "这张图没有 C2PA 内容凭证（绝大多数网图都没有，属正常）"

    return "present", "这张图带有 C2PA 内容凭证，需人工查看凭证内容确认编辑历史"


def build_evidence(image_path, status, note, raw_output):
    return {
        "tool": "c2pa",
        "source_asset_id": Path(image_path).name,
        "observed": note,
        "cannot_prove": "没有凭证不等于图片是假的；有凭证也不等于没被恶意编辑过",
        "evidence": [
            {"c2pa_status": status, "detail": note},
        ],
        "raw_output": raw_output[:2000],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/c2pa_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    stdout, stderr, returncode = run_c2pa(image_path)
    status, note = judge(stdout, stderr, returncode)
    report = build_evidence(image_path, status, note, stdout or stderr)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"c2pa_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

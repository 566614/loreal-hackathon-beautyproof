# -*- coding: utf-8 -*-
"""
流水线编排 —— 一张图进来，一份完整鉴定结论出去

用法：
    python tools/pipeline.py <图片路径>              # 单张，跑全套 6 个工具
    python tools/pipeline.py <图片路径> --fast       # 跳过慢的深度学习模型，秒出结论
    python tools/pipeline.py <图1> <图2> --json-only # 只写 JSON，不打印详细过程

大白话：
    以前要手动挨个敲六个工具、再敲规则引擎，容易漏、也容易弄错顺序。
    这个脚本把整套流程串成"一步"：送图进去，出一份可以直接给人看的结论。

产出：
    outputs/analysis_<图片名>.json   完整结果（Web 界面和报告都读它）
    reports/report_<图片名>.md       人能读的 Markdown 鉴定报告
"""
import base64
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
PY = sys.executable

# 六个工具：(证据里的名字, 脚本名, 是否默认开启)
# 深度学习那两个（aigc / trufor）要加载几百 MB 的模型，慢，所以支持 --fast 跳过
TOOL_LIST = [
    ("hash", "hash_tool.py", True),
    ("c2pa", "c2pa_tool.py", True),
    ("ela", "ela_tool.py", True),
    ("ocr", "ocr_tool.py", True),
    ("aigc", "aigc_tool.py", True),
    ("trufor", "trufor_tool.py", True),
]
SLOW_TOOLS = {"aigc", "trufor"}

# 每个工具的"人话说明" —— 报告页和网页都用它，避免各写一套口径
TOOL_META = {
    "hash": {"label": "文件指纹", "job": "给这张图算一个独一无二的身份证号，看它是不是跟已知原图一字不差"},
    "c2pa": {"label": "内容凭证", "job": "查图片里有没有官方的「出生证」，记录它是谁拍的、被哪些软件改过"},
    "ela": {"label": "压缩痕迹", "job": "把图压一遍再还原，看哪块区域的压缩反应跟周围不一样"},
    "ocr": {"label": "文字识别", "job": "把图上所有字抄下来，跟品牌方给的标准文案逐字比对"},
    "aigc": {"label": "AI 生成检测", "job": "判断这张图是不是 AI 画的（本模型在本素材上零区分度，仅作参考）"},
    "trufor": {"label": "篡改痕迹检测", "job": "用 CVPR 2023 的取证模型查整图有没有被人工动过，并定位可疑区域"},
}


# ---------------------------------------------------------------- 图片 → base64
def to_data_uri(path, max_side=900, quality=85):
    """把图片压成缩略图再转成 data URI —— 这样报告 / 网页不用依赖原图文件，拷走就能看"""
    try:
        from PIL import Image
        img = Image.open(path)
        img = img.convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def image_info(path):
    """读出图片的基本信息（宽高、大小），顺带做个缩略图"""
    info = {
        "name": Path(path).name,
        "stem": Path(path).stem,
        "path": str(Path(path).resolve()),
        "size_bytes": Path(path).stat().st_size if Path(path).exists() else None,
    }
    try:
        from PIL import Image
        with Image.open(path) as im:
            info["width"], info["height"] = im.size
            info["format"] = im.format
    except Exception:  # noqa: BLE001
        info["width"] = info["height"] = None
    info["preview"] = to_data_uri(path)
    return info


# ---------------------------------------------------------------- 跑工具
def run_tool(script, image_path, timeout=900):
    """跑单个工具脚本，把它的输出 JSON 读回来"""
    proc = subprocess.run(
        [PY, str(TOOLS / script), str(image_path)],
        cwd=str(REPO), capture_output=True, text=True,
        encoding="utf-8", errors="ignore", timeout=timeout,
    )
    return proc


def collect_evidence(stem):
    """收集 outputs/ 下所有 <tool>_<stem>.json —— 与规则引擎口径完全一致"""
    sys.path.insert(0, str(TOOLS))
    import rule_engine  # noqa: E402
    return rule_engine.load_evidence(REPO, stem)


def judge(evidence):
    sys.path.insert(0, str(TOOLS))
    import rule_engine  # noqa: E402
    return rule_engine.judge(evidence)


# ---------------------------------------------------------------- 主流程
def analyze(image_path, fast=False, skip=(), verbose=True, timeout=900, on_progress=None):
    """对一张图跑完整流水线，返回结构化结果

    on_progress: 可选回调 fn(tool, 第几个, 共几个, 状态) —— Web 界面靠它显示实时进度
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"找不到这张图: {image_path}")
    stem = image_path.stem

    todo = [t for t in TOOL_LIST
            if t[0] not in skip and t[2] and not (fast and t[0] in SLOW_TOOLS)]

    t0 = time.time()
    ran, failed = [], []
    for idx, (tool, script, _) in enumerate(todo, 1):
        if verbose:
            print(f"  · {tool:7s} ...", end="", flush=True)
        if on_progress:
            on_progress(tool, idx, len(todo), "running")
        t = time.time()
        try:
            proc = run_tool(script, image_path, timeout=timeout)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
            proc = None
        if ok:
            ran.append(tool)
            if verbose:
                print(f" 完成 ({time.time() - t:.1f}s)")
        else:
            failed.append(tool)
            if verbose:
                print(f" 跳过（{'超时' if proc is None else '工具报错'}）")
        if on_progress:
            on_progress(tool, idx, len(todo), "done" if ok else "failed")

    evidence = collect_evidence(stem)
    risk_level, reasons = judge(evidence)

    # TruFor 的定位图（如果画出来了就一起带上）
    tru = evidence.get("trufor", {}).get("evidence", [{}])
    tru0 = tru[0] if tru else {}
    visuals = {}
    for key in ("heatmap", "overlay"):
        name = tru0.get(key)
        if name:
            p = REPO / "outputs" / name
            if p.exists():
                visuals[key] = to_data_uri(p, max_side=1100, quality=90)

    result = {
        "schema": "beautyproof/analysis@1",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "image": image_info(image_path),
        "verdict": {
            "risk_level": risk_level,
            "reasons": reasons,
            "tools_used": sorted(evidence.keys()),
            "tools_ran": ran,
            "tools_failed": failed,
        },
        "evidence": evidence,
        "visuals": visuals,
    }

    out = REPO / "outputs" / f"analysis_{stem}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["_output_file"] = str(out)
    return result


def write_markdown(result):
    """生成人能读的 Markdown 报告（复用现成的 report_generator.py）"""
    stem = result["image"]["stem"]
    try:
        run_tool("report_generator.py", stem, timeout=120)
        p = REPO / "reports" / f"report_{stem}.md"
        return str(p) if p.exists() else None
    except Exception as e:  # noqa: BLE001
        return f"（Markdown 报告生成失败：{type(e).__name__}：{e}）"


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        return 1
    fast = "--fast" in args
    json_only = "--json-only" in args
    paths = [a for a in args if not a.startswith("--")]

    for p in paths:
        print(f"\n=== 鉴定：{Path(p).name} ===")
        res = analyze(p, fast=fast, verbose=not json_only)
        v = res["verdict"]
        if json_only:
            print(json.dumps({"image": res["image"]["name"], "risk_level": v["risk_level"]},
                             ensure_ascii=False))
        else:
            print(f"\n  结论：{v['risk_level']}")
            for r in v["reasons"]:
                print(f"    - {r}")
            print(f"  用时 {res['elapsed_seconds']}s")
            print(f"  完整结果：{res['_output_file']}")
            md = write_markdown(res)
            if md:
                print(f"  文字报告：{md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

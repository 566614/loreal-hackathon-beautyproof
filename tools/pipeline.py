# -*- coding: utf-8 -*-
"""
流水线编排 —— 一张图进来，一份完整鉴定结论出去

用法：
    python tools/pipeline.py <图片路径>              # 单张，跑全套 6 个工具
    python tools/pipeline.py <图片路径> --fast       # 跳过慢的深度学习模型，秒出结论
    python tools/pipeline.py <图1> <图2> --json-only # 只写 JSON，不打印详细过程
    python tools/pipeline.py <图片路径> --pdf    # 额外导出一份 PDF（法务 / 品牌方存档）
    python tools/pipeline.py <图片路径> --llm    # 开启大模型「人话解读」层（需本地 Ollama + Qwen3-VL-4B）
    python tools/pipeline.py <图片路径> --roundtable  # 开启方案B 多 Agent 圆桌交叉复核

    # 带上配套文案一起鉴定 —— 这才是赛题要的「文案 + 图片」双模态
    python tools/pipeline.py <图片路径> --text "原相机实拍，无滤镜，七天美白亲测有效"
    python tools/pipeline.py <图片路径> --text-file 种草文.txt

大白话：
    以前要手动挨个敲六个工具、再敲规则引擎，容易漏、也容易弄错顺序。
    这个脚本把整套流程串成"一步"：送图进去，出一份可以直接给人看的结论。

产出：
    outputs/analysis_<图片名>.json   完整结果（Web 界面和报告都读它）
    reports/report_<图片名>.md       人能读的 Markdown 鉴定报告
    reports/report_<图片名>.pdf      （加 --pdf 时）可直接存档/转发的 PDF 版
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
    "aigc": {"label": "AI 生成检测", "job": "判断这张图是不是整张由 AI 生成的（用的是本域微调模型：拿 10 张真实美妆照 + 20 张即梦 AI 图专门练过，补 TruFor 查不出的「整图 AI 生成」盲区）"},
    "trufor": {"label": "篡改痕迹检测", "job": "用 CVPR 2023 的取证模型查整图有没有被人工动过，并定位可疑区域"},
    # 文案侧（只有传了 --text 才会跑，不影响原来只鉴定图片的用法）
    "text": {"label": "文案体检", "job": "查文案里有没有《广告法》《化妆品监督管理条例》点名的敏感宣称，逐条给出法律出处、命中位置和上下文"},
    "crossmodal": {"label": "图文交叉验证", "job": "拿文案说的去跟图像侧结论对质 —— 文案声称原相机实拍、图却被判成 AI 生成的，这种自相矛盾比任何单侧分数都硬"},
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


def explain_verdict(risk_level, evidence):
    """把分级结果翻译成品牌方 / 法务也看得懂的大白话"""
    sys.path.insert(0, str(TOOLS))
    import rule_engine  # noqa: E402
    return rule_engine.explain(risk_level, evidence)


def run_text_tools(stem, text, verbose=True, on_progress=None, base_idx=0, total=8):
    """跑文案侧两个工具，结果写进 outputs/ —— 规则引擎随后就能和图像侧证据一起读到

    text_tool       只看文字本身：违禁宣称（硬规则，有法律出处）+ 写作特征（弱提示）
    crossmodal_tool 拿文案去跟图像侧的结论对质：说实拍却是 AI 图 → 硬矛盾

    写文件而不是直接返回，是为了和图像侧共用同一套 outputs/<tool>_<stem>.json 约定，
    规则引擎不用为文案侧开特例。
    """
    sys.path.insert(0, str(TOOLS))
    import text_tool  # noqa: E402
    import crossmodal_tool  # noqa: E402

    written = []

    # 1) 文案体检
    if verbose:
        print("  · text    ...", end="", flush=True)
    if on_progress:
        on_progress("text", base_idx + 1, total, "running")
    try:
        rep = text_tool.build_evidence(text, source_id=stem, mode="single")
        (REPO / "outputs" / f"text_{stem}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append("text")
        if verbose:
            print(" 完成")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f" 跳过（{type(e).__name__}）")
    if on_progress:
        on_progress("text", base_idx + 1, total, "done" if "text" in written else "failed")

    # 2) 图文交叉验证（要读图像侧已跑完的证据，所以必须排在图像工具之后）
    if verbose:
        print("  · crossmodal ...", end="", flush=True)
    if on_progress:
        on_progress("crossmodal", base_idx + 2, total, "running")
    try:
        img_ev = collect_evidence(stem)
        cons, negated = crossmodal_tool.check(text, img_ev)
        rep = crossmodal_tool.build_evidence(text, cons, img_ev, stem, negated=negated)
        (REPO / "outputs" / f"crossmodal_{stem}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append("crossmodal")
        if verbose:
            print(" 完成")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f" 跳过（{type(e).__name__}）")
    if on_progress:
        on_progress("crossmodal", base_idx + 2, total, "done" if "crossmodal" in written else "failed")

    return written


# ---------------------------------------------------------------- 主流程
def analyze(image_path, fast=False, skip=(), verbose=True, timeout=900,
            on_progress=None, text=None, llm=False, roundtable=False):
    """对一张图跑完整流水线，返回结构化结果

    on_progress: 可选回调 fn(tool, 第几个, 共几个, 状态) —— Web 界面靠它显示实时进度
    text:        可选的配套文案（种草文 / 评论）。传了才跑文案侧两个工具；
                 不传时行为与以前完全一致，仍是纯图片鉴定。
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"找不到这张图: {image_path}")

    # 先确认它真的是一张能被读出来的图：否则后面六个工具会各自以奇怪的方式失败，
    # 报错信息没法看。这里一次性说清楚。
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            im.verify()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"这不是一张能被识别的图片：{image_path.name}（{type(e).__name__}）")

    stem = image_path.stem
    info = image_info(image_path)  # 提前拿到格式/尺寸，Agent 要用它做判断

    # 清掉这张图上一轮留下的证据文件。
    # 否则两件事会出错：① Agent 以为某工具已经跑过，跳过规则失效；
    # ② 某个工具本次失败了，judge 却拿上次的旧证据凑数，输出看起来正常。
    # 失败就该表现为「没有这项证据」，而不是拿旧的冒充。
    for _tool in [t for t, _, _ in TOOL_LIST] + ["text", "crossmodal"]:
        _f = REPO / "outputs" / f"{_tool}_{stem}.json"
        if _f.exists():
            try:
                _f.unlink()
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- Agent 调度
    # 不再是「六个工具按固定顺序跑完」，而是分三波：每跑完一波，
    # 先问 Agent「根据已查到的东西，下一波哪些值得跑」，再决定跑不跑。
    sys.path.insert(0, str(TOOLS))
    import planner  # noqa: E402

    ctx = {
        "has_text": bool(text and text.strip()),
        "image_format": info.get("format"),
        "image_pixels": (info.get("width") or 0) * (info.get("height") or 0),
    }

    script_of = {tool: script for tool, script, _ in TOOL_LIST}
    enabled = {tool for tool, _, on in TOOL_LIST if on and tool not in skip}

    total = len([x for x in enabled]) + (2 if ctx["has_text"] else 0)
    t0 = time.time()
    ran, failed = [], []
    skipped, decisions = set(), []
    idx = 0

    def _note(tool, state, spent=None):
        if on_progress:
            on_progress(tool, idx, total, state)

    for wave_name, wave_tools in planner.WAVES:
        ev_now = collect_evidence(stem)
        wave_skip, dec = planner.decide_next(ev_now, ctx, done=set(ran))
        decisions.extend(dec)
        skipped |= wave_skip

        for tool in wave_tools:
            if tool not in enabled:
                continue
            idx += 1
            # 快速预览模式：跳过两个深度学习模型（这条也记进决策，理由要写清楚）
            if fast and tool in SLOW_TOOLS:
                skipped.add(tool)
                decisions.append({
                    "tool": tool, "action": "skip", "rule": "F1",
                    "reason": "快速预览模式：跳过深度学习模型换取秒出结果，代价是证据不全",
                })
                if verbose:
                    print(f"  · {tool:7s} ... 跳过（快速预览）")
                _note(tool, "skipped")
                continue
            if tool in wave_skip:
                if verbose:
                    print(f"  · {tool:7s} ... 跳过（Agent 判断：查了没意义）")
                _note(tool, "skipped")
                continue

            if verbose:
                print(f"  · {tool:7s} ...", end="", flush=True)
            _note(tool, "running")
            ts = time.time()
            try:
                proc = run_tool(script_of[tool], image_path, timeout=timeout)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
            if ok:
                ran.append(tool)
                if verbose:
                    print(f" 完成 ({time.time() - ts:.1f}s)")
            else:
                failed.append(tool)
                if verbose:
                    print(" 跳过（工具报错或超时）")
            _note(tool, "done" if ok else "failed")

    # 文案侧（可选）：只有传了 text 才跑，写完 outputs/ 后下面的 collect_evidence 会一起收上来
    text_tools_ran = []
    if text and text.strip():
        idx += 1
        text_tools_ran = run_text_tools(
            stem, text.strip(), verbose=verbose, on_progress=on_progress,
            base_idx=idx - 1, total=total,
        )
        idx += 1
    else:
        for _t in planner.TEXT_TOOLS:
            skipped.add(_t)

    evidence = collect_evidence(stem)
    risk_level, reasons = judge(evidence)
    plain = explain_verdict(risk_level, evidence)

    # Agent 的后半段：判完级之后该干什么（进复核队列 / 核改文案 / 核实来源）
    actions = planner.decide_actions(risk_level, evidence)
    agent_log = planner.build_log(decisions, actions, ran + text_tools_ran, skipped)

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
        "image": info,
        "verdict": {
            "risk_level": risk_level,
            "explain": plain,
            "reasons": reasons,
            "tools_used": sorted(evidence.keys()),
            "tools_ran": ran + text_tools_ran,
            "tools_failed": failed,
        },
        "text_input": (text or "").strip() or None,
        "agent": agent_log,   # 决策过程：跑了什么、跳了什么、为什么
        "evidence": evidence,
        "visuals": visuals,
        "tool_meta": TOOL_META,   # 每个工具的人话说明，网页/离线 Demo 直接读它
    }

    # 可选：方案 C 的大模型人话解读层（默认关闭，--llm 开启）。
    # 模型不可用时自动降级，不影响主流程。
    if llm:
        try:
            import llm_explainer  # noqa: E402
            llm_text = llm_explainer.generate_for_pipeline(
                result, image_path=str(image_path), verbose=verbose)
            if llm_text:
                result["llm_explanation"] = llm_text
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [LLM] 解读层异常，已跳过：{type(e).__name__}：{e}")

    # 可选：方案 B 的多 Agent 圆桌交叉复核（默认关闭，--roundtable 开启）。
    # 纯结构化角色复核，零额外依赖，随时可跑。
    if roundtable:
        try:
            import roundtable  # noqa: E402
            rt = roundtable.run_roundtable(stem, evidence, {
                "risk_level": risk_level, "reasons": reasons})
            if rt:
                result["roundtable"] = rt
                if verbose:
                    print("  [圆桌] 多 Agent 交叉复核完成")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [圆桌] 复核异常，已跳过：{type(e).__name__}：{e}")

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


def write_pdf(stem):
    """把 Markdown 报告转成 PDF（复用 tools/export_pdf.py）

    为什么要这一步：法务和品牌方要的是一份能存档、能转发、带页脚边界声明的文件，
    Markdown 对他们不方便。导出失败不影响主流程，所以这里吞掉异常只返回 None。
    """
    try:
        run_tool("export_pdf.py", stem, timeout=180)
        f = REPO / "reports" / f"report_{stem}.pdf"
        return str(f) if f.exists() else None
    except Exception:  # noqa: BLE001
        return None


def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        print(__doc__)
        return 1
    fast = "--fast" in args
    json_only = "--json-only" in args
    want_pdf = "--pdf" in args
    want_llm = "--llm" in args
    want_roundtable = "--roundtable" in args

    # 配套文案：--text "..." 直接给，或 --text-file 文件.txt（一整段）
    text = None
    if "--text" in args:
        i = args.index("--text")
        if i + 1 < len(args):
            text = args[i + 1]
    if "--text-file" in args:
        i = args.index("--text-file")
        if i + 1 < len(args):
            text = Path(args[i + 1]).read_text(encoding="utf-8")

    # 参数解析：--text / --text-file 后面跟的那个值不能被当成图片路径
    known = {"--fast", "--json-only", "--pdf"}
    paths, skip_next = [], False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ("--text", "--text-file"):
            skip_next = True
            continue
        if not a.startswith("--"):
            paths.append(a)

    for p in paths:
        print(f"\n=== 鉴定：{Path(p).name} ===")
        res = analyze(p, fast=fast, verbose=not json_only, text=text,
                       llm=want_llm, roundtable=want_roundtable)
        v = res["verdict"]
        if json_only:
            print(json.dumps({"image": res["image"]["name"], "risk_level": v["risk_level"]},
                             ensure_ascii=False))
        else:
            print(f"\n  结论：{v['risk_level']}")
            for r in v["reasons"]:
                print(f"    - {r}")
            ag = res.get("agent") or {}
            if ag.get("summary"):
                print(f"  Agent：{ag['summary']}")
            for d in (ag.get("decisions") or []):
                if d["action"] == "skip":
                    print(f"    跳过 {d['tool']}（{d['rule']}）：{d['reason'][:70]}…")
            for a in (ag.get("actions") or []):
                print(f"    动作：{a['action']}（{a['rule']}）")
            print(f"  用时 {res['elapsed_seconds']}s")
            print(f"  完整结果：{res['_output_file']}")
            md = write_markdown(res)
            if md:
                print(f"  文字报告：{md}")
            if want_pdf:
                pf = write_pdf(Path(p).stem)
                print(f"  PDF 报告：{pf}" if pf else "  PDF 报告：导出失败（可手动跑 tools/export_pdf.py）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

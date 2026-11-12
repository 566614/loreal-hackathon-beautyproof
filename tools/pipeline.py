# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队流水线编排（串联八工具 + 规则引擎 + 报告），核心创新贡献
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

    # 视频（第三模态）：自动认扩展名，抽帧 → 逐帧复用图像流水线 → 视频级聚合
    python tools/pipeline.py <视频路径.mp4/mov/avi/webm/mkv>
    python tools/pipeline.py <视频路径> --max-frames 6 --fps-sample 0.5
    python tools/pipeline.py <含视频的目录> --fast
    # 开关与图片侧一致：--fast / --json-only / --llm / --roundtable 都会透传过去

大白话：
    以前要手动挨个敲六个工具、再敲规则引擎，容易漏、也容易弄错顺序。
    这个脚本把整套流程串成"一步"：送图进去，出一份可以直接给人看的结论。

产出：
    outputs/analysis_<图片名>.json   完整结果（Web 界面和报告都读它）
    reports/report_<图片名>.md       人能读的 Markdown 鉴定报告
    reports/report_<图片名>.pdf      （加 --pdf 时）可直接存档/转发的 PDF 版

    # 视频输入时（走 tools/run_video.py → tools/video_tool.py）
    outputs/analysis_video_<视频名>.json   视频级聚合结果（frames[] 逐帧证据 + cannot_prove）
    reports/report_video_<视频名>.md       视频级 Markdown 鉴定报告
    outputs/video_frames/<视频名>/frame_*.jpg   抽出来的帧
    outputs/<tool>_frame_*.json             每帧的逐工具证据（与图片侧同一套格式）
"""
import datetime as dt
import hashlib
import os
import shutil
import uuid
import json
import subprocess
import sys
import time
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
PY = sys.executable
PUBLISH_LOCK = threading.Lock()

# 七个工具：(证据里的名字, 脚本名, 是否默认开启)
# 深度学习那两个（aigc / trufor）要加载几百 MB 的模型，慢，所以支持 --fast 跳过
# spectral 是纯 numpy FFT，不需要模型，秒级返回，所以不进 SLOW_TOOLS
TOOL_LIST = [
    ("hash", "hash_tool.py", True),
    ("c2pa", "c2pa_tool.py", True),
    ("ela", "ela_tool.py", True),
    ("spectral", "spectral_tool.py", True),
    ("ocr", "ocr_tool.py", True),
    ("ai_label", "ai_label_tool.py", True),
    ("aigc", "aigc_tool.py", True),
    ("trufor", "trufor_tool.py", True),
]
SLOW_TOOLS = {"aigc", "trufor"}

# 视频扩展名 —— 命中就走视频流水线（抽帧 → 逐帧复用图像流水线 → 视频级聚合）
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
# 这些开关后面跟一个值，值不能被当成输入路径（--max-frames 6 里的 "6" 不是图片）
VALUE_FLAGS = ("--text", "--text-file", "--max-frames", "--fps-sample", "--product", "--references")

# 每个工具的"人话说明" —— 报告页和网页都用它，避免各写一套口径
TOOL_META = {
    "hash": {"label": "文件指纹", "job": "给这张图算一个独一无二的身份证号，看它是不是跟已知原图一字不差"},
    "c2pa": {"label": "内容凭证", "job": "查图片里有没有官方的「出生证」，记录它是谁拍的、被哪些软件改过"},
    "ela": {"label": "压缩痕迹", "job": "把图压一遍再还原，看哪块区域的压缩反应跟周围不一样（注意：它只查『局部被二次编辑』，对整图 AI 生成基本无效，所以不能单独给 AI 图定案）"},
    "spectral": {"label": "频域取证", "job": "看这张图的高频能量和频谱规整度像不像相机直出照片 —— 这是与压缩历史无关的独立证据线，用来兜住 AIGC 模型换生成器就失效的盲区"},
    "ocr": {"label": "文字识别", "job": "把图上所有字抄下来，跟品牌方给的标准文案逐字比对"},
    "ai_label": {"label": "可见AI标识", "job": "识别图上的 AI 来源标识。可见标识可能被复制、误读或缺失。"},
    "aigc": {"label": "AI 生成检测", "job": "返回当前模型对整图生成的倾向分，并记录模型身份。"},
    "trufor": {"label": "篡改痕迹检测", "job": "通过 TruFor 找局部异常线索，定位区域供原图复核。"},
    # 文案侧（只有传了 --text 才会跑，不影响原来只鉴定图片的用法）
    "text": {"label": "文案体检", "job": "查文案里有没有《广告法》《化妆品监督管理条例》点名的敏感宣称，逐条给出法律出处、命中位置和上下文"},
    "crossmodal": {"label": "图文交叉验证", "job": "核对配图用途和来源证据，并比较归属于同一产品、指标及条件的宣称。"},
}


# ---------------------------------------------------------------- 图片 → base64
def to_data_uri(path, max_side=900, quality=85):
    """把图片压成缩略图再转成 data URI —— 这样报告 / 网页不用依赖原图文件，拷走就能看。
    实际编码逻辑收敛到 tools/_imageutil.image_to_base64（pipeline 与 llm_explainer 共用单一实现）。
    """
    sys.path.insert(0, str(TOOLS))
    from _imageutil import image_to_base64
    return image_to_base64(path, max_side=max_side, quality=quality)


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
def run_tool(script, image_path, timeout=900, output_dir=None):
    """跑单个工具脚本，把它的输出 JSON 读回来"""
    env = dict(os.environ, BP_RUN_OUTPUT=str(output_dir)) if output_dir else None
    proc = subprocess.run(
        [PY, str(TOOLS / script), str(image_path)],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="ignore", timeout=timeout,
    )
    return proc


def _import_rule_engine():
    """集中处理「把 tools/ 加进 path 再 import rule_engine」这件事。
    之前 collect_evidence / judge / explain_verdict 三处各自写一遍同样的
    sys.path.insert + import，既重复又容易漏。收到一处即可。
    """
    sys.path.insert(0, str(TOOLS))
    import rule_engine  # noqa: E402
    return rule_engine


def collect_evidence(stem, output_dir=None):
    """收集 outputs/ 下所有 <tool>_<stem>.json —— 与规则引擎口径完全一致"""
    return _import_rule_engine().load_evidence(REPO, stem, output_dir)


def judge(evidence):
    return _import_rule_engine().judge(evidence)


def explain_verdict(risk_level, evidence):
    """把分级结果翻译成品牌方 / 法务也看得懂的大白话"""
    return _import_rule_engine().explain(risk_level, evidence)


# 理由文本里出现这些工具别名 → 判定该工具「对最终档位有贡献」。
# 规则引擎的 reasons 每条都以工具名为前缀，据此把「档位」回溯到具体工具证据，
# 形成「工具证据 → 档位 → 处置建议」的可追溯链路（赛题要的"核验过程清晰"）。
_REASON_TOOL_HINTS = [
    ("图文交叉验证", "crossmodal"),
    ("AI 生成检测", "aigc"),
    ("TruFor", "trufor"),
    ("文案体检", "text"),
    ("ELA", "ela"),
    ("C2PA", "c2pa"),
]

# 这些措辞说明该工具「这次没有抬高风险档位」（得分偏低 / 弃权 / 跳过 / 没查到凭证），
# 即便它出现在 reasons 里、带工具前缀，也不能算作"推动最终档位"的证据，否则会误导复核。
_NEGATIVE_MARKERS = [
    "看起来像真实", "拿不准", "不做判定", "未返回", "未部署", "未命中",
    "未发现明显", "未检出", "没能读出凭证", "现有工具未发现", "未查出明确",
    "未达定级线", "已跳过", "未做图文比对", "未发现互相矛盾", "仅提示",
    "<0.2", "<0.5", "<0.9",
]


def _contributing_tools(reasons):
    """从规则引擎 reasons 反查「哪些工具的证据直接推动了最终档位」

    只把"抬高了档位"的工具算进来：带工具前缀、且不是上述负面/弃权措辞的理由。
    例如 aigc 给出 0.0（<0.2，看起来像真实图）虽出现在 reasons 里，但它是**否定信号**，
    不应被记成"推动 high_risk 的元凶"——真正推动的是 TruFor。
    """
    contrib = set()
    for r in (reasons or []):
        if any(m in r for m in _NEGATIVE_MARKERS):
            continue
        for hint, tool in _REASON_TOOL_HINTS:
            if hint in r:
                contrib.add(tool)
                break
    return contrib


def build_decision_trace(stem, image_name, risk_level, reasons, evidence):
    """导出「工具证据 → 档位 → 处置建议」的可追溯链路 dict

    返回结构：
        schema / image_stem / image_name / generated_at
        risk_level                          最终档位
        reasons（列表）+ reason（合并段）    判定依据，供复核逐项对照
        action_playbook                     来自 rule_engine.ACTION_PLAYBOOK 的处置建议
        trace[]                             每个工具的核验记录：
            tool                          工具名
            observed                      该工具本次观察到的现象
            evidence_ref                  outputs/<tool>_<stem>.json 证据文件路径
            contributed_to_risk_level    该工具是否直接推动了最终档位

    纯结构化、零深度学习依赖，单测可直接喂 mock evidence 验证。
    """
    rule_engine = _import_rule_engine()
    contrib = _contributing_tools(reasons)
    # 先排图像侧固定工具，再排其余（文案侧 / 圆桌等），让"核验时间线"稳定可读
    base_tools = [t for t, _, _ in TOOL_LIST]
    base_set = set(base_tools)
    ordered = base_tools + [t for t in evidence if t not in base_set]
    trace = []
    for tool in ordered:
        if tool not in evidence:
            continue
        data = evidence[tool]
        trace.append({
            "tool": tool,
            "observed": data.get("observed") or "（无观察记录）",
            "evidence_ref": f"outputs/{tool}_{stem}.json",
            "contributed_to_risk_level": tool in contrib,
        })
    return {
        "schema": "beautyproof/decision-trace@1",
        "image_stem": stem,
        "image_name": image_name,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "risk_level": risk_level,
        "reasons": list(reasons),
        "reason": "\n".join(reasons) if reasons else "（无判定依据）",
        "action_playbook": rule_engine.ACTION_PLAYBOOK[risk_level],
        "trace": trace,
    }


def run_text_tools(stem, text, verbose=True, on_progress=None, base_idx=0, total=8,
                   output_dir=None, product=None, body_claims=None, image_claims=None, image_present=None):
    """跑文案侧两个工具，结果写进 outputs/ —— 规则引擎随后就能和图像侧证据一起读到

    text_tool       只看文字本身：违禁宣称（硬规则，有法律出处）+ 写作特征（弱提示）
    crossmodal_tool 拿文案去跟图像侧的结论对质：说实拍却是 AI 图 → 硬矛盾

    写文件而不是直接返回，是为了和图像侧共用同一套 outputs/<tool>_<stem>.json 约定，
    规则引擎不用为文案侧开特例。
    """
    sys.path.insert(0, str(TOOLS))
    import text_tool  # noqa: E402
    import crossmodal_tool  # noqa: E402

    output_dir = output_dir or REPO / "outputs"
    written = []

    # 1) 文案体检
    if verbose:
        print("  · text    ...", end="", flush=True)
    if on_progress:
        on_progress("text", base_idx + 1, total, "running")
    try:
        rep = text_tool.build_evidence(text, source_id=stem, mode="single")
        (output_dir / f"text_{stem}.json").write_text(
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
        img_ev = collect_evidence(stem, output_dir)
        cons, negated = crossmodal_tool.check(text, img_ev, product, body_claims=body_claims,
                                              image_claims=image_claims, image_present=image_present)
        rep = crossmodal_tool.build_evidence(text, cons, img_ev, stem, negated=negated)
        (output_dir / f"crossmodal_{stem}.json").write_text(
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
def analyze(image_path=None, fast=False, skip=(), verbose=True, timeout=900,
            on_progress=None, text=None, llm=False, roundtable=False, *,
            product=None, references=None, publish=True, verified_references=(), market=None,
            reference_image=None):
    """对一张图跑完整流水线，返回结构化结果

    on_progress: 可选回调 fn(tool, 第几个, 共几个, 状态) —— Web 界面靠它显示实时进度
    text:        可选的配套文案（种草文 / 评论）。传了才跑文案侧两个工具；
                 不传时行为与以前完全一致，仍是纯图片鉴定。
    """
    import claim_review
    from tempfile import TemporaryDirectory

    text = (text or "").strip()
    product = (product or "").strip() or None
    market = (market or "").strip() or None
    references = claim_review.normalize_references(references)
    if not image_path and not text:
        raise ValueError("至少提供图片或正文")
    if reference_image and not image_path:
        raise ValueError("参考图片需要与宣传图片一起提交")
    original = Path(image_path).resolve() if image_path else None
    if original and not original.is_file():
        raise FileNotFoundError(f"找不到图片: {original.name}")

    run_id = uuid.uuid4().hex
    output_dir = REPO / "outputs" / "runs" / run_id
    output_dir.mkdir(parents=True)
    if original:
        # Take a stable snapshot. Detectors and the file hash use identical bytes.
        with TemporaryDirectory(dir=output_dir) as input_dir:
            image_path = Path(input_dir) / original.name
            shutil.copyfile(original, image_path)
            try:
                from PIL import Image
                with Image.open(image_path) as im:
                    im.verify()
            except (OSError, ValueError) as e:
                raise ValueError(f"这不是一张可读取的图片：{original.name}") from e
            info = image_info(image_path)
            with image_path.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            info.update(path=str(original), sha256=digest.hexdigest())
            visual_match = None
            if reference_image:
                ref_path = Path(input_dir) / "reference_image"
                shutil.copyfile(reference_image, ref_path)
                try:
                    import package_match
                    visual_match = package_match.locate(image_path, ref_path)
                except Exception:  # Optional OpenCV decoding/matching must not abort the remaining checks.
                    visual_match = {"status": "无法判断", "regions": [],
                                    "reason": "参考图无法读取或本地定位组件不可用。"}
            return _analyze_run(image_path, info, output_dir, fast, skip, verbose, timeout,
                                on_progress, text, llm, roundtable, product, references, publish, verified_references, market, visual_match)

    return _analyze_run(None, None, output_dir, fast, skip, verbose, timeout,
                        on_progress, text, llm, roundtable, product, references, publish, verified_references, market)


def _analyze_run(image_path, info, output_dir, fast, skip, verbose, timeout,
                 on_progress, text, llm, roundtable, product, references, publish, verified_references, market, visual_match=None):
    import claim_review
    import crossmodal_tool

    stem = info["stem"] if info else "text"
    image_info_fields = info or {}
    # ---------------------------------------------------------------- Agent 调度
    # 不再是「六个工具按固定顺序跑完」，而是分三波：每跑完一波，
    # 先问 Agent「根据已查到的东西，下一波哪些值得跑」，再决定跑不跑。
    sys.path.insert(0, str(TOOLS))
    import planner  # noqa: E402

    ctx = {
        "has_text": bool(text and text.strip()),
        "image_format": image_info_fields.get("format"),
        "image_pixels": (image_info_fields.get("width") or 0) * (image_info_fields.get("height") or 0),
    }

    script_of = {tool: script for tool, script, _ in TOOL_LIST}
    enabled = {tool for tool, _, on in TOOL_LIST if image_path and on and tool not in skip}

    total = len([x for x in enabled]) + (2 if ctx["has_text"] else 0)
    t0 = time.time()
    ran, failed = [], []
    tool_errors = {}  # 工具失败诊断（stderr 末尾），新增字段，不破坏 tools_failed 形状
    skipped, decisions = set(), []
    idx = 0

    def _note(tool, state, spent=None):
        if on_progress:
            on_progress(tool, idx, total, state)

    for wave_name, wave_tools in planner.WAVES:
        ev_now = collect_evidence(stem, output_dir)
        wave_skip, dec = planner.decide_next(ev_now, ctx, done=set(ran))
        if not fast:
            required = {"aigc", "trufor", "ocr"} & enabled
            wave_skip -= required
            dec = [d for d in dec if d["tool"] not in required]
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
            proc = None
            try:
                proc = run_tool(script_of[tool], image_path, timeout=timeout, output_dir=output_dir)
                ok = proc.returncode == 0 and (output_dir / f"{tool}_{stem}.json").is_file()
            except subprocess.TimeoutExpired:
                ok = False
            if ok:
                ran.append(tool)
                if verbose:
                    print(f" 完成 ({time.time() - ts:.1f}s)")
            else:
                failed.append(tool)
                # 抓取工具报错诊断：以前只说"跳过"，真崩了没处查；
                # 现在把 stderr 末尾截一段，既打印也写进结果的 tool_errors。
                err = (proc.stderr or "").strip() if proc is not None else ""
                tail = err[-600:] if err else "（超时，无 stderr）"
                tool_errors[tool] = "timeout" if proc is None else (err[:600] or "no_stderr")
                if verbose:
                    if err:
                        print(f" 跳过（工具报错）：…{tail}")
                    else:
                        print(" 跳过（工具报错或超时）")
            _note(tool, "done" if ok else "failed")

    evidence = collect_evidence(stem, output_dir)
    ocr = claim_review.ocr_lines(evidence)
    uncertain_ocr = [row for row in evidence.get("ocr", {}).get("evidence", [])
                     if isinstance(row, dict) and isinstance(row.get("text"), str)
                     and isinstance(row.get("confidence"), (int, float)) and row["confidence"] < .70]
    image_text = "\n".join(line.get("text", "") for line in ocr if isinstance(line, dict))
    body_claims = claim_review.extract(text, product)
    image_claims = claim_review.extract(image_text, product, "image")
    claim_model_status = "insufficient_evidence"
    if llm and (text or image_text):
        import llm_explainer
        model_rows = [llm_explainer.extract_claim_fields(source, product)
                      for source in (text, image_text) if source]
        claim_model_status = "completed" if all(rows is not None for rows in model_rows) else "model_unavailable"
        by_source = iter(model_rows)
        for source, facts, kind in ((text, body_claims, "body"), (image_text, image_claims, "image")):
            if source:
                claim_review.enrich(facts, source, next(by_source) or [], kind)
    for claim in image_claims:
        claim["regions"] = claim_review.ocr_regions(ocr, claim["span"])

    text_tools_ran = []
    if text:
        text_tools_ran = run_text_tools(
            stem, text, verbose=verbose, on_progress=on_progress, base_idx=idx,
            total=total, output_dir=output_dir, product=product,
            body_claims=body_claims, image_claims=image_claims, image_present=bool(image_path),
        )
        evidence = collect_evidence(stem, output_dir)
    elif image_text:
        import text_tool
        evidence["text"] = text_tool.build_evidence(image_text, stem)
        evidence["crossmodal"] = crossmodal_tool.build_evidence("", [], evidence, stem)
        text_tools_ran.append("text")
        skipped.add("crossmodal")
    else:
        skipped.update(planner.TEXT_TOOLS)

    cross = (evidence.get("crossmodal", {}).get("evidence") or [{}])[0]
    if visual_match is not None:
        cross["package_reference"] = visual_match
    if text and image_path:
        cross["alignment"] = crossmodal_tool.alignment(body_claims, image_claims)
    catalog = claim_review.verified_catalog()
    selected = {ref["id"]: ref for ref in verified_references}
    selected.update(catalog)
    claims = claim_review.assess(body_claims, image_claims, references, cross.get("contradictions", []),
                                 verified_references=selected.values(), market=market)
    context_conflicts = claim_review.context_conflicts(body_claims)
    semantic_findings, semantic_status = [], "insufficient_evidence"
    semantic_reference_count = None
    semantic_reference_total = None
    if llm and text and os.environ.get("BP_TEXT_API_ONLY") == "1":
        import llm_explainer
        products = {c["product"] for c in body_claims if c.get("product")}
        if product:
            products.add(product)
        versions = set(claim_review.PRODUCT_VERSION.findall(text))
        manual_ids = list(dict.fromkeys(ref["id"] for ref in verified_references if ref["id"] in catalog))
        ordered = [catalog[key] for key in manual_ids] + [ref for key, ref in catalog.items() if key not in manual_ids]
        eligible = [ref for ref in ordered if len(products) == 1 and ref["product"] in products
                    and market and ref.get("market") == market
                    and (len(versions) != 1 or not ref.get("product_version")
                         or ref["product_version"] in versions)]
        semantic_reference_total = len(eligible)
        relevant = [{k: ref[k] for k in ("id", "source", "excerpt", "product", "market", "source_type", "version", "product_version", "locator", "method", "sample", "population", "period", "conditions") if k in ref}
                    for ref in eligible[:8]]
        semantic_reference_count = len(relevant)
        result = llm_explainer.semantic_review(text, relevant)
        semantic_status = "completed" if result is not None else "model_unavailable"
        semantic_findings = result or []
    asset_hash = image_info_fields.get("sha256")
    for tool, item in evidence.items():
        item.update(run_id=output_dir.name, image_sha256=asset_hash)
        (output_dir / f"{tool}_{stem}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    risk_level, reasons = judge(evidence)
    if any(claim["status"] == "存在冲突" for claim in claims):
        if risk_level != "high_risk":
            risk_level = "suspicious"
        reasons.append("宣称核验：图文或对应参考资料存在冲突")
    if any(claim["status"] == "证据不足" for claim in claims):
        reasons.append("宣称核验：部分宣称缺少对应产品、指标和条件的证据")
    if context_conflicts:
        if risk_level != "high_risk":
            risk_level = "suspicious"
        reasons.append("正文核验：同一产品及条件下存在不同数值，请核对原文")
    if risk_level == "credible":
        risk_level = "inconclusive"

    signals = _import_rule_engine().image_signals(evidence)
    checks = {}
    for tool, _, _ in TOOL_LIST:
        item = (evidence.get(tool, {}).get("evidence") or [{}])[0]
        unavailable = tool in failed or item.get("available") is False
        status = "completed" if tool in evidence and not unavailable else "insufficient_evidence"
        if unavailable and tool in {"aigc", "trufor", "ocr"}:
            status = "model_unavailable"
        checks[tool] = {"status": status}
        if tool in tool_errors:
            checks[tool]["reason"] = tool_errors[tool]
        elif item.get("reason"):
            checks[tool]["reason"] = item["reason"]
        if tool == "aigc" and signals["origin"]["score"] is not None and not signals["origin"]["threshold_profile"]:
            checks[tool]["decision_status"] = "insufficient_evidence"
    checks["claim_model"] = {"status": claim_model_status}
    checks["semantic_review"] = {"status": semantic_status}
    if semantic_reference_count is not None:
        checks["semantic_review"]["reference_count"] = semantic_reference_count
        checks["semantic_review"]["reference_total"] = semantic_reference_total
        if semantic_status == "model_unavailable":
            checks["semantic_review"]["reason"] = "DeepSeek 未配置、请求失败或结果无法解析。"
        elif semantic_reference_count == 0:
            checks["semantic_review"]["reason"] = "未找到产品及市场一致的资料；仅检查正文上下文。"
        elif semantic_reference_total > semantic_reference_count:
            checks["semantic_review"]["reason"] = f"符合条件的 {semantic_reference_total} 条资料中仅审核了前 {semantic_reference_count} 条；手选资料优先。"
    if llm and os.environ.get("BP_TEXT_API_ONLY") == "1":
        checks["claim_model"]["model"] = os.environ.get("BP_DEEPSEEK_MODEL", "deepseek-flash")
        if claim_model_status == "model_unavailable":
            checks["claim_model"]["reason"] = ("DeepSeek 请求失败或返回内容无法解析" if os.environ.get("DEEPSEEK_API_KEY")
                                                 else "尚未配置 DEEPSEEK_API_KEY")
    checks["text"] = {"status": "completed" if "text" in evidence else "insufficient_evidence"}
    checks["crossmodal"] = {"status": "completed" if text and cross.get("conflict_level") in
                            ("none", "medium", "hard") else "insufficient_evidence"}
    plain = explain_verdict(risk_level, evidence)
    if any(c["status"] == "存在冲突" and c["references"] for c in claims) and risk_level == "suspicious":
        plain.update(headline="宣传宣称与提供的依据不一致",
                     summary="发现对应产品、指标和条件下的数值差异，详见宣称核验。",
                     what_to_do="核对原始资料和文案后再决定是否发布。")

    # 把规则引擎的定级结论落盘为 verdict_<stem>.json —— 之前漏了这一步，
    # 导致 report_generator 读不到定级、样例报告一律显示「NO_VERDICT」，
    # 评委实测会以为流水线没出结论（交付一致性硬伤）。现在与 pipeline 的
    # judge 结论同源，报告页 / 网页 / PDF 读到的是同一份结论。
    try:
        (output_dir / f"verdict_{stem}.json").write_text(
            json.dumps({"risk_level": risk_level, "reasons": reasons},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    # 决策闭环可视化前置：导出「工具证据 → 档位 → 处置建议」可追溯链路。
    # 这一步零依赖，必须在写最终 analysis 之前完成，保证即使后续 roundtable/llm
    # 异常也不丢这条核心链路。
    decision_trace = build_decision_trace(
        stem, image_info_fields.get("name") or "正文", risk_level, reasons, evidence)
    for item in decision_trace["trace"]:
        item["evidence_ref"] = f"outputs/runs/{output_dir.name}/{item['tool']}_{stem}.json"
    trace_path = output_dir / f"decision_trace_{stem}.json"
    trace_path.write_text(
        json.dumps(decision_trace, ensure_ascii=False, indent=2), encoding="utf-8")

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
            p = output_dir / name
            if p.exists():
                visuals[key] = to_data_uri(p, max_side=1100, quality=90)

    result = {
        "schema": "beautyproof/analysis@1",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "image": info,
        "run_id": output_dir.name,
        "audit": {"claims": claims, "context_conflicts": context_conflicts, "semantic_findings": semantic_findings,
                  "uncertain_ocr": uncertain_ocr, "checks": checks, "image_signals": signals,
                  "crossmodal": cross,
                  "ai_illustration_disclosed": bool(cross.get("ai_illustration_disclosed"))},
        "verdict": {
            "risk_level": risk_level,
            "explain": plain,
            "reasons": reasons,
            "tools_used": sorted(evidence.keys()),
            "tools_ran": ran + text_tools_ran,
            "tools_failed": failed,
            "tool_errors": tool_errors,
        },
        "text_input": (text or "").strip() or None,
        "agent": agent_log,   # 决策过程：跑了什么、跳了什么、为什么
        "decision_trace_file": str(trace_path),  # 工具→证据→档位→建议 可追溯链路
        "evidence": evidence,
        "visuals": visuals,
        "tool_meta": TOOL_META,   # 每个工具的人话说明，网页/离线 Demo 直接读它
    }

    # 可选：方案 C 的大模型人话解读层（默认关闭，--llm 开启）。
    # 模型不可用时自动降级，不影响主流程。
    if llm and not os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("BP_TEXT_API_ONLY") != "1":
        try:
            import llm_explainer  # noqa: E402
            llm_text = llm_explainer.generate_for_pipeline(
                result, image_path=str(image_path) if image_path else None,
                verbose=verbose, output_dir=output_dir)
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
                "risk_level": risk_level, "reasons": reasons}, output_dir=output_dir)
            if rt:
                result["roundtable"] = rt
                if verbose:
                    print("  [圆桌] 多 Agent 交叉复核完成")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [圆桌] 复核异常，已跳过：{type(e).__name__}：{e}")

    out = output_dir / f"analysis_{stem}.json"
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    out.write_text(payload, encoding="utf-8")
    if publish and info:
        # Preserve the latest named snapshot for existing sample/review readers.
        # Decisions and report generation always read the isolated run.
        snapshot = output_dir / "publish.json"
        snapshot.write_text(payload, encoding="utf-8")
        with PUBLISH_LOCK:
            snapshot.replace(REPO / "outputs" / out.name)
    result["_output_file"] = str(out)
    return result


def write_markdown(result):
    """生成人能读的 Markdown 报告（复用现成的 report_generator.py）"""
    stem = result["image"]["stem"] if result["image"] else "text"
    try:
        output_dir = Path(result["_output_file"]).parent
        proc = run_tool("report_generator.py", stem, timeout=120, output_dir=output_dir)
        if proc.returncode != 0:
            return None
        p = output_dir / f"report_{stem}.md"
        return str(p) if p.exists() else None
    except Exception as e:  # noqa: BLE001
        return f"（Markdown 报告生成失败：{type(e).__name__}：{e}）"


def write_pdf(stem, output_dir=None):
    """把 Markdown 报告转成 PDF（复用 tools/export_pdf.py）

    为什么要这一步：法务和品牌方要的是一份能存档、能转发、带页脚边界声明的文件，
    Markdown 对他们不方便。导出失败不影响主流程，所以这里吞掉异常只返回 None。
    """
    try:
        proc = run_tool("export_pdf.py", stem, timeout=180, output_dir=output_dir)
        if proc.returncode != 0:
            return None
        f = (output_dir or REPO / "reports") / f"report_{stem}.pdf"
        return str(f) if f.exists() else None
    except Exception:  # noqa: BLE001
        return None


def is_video_input(p):
    """这个输入是不是视频：扩展名命中即可；传目录时，目录里含视频也算。

    为什么要认目录：赛题素材包常是「一个文件夹里一堆 mp4」，让评委挨个敲路径不现实。
    目录里既有图又有视频时，视频归视频、图归图（见 expand_video_inputs）。
    """
    p = Path(p)
    if p.is_dir():
        try:
            return any(f.is_file() and f.suffix.lower() in VIDEO_EXTS
                       for f in sorted(p.iterdir()))
        except Exception:  # noqa: BLE001 目录不可读就当它不是视频目录，走原逻辑报错
            return False
    return p.suffix.lower() in VIDEO_EXTS


def expand_video_inputs(p):
    """把「一个目录」展开成它里面的视频文件列表；本来就是文件就原样返回。"""
    p = Path(p)
    if p.is_dir():
        return sorted(f for f in p.iterdir()
                      if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
    return [p]


def run_video_input(video_path, fast=False, json_only=False, llm=False,
                    roundtable=False, max_frames=None, fps_sample=None):
    """视频输入的统一入口：转调 tools/run_video.py（内部再调 video_tool.analyze_video）。

    为什么不在这里重新实现一遍：检测逻辑只有一份（逐帧复用 pipeline.analyze），
    管道这里只做「认出视频 + 把开关透传过去」，避免两条视频路径日后各写一套。
    """
    sys.path.insert(0, str(TOOLS))
    import run_video  # noqa: E402

    argv = ["--input", str(video_path)]
    if fast:
        argv.append("--fast")
    if json_only:
        argv.append("--json-only")
    if llm:
        argv.append("--llm")
    if roundtable:
        argv.append("--roundtable")
    if max_frames is not None:
        argv += ["--max-frames", str(max_frames)]
    if fps_sample is not None:
        argv += ["--fps-sample", str(fps_sample)]
    return run_video.main(argv)


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

    # 视频侧专用开关（带值），默认值 None = 不传，让 run_video 用它自己的默认
    max_frames = None
    if "--max-frames" in args:
        i = args.index("--max-frames")
        if i + 1 < len(args):
            try:
                max_frames = int(args[i + 1])
            except ValueError:
                print(f"--max-frames 需要一个整数，收到：{args[i + 1]}")
                return 1
    fps_sample = None
    if "--fps-sample" in args:
        i = args.index("--fps-sample")
        if i + 1 < len(args):
            try:
                fps_sample = float(args[i + 1])
            except ValueError:
                print(f"--fps-sample 需要一个数字，收到：{args[i + 1]}")
                return 1

    # 参数解析：带值的开关后面那个值不能被当成输入路径
    paths, skip_next = [], False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in VALUE_FLAGS:
            skip_next = True
            continue
        if not a.startswith("--"):
            paths.append(a)

    for flag in ("--product", "--references"):
        if flag in args and (args.index(flag) + 1 == len(args) or args[args.index(flag) + 1].startswith("--")):
            print(f"{flag} 需要一个值", file=sys.stderr)
            return 1

    def option(flag):
        return args[args.index(flag) + 1] if flag in args else None

    product = option("--product")
    references_path = option("--references")
    references = json.loads(Path(references_path).read_text(encoding="utf-8")) if references_path else None
    if not paths and text:
        result = analyze(text=text, product=product, references=references, verbose=False, llm=want_llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not paths:
        print("请提供图片路径或 --text 正文", file=sys.stderr)
        return 1

    rc = 0
    for p in paths:
        # ---------------------------------------------------------- 输入类型路由
        # 视频走视频流水线（抽帧 → 逐帧复用图像流水线 → 视频级聚合 + 视频报告）；
        # 图片走原来的图像主路径，一条代码都不变。
        if is_video_input(p):
            for vp in expand_video_inputs(p):
                print(f"\n=== 鉴定（视频）：{vp.name} ===")
                try:
                    vrc = run_video_input(
                        vp, fast=fast, json_only=json_only, llm=want_llm,
                        roundtable=want_roundtable, max_frames=max_frames,
                        fps_sample=fps_sample)
                    if vrc:
                        rc = vrc
                except Exception as e:  # noqa: BLE001
                    print(f"  视频分析失败：{type(e).__name__}：{e}")
                    rc = 1
            continue

        print(f"\n=== 鉴定：{Path(p).name} ===")
        res = analyze(p, fast=fast, verbose=not json_only, text=text,
                       llm=want_llm, roundtable=want_roundtable,
                          product=product, references=references)
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
                pf = write_pdf(Path(p).stem, Path(res["_output_file"]).parent)
                print(f"  PDF 报告：{pf}" if pf else "  PDF 报告：导出失败（可手动跑 tools/export_pdf.py）")
    return rc


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
美妆图取证台 —— 本地 Web 演示服务

启动：
    ./run.sh web/app.py          # 或 python web/app.py
    然后浏览器打开 http://127.0.0.1:5000

大白话：
    比赛演示不能让评委盯着命令行看。这个网页让任何人
    拖一张图进来，看着六个检测工具一步步跑完，
    最后拿到一份"可信 / 可疑 / 高风险"的鉴定结论。

接口：
    GET  /                 主页
    GET  /api/samples      已有预生成结果的示例样本列表
    GET  /api/sample/<stem> 读某个示例的完整分析结果
    POST /api/analyze      上传图片开跑（multipart: file / fast=0|1）
    POST /api/agent/turn   对话审核回合（multipart: message / file / previous_job_id）
    GET  /api/job/<id>     轮询任务进度与结果
"""
import json
import os
import re
import sys
import threading
import traceback
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, render_template, request

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import pipeline  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 最大 32MB

UPLOADS = REPO / "uploads"
UPLOADS.mkdir(exist_ok=True)

JOBS = {}          # job_id -> {status, steps, result, error}
JOBS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
CATALOG_LOCK = threading.Lock()

# 任务结果里存着完整 analysis（含原图 data URI，单个可达数 MB）。
# 演示时长跑或接口被反复调用时 JOBS 会无限增长，故设上限并淘汰最旧的已完结任务。
MAX_JOBS = 64


def _prune_jobs():
    """把 JOBS 压到 MAX_JOBS 以内：先淘汰已完结的，仍超限再按插入顺序淘汰最旧的。"""
    with JOBS_LOCK:
        if len(JOBS) <= MAX_JOBS:
            return
        for k in [k for k, v in JOBS.items() if v["status"] != "running"]:
            JOBS.pop(k, None)
            if len(JOBS) <= MAX_JOBS:
                return
        while len(JOBS) > MAX_JOBS:
            JOBS.pop(next(iter(JOBS)), None)   # dict 保序 → 即最旧


_STEM_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _safe_stem(stem) -> bool:
    """只允许字母 / 数字 / 下划线 / 点 / 横线，长度 1~128（防目录穿越，纵深防御）。"""
    return bool(_STEM_RE.match(stem or ""))

RISK_ZH = {
    "high_risk": "高风险",
    "suspicious": "可疑",
    "credible": "可信",
    "inconclusive": "无法判定",
}


def _run_job(job_id, image_path, fast, text=None, product=None, references=None, llm=False,
             verified_references=(), market=None, reference_image=None):
    job = JOBS[job_id]

    def cb(tool, idx, total, state):
        with JOBS_LOCK:
            job["steps"].append({"tool": tool, "state": state, "index": idx, "total": total})

    try:
        from contextlib import nullcontext
        with MODEL_LOCK if image_path or llm else nullcontext():
            res = pipeline.analyze(image_path, fast=fast, on_progress=cb, timeout=120,
                                   verbose=False, text=text, product=product, references=references, llm=llm,
                                   verified_references=verified_references, market=market,
                                   reference_image=reference_image)
        res["verdict"]["risk_zh"] = RISK_ZH.get(res["verdict"]["risk_level"],
                                                res["verdict"]["risk_level"])
        res["tool_meta"] = pipeline.TOOL_META
        with JOBS_LOCK:
            job["result"] = res
            job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        with JOBS_LOCK:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {str(e)[:300]}"


@app.get("/")
def index():
    bundle = Path(app.static_folder) / "workbench"
    if all((bundle / name).is_file() for name in ("main.js", "main.css")):
        return render_template("workbench.html")
    return render_template("index.html")


@app.get("/api/samples")
def samples():
    """列出 outputs/ 下已生成 analysis_*.json 的示例（不含内部字段）"""
    items = []
    for f in sorted((REPO / "outputs").glob("analysis_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            items.append({
                "stem": data["image"]["stem"],
                "name": data["image"]["name"],
                "risk_level": data["verdict"]["risk_level"],
                "risk_zh": RISK_ZH.get(data["verdict"]["risk_level"], ""),
            })
        except Exception:  # noqa: BLE001
            continue
    return jsonify({"samples": items})


@app.get("/api/sample/<stem>")
def sample(stem):
    if not _safe_stem(stem):
        return jsonify({"error": "sample not found"}), 404
    f = REPO / "outputs" / f"analysis_{stem}.json"
    if not f.exists():
        return jsonify({"error": "sample not found"}), 404
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"结果文件损坏或不可读：{type(exc).__name__}"}), 500
    # 兜底：老结果里没有工具说明和中文档位，这里补上，免得页面上卡片描述空白
    data.setdefault("tool_meta", pipeline.TOOL_META)
    lvl = data.get("verdict", {}).get("risk_level", "")
    data["verdict"]["risk_zh"] = RISK_ZH.get(lvl, lvl)
    return jsonify(data)


@app.get("/api/reference-catalog")
def reference_catalog():
    from claim_review import verified_catalog
    try:
        rows = verified_catalog().values()
    except (OSError, ValueError):
        return jsonify({"error": "无法读取已核验依据清单"}), 503
    return jsonify({"references": [{k: row[k] for k in ("id", "source", "product", "version", "locator")}
                                   for row in rows]})


@app.get("/api/config")
def config():
    return jsonify({"deepseek_configured": bool(os.environ.get("DEEPSEEK_API_KEY"))})


@app.post("/api/reference-catalog")
def add_reference():
    from claim_review import add_verified_reference
    try:
        with CATALOG_LOCK:
            row = add_verified_reference(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError:
        return jsonify({"error": "无法保存本地依据，请检查文件权限"}), 503
    return jsonify({"reference": {k: row[k] for k in ("id", "source", "product", "version", "locator")}}), 201


@app.post("/api/damage-review")
def damage_review():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"error": "请上传一张 JPG 或 PNG 图片。"}), 400

    from doubao_damage import analyze_image

    try:
        roi = json.loads(request.form["roi"]) if "roi" in request.form else None
    except ValueError:
        return jsonify({"error": "无法读取框选坐标。"}), 400
    try:
        result = analyze_image(upload.read(), roi)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


def _run_agent_job(job, workspace, path, message, copy, options, previous):
    def progress(tool, idx, total, state):
        with JOBS_LOCK:
            job["steps"].append({"tool": tool, "state": state, "index": idx, "total": total})

    try:
        try:
            import review_agents

            result, context = review_agents.run_turn(
                message, copy, path, Path(workspace.name), options, previous, MODEL_LOCK, progress,
            )
        finally:
            workspace.cleanup()
        with JOBS_LOCK:
            job.update(status="done", result=result, context=context)
    except Exception:
        app.logger.exception("Agent turn failed")
        with JOBS_LOCK:
            job.update(status="error", error="本次审核未完成，请重新发送。")
            if options.get("evidence_followup"):
                previous["evidence_session"]["followup_job_id"] = None


def _agent_image(upload, directory, name):
    from PIL import Image

    if not upload or not upload.filename:
        return None
    try:
        with Image.open(upload.stream) as image:
            suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image.format)
            if not suffix or getattr(image, "n_frames", 1) != 1:
                raise ValueError("请选择单张 JPG 或 PNG 图片。")
            image.verify()
        upload.stream.seek(0)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请换一张 JPG 或 PNG。") from exc
    path = directory / (name + suffix)
    upload.save(path)
    return path


@app.post("/api/agent/turn")
def agent_turn():
    from claim_review import normalize_references, verified_catalog

    role = request.form.get("role")
    if role not in (None, "consumer", "reviewer"):
        return jsonify({"error": "请选择消费者或企业审核人员入口。"}), 400
    publicity_only = role == "consumer"
    evidence_followup = request.form.get("evidence_followup") == "1"
    message = request.form.get("message", "").strip()
    copy = request.form.get("copy", "").strip() or None
    upload = request.files.get("file")
    if not message and not copy and not (upload and upload.filename) and not evidence_followup:
        return jsonify({"error": "请填写审核需求或附上图片。"}), 400
    if len(message) > 12000 or (copy and len(copy) > 12000):
        return jsonify({"error": "消息和宣传正文各最多 12000 字。"}), 400
    try:
        references = normalize_references(json.loads(request.form.get("references", "null")))
        ids = json.loads(request.form.get("reference_ids", "[]"))
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError("reference_ids 须为 ID 数组")
        roi = json.loads(request.form["roi"]) if "roi" in request.form else None
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        catalog = verified_catalog() if ids else {}
    except (OSError, ValueError):
        return jsonify({"error": "无法读取已核验依据清单。"}), 503
    if any(item not in catalog for item in ids):
        return jsonify({"error": "已核验依据不存在，请刷新后重选。"}), 400

    previous_id = request.form.get("previous_job_id")
    with JOBS_LOCK:
        previous = JOBS.get(previous_id, {}).get("context")
    if previous_id and (previous is None or previous.get("publicity_only", False) != publicity_only):
        return jsonify({"error": "上一回合已过期，请重新送审。"}), 409
    session = previous.get("evidence_session") if previous else None
    if evidence_followup:
        if publicity_only or not session:
            return jsonify({"error": "上一回合未等待功效试验补证，请重新送审。"}), 409
        if not ids or references:
            return jsonify({"error": "补证请选择已核验资料，用户摘录请先核对并录入资料库。"}), 400
        if (upload and upload.filename) or request.files.get("reference_image") or roi is not None:
            return jsonify({"error": "补证沿用上一回合图片，修改图片或选区请重新送审。"}), 400
        for field, supplied in (("copy", copy), ("product", request.form.get("product")),
                                ("market", request.form.get("market"))):
            if supplied is not None and supplied != session[field]:
                return jsonify({"error": "补证须沿用上一回合正文、产品和市场。"}), 400

    workspace = TemporaryDirectory(prefix="beautyproof-turn-")
    worker_started = False
    try:
        directory = Path(workspace.name)
        path = _agent_image(upload, directory, "submission")
        reference_image = _agent_image(request.files.get("reference_image"), directory, "reference")
        if (reference_image or roi) and not path:
            return jsonify({"error": "请同时附上要审核的图片。"}), 400
        options = {
            "evidence_followup": evidence_followup,
            "publicity_only": publicity_only,
            "llm": request.form.get("llm") != "0", "fast": request.form.get("fast") == "1",
            "product": request.form.get("product"), "market": request.form.get("market"),
            "references": references, "verified_references": [catalog[i] for i in dict.fromkeys(ids)],
            "reference_image": reference_image, "roi": roi,
            "image_name": Path(upload.filename).name if path else None,
        }
        job_id = uuid.uuid4().hex
        job = {"status": "running", "steps": [], "result": None, "error": None}
        with JOBS_LOCK:
            if len(JOBS) >= MAX_JOBS and all(item["status"] == "running" for item in JOBS.values()):
                return jsonify({"error": "审核服务正忙，请稍后重试。"}), 503
            if evidence_followup:
                if session["followup_job_id"]:
                    return jsonify({"error": "该次送审已提交补证，最多续审一次。"}), 409
                session["followup_job_id"] = job_id
            JOBS[job_id] = job
        _prune_jobs()
        try:
            threading.Thread(
                target=_run_agent_job, args=(job, workspace, path, message, copy, options, previous or {}),
                daemon=True,
            ).start()
        except RuntimeError:
            with JOBS_LOCK:
                JOBS.pop(job_id, None)
                if evidence_followup:
                    session["followup_job_id"] = None
            return jsonify({"error": "审核服务正忙，请稍后重试。"}), 503
        worker_started = True
        return jsonify({"job_id": job_id})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        if not worker_started:
            workspace.cleanup()


@app.post("/api/analyze")
def analyze():
    from claim_review import normalize_references, verified_catalog

    f = request.files.get("file")
    reference_file = request.files.get("reference_image")
    text = (request.form.get("text") or "").strip() or None
    if not text and not (f and f.filename):
        return jsonify({"error": "请提供图片或正文"}), 400
    if reference_file and reference_file.filename and not (f and f.filename):
        return jsonify({"error": "请同时上传宣传图片和包装参考图"}), 400
    fast = request.form.get("fast") == "1"
    product = (request.form.get("product") or "").strip() or None
    market = (request.form.get("market") or "").strip() or None
    llm = request.form.get("llm") == "1"
    try:
        references = normalize_references(json.loads(request.form.get("references", "null")))
        ids = json.loads(request.form.get("reference_ids", "[]"))
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ValueError("reference_ids 须为 ID 数组")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        catalog = verified_catalog() if ids else {}
    except (OSError, ValueError):
        return jsonify({"error": "无法读取已核验依据清单"}), 503
    if any(i not in catalog for i in ids):
        return jsonify({"error": "已核验依据不存在，请刷新后重选"}), 400
    verified_references = [catalog[i] for i in dict.fromkeys(ids)]

    path = None
    reference_path = None
    if f and f.filename:
        safe = uuid.uuid4().hex[:8] + "_" + Path(f.filename).name
        path = UPLOADS / safe
        f.save(path)
        from PIL import Image
        try:
            with Image.open(path) as im:
                im.verify()
        except (OSError, ValueError, Image.DecompressionBombError):
            path.unlink(missing_ok=True)
            return jsonify({"error": "无法读取所选图片，请换一张 PNG / JPG"}), 400

    if reference_file and reference_file.filename:
        reference_path = UPLOADS / (uuid.uuid4().hex[:8] + "_" + Path(reference_file.filename).name)
        reference_file.save(reference_path)
        try:
            with Image.open(reference_path) as im:
                im.verify()
        except (OSError, ValueError, Image.DecompressionBombError):
            reference_path.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            return jsonify({"error": "无法读取包装参考图，请换一张 PNG / JPG"}), 400

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "steps": [], "result": None, "error": None}
    _prune_jobs()
    threading.Thread(target=_run_job, args=(job_id, path, fast, text, product, references, llm, verified_references, market, reference_path),
                     daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def job(job_id):
    # 查不到时返回 404 —— 原实现直接 JOBS[job_id] 会抛 KeyError 变成 500，
    # 前端轮询一个已淘汰/不存在的 id 时只能显示"失败"，分不清是任务失败还是 id 无效。
    with JOBS_LOCK:
        rec = JOBS.get(job_id)
        if rec is None:
            return jsonify({"status": "not_found", "steps": [], "result": None,
                            "error": "任务不存在或已过期，请重新提交"}), 404
        snap = {
            "status": rec["status"],
            "steps": list(rec["steps"]),
            "result": rec["result"],
            "error": rec["error"],
        }
    return jsonify(snap)


if __name__ == "__main__":
    print("内容取证台已启动 → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

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
    GET  /api/job/<id>     轮询任务进度与结果
"""
import json
import sys
import threading
import traceback
import uuid
from pathlib import Path

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

RISK_ZH = {
    "high_risk": "高风险",
    "suspicious": "可疑",
    "credible": "可信",
    "inconclusive": "无法判定",
}


def _run_job(job_id, image_path, fast, text=None):
    job = JOBS[job_id]

    def cb(tool, idx, total, state):
        with JOBS_LOCK:
            job["steps"].append({"tool": tool, "state": state, "index": idx, "total": total})

    try:
        res = pipeline.analyze(image_path, fast=fast, on_progress=cb,
                               verbose=False, text=text)
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
    f = REPO / "outputs" / f"analysis_{stem}.json"
    if not f.exists():
        return jsonify({"error": "sample not found"}), 404
    data = json.loads(f.read_text(encoding="utf-8"))
    # 兜底：老结果里没有工具说明和中文档位，这里补上，免得页面上卡片描述空白
    data.setdefault("tool_meta", pipeline.TOOL_META)
    lvl = data.get("verdict", {}).get("risk_level", "")
    data["verdict"]["risk_zh"] = RISK_ZH.get(lvl, lvl)
    return jsonify(data)


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "没收到图片，请重试"}), 400
    fast = request.form.get("fast") == "1"
    # 配套文案（选填）：填了才会跑文案体检 + 图文交叉验证
    text = (request.form.get("text") or "").strip() or None

    safe = uuid.uuid4().hex[:8] + "_" + Path(f.filename).name
    path = UPLOADS / safe
    f.save(path)

    # 先确认它真的是张图：否则六个工具会各自以奇怪的方式失败，前端只看到"鉴定失败"
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
    except Exception:  # noqa: BLE001
        path.unlink(missing_ok=True)
        return jsonify({"error": f"「{Path(f.filename).name}」不是一张能被识别的图片，请换一张 PNG / JPG"}), 400

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "steps": [], "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, str(path), fast, text), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def job(job_id):
    with JOBS_LOCK:
        snap = {
            "status": JOBS[job_id]["status"],
            "steps": list(JOBS[job_id]["steps"]),
            "result": JOBS[job_id]["result"],
            "error": JOBS[job_id]["error"],
        }
    return jsonify(snap)


if __name__ == "__main__":
    print("内容取证台已启动 → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

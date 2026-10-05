# -*- coding: utf-8 -*-
"""2026-10-05 · 全面复检 · 第三批：Web 服务安全与健壮性加固

只加固、不改接口：所有正常请求的响应结构与状态码保持原样。

web/app.py
 1. `/api/job/<job_id>` 查不到会抛 KeyError → 500。改为返回 404（这是真 bug：
    页面轮询一个过期/不存在的 id 就会拿到 500，前端只能显示"失败"）。
 2. `JOBS` 字典无上限 —— 每个 job 存着完整 analysis result（含原图 data URI，可能几 MB），
    长时间演示或被人反复刷接口会无限吃内存。加 MAX_JOBS 上限 + 淘汰最旧的已完结任务。
 3. `/api/sample/<stem>` 遇到损坏 JSON 直接抛 → 500。改为返回明确错误。
 4. stem 白名单校验（纵深防御）：这些接口把 stem 直接拼进 `outputs/analysis_<stem>.json`。
    Werkzeug 默认 string 转换器不匹配 '/'，常规 `../` 已被路由层挡住；但线上 demo 公开可访问、
    且 %2F 在不同服务器/代理下解码行为不一致，再挡一次成本极低。

web/review_app.py
 5. `load_analysis()` 是本文件所有读结果端点的唯一入口，在其中做 stem 校验即可一次覆盖
    /api/review/<stem>、/api/report/<stem>、/api/review(POST) 与队列接口。
 6. `report()` 里 `err if 'err' in dir() else ""` 是脆弱写法（靠 dir() 探测局部变量是否存在），
    改为进入 try 前显式 `err = None`，语义等价且不依赖运行时 introspection。

幂等：逐处断言命中 1 次。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HELPER = '''

# ---------------------------------------------------------------- 纵深防御：标识白名单
# 这些接口把 stem 直接拼进 `outputs/analysis_<stem>.json`。Werkzeug 的 string 转换器
# 默认不匹配 '/'，所以常规 `../` 已被路由层挡住；但本项目有公开在线 demo，
# 且 %2F 在不同服务器 / 反代下的解码行为并不一致 —— 再挡一次，成本极低。
import re as _re

_STEM_RE = _re.compile(r"^[A-Za-z0-9_.\\-]{1,128}$")


def _safe_stem(stem) -> bool:
    """只允许字母 / 数字 / 下划线 / 点 / 横线，长度 1~128。"""
    return bool(_STEM_RE.match(stem or ""))

'''

PATCHES = [
    # ---------------- web/app.py ----------------
    ("web/app.py",
     '''import json
import sys
import threading
import traceback
import uuid
from pathlib import Path''',
     '''import json
import re
import sys
import threading
import traceback
import uuid
from pathlib import Path'''),

    ("web/app.py",
     '''JOBS = {}          # job_id -> {status, steps, result, error}
JOBS_LOCK = threading.Lock()''',
     '''JOBS = {}          # job_id -> {status, steps, result, error}
JOBS_LOCK = threading.Lock()

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


_STEM_RE = re.compile(r"^[A-Za-z0-9_.\\-]{1,128}$")


def _safe_stem(stem) -> bool:
    """只允许字母 / 数字 / 下划线 / 点 / 横线，长度 1~128（防目录穿越，纵深防御）。"""
    return bool(_STEM_RE.match(stem or ""))'''),

    ("web/app.py",
     '''@app.get("/api/sample/<stem>")
def sample(stem):
    f = REPO / "outputs" / f"analysis_{stem}.json"
    if not f.exists():
        return jsonify({"error": "sample not found"}), 404
    data = json.loads(f.read_text(encoding="utf-8"))''',
     '''@app.get("/api/sample/<stem>")
def sample(stem):
    if not _safe_stem(stem):
        return jsonify({"error": "sample not found"}), 404
    f = REPO / "outputs" / f"analysis_{stem}.json"
    if not f.exists():
        return jsonify({"error": "sample not found"}), 404
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"结果文件损坏或不可读：{type(exc).__name__}"}), 500'''),

    ("web/app.py",
     '''    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "steps": [], "result": None, "error": None}''',
     '''    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "steps": [], "result": None, "error": None}
    _prune_jobs()'''),

    ("web/app.py",
     '''@app.get("/api/job/<job_id>")
def job(job_id):
    with JOBS_LOCK:
        snap = {
            "status": JOBS[job_id]["status"],
            "steps": list(JOBS[job_id]["steps"]),
            "result": JOBS[job_id]["result"],
            "error": JOBS[job_id]["error"],
        }
    return jsonify(snap)''',
     '''@app.get("/api/job/<job_id>")
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
    return jsonify(snap)'''),

    # ---------------- web/review_app.py ----------------
    ("web/review_app.py",
     '''def load_analysis(stem: str) -> dict | None:
    """读 outputs/analysis_<stem>.json，补上中文风险档位与定位图缩略。"""
    f = OUTPUTS / f"analysis_{stem}.json"''',
     '''_STEM_RE = re.compile(r"^[A-Za-z0-9_.\\-]{1,128}$")


def _safe_stem(stem) -> bool:
    """只允许字母 / 数字 / 下划线 / 点 / 横线（防目录穿越，纵深防御）。

    放在 load_analysis 里即可一次覆盖本文件所有读结果端点
    （/api/review/<stem>、/api/review POST、队列相关接口）。
    """
    return bool(_STEM_RE.match(stem or ""))


def load_analysis(stem: str) -> dict | None:
    """读 outputs/analysis_<stem>.json，补上中文风险档位与定位图缩略。"""
    if not _safe_stem(stem):
        return None
    f = OUTPUTS / f"analysis_{stem}.json"'''),

    ("web/review_app.py",
     '''def report(stem):
    """导出该图 PDF（复用 tools/export_pdf.py），失败则回退 Markdown 预览。"""
    md_path = REPORTS / f"report_{stem}.md"''',
     '''def report(stem):
    """导出该图 PDF（复用 tools/export_pdf.py），失败则回退 Markdown 预览。"""
    if not _safe_stem(stem):
        return jsonify({"error": "未找到该图的报告"}), 404
    err = None   # 显式初始化：原写法 `err if 'err' in dir() else ''` 靠运行时自省探测变量是否存在
    md_path = REPORTS / f"report_{stem}.md"'''),

    ("web/review_app.py",
     '''    return Response(
        text,
        mimetype="text/markdown; charset=utf-8",
        headers={"X-Pdf-Export": "failed", "X-Pdf-Error": (err if 'err' in dir() else "")},
    )''',
     '''    return Response(
        text,
        mimetype="text/markdown; charset=utf-8",
        headers={"X-Pdf-Export": "failed", "X-Pdf-Error": (err or "")},
    )'''),
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
            assert n == 1, f"{rel} 命中 {n} 次：{old[:60]}"
            text = text.replace(old, new)
            print(f"[ok]   {rel}  ← {old[:56]}")
        p.write_text(text, encoding="utf-8")

    # review_app.py 需要 re
    ra = ROOT / "web" / "review_app.py"
    t = ra.read_text(encoding="utf-8")
    if "\nimport re\n" not in t:
        t = t.replace("import os\nimport sys", "import os\nimport re\nimport sys", 1)
        ra.write_text(t, encoding="utf-8")
        print("[ok]   web/review_app.py  补 import re")


if __name__ == "__main__":
    main()

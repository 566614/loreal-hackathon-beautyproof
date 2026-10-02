# -*- coding: utf-8 -*-
"""种草内容核验 · 可交互 Agent（独立 Flask 应用，端口 5001）

为什么独立一个 app
------------------
原来的 web/app.py 是图导向（拖一张图进去判图）。这个页面把入口换成"先贴文案、再选配图"，
对应赛题点名的三个场景里的第一个：种草内容核验。
独立跑 5001，不碰 app.py、不碰 pipeline 核心，避免影响早鸟版本。

启动： python web/content_app.py  →  http://127.0.0.1:5001
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

app = Flask(__name__, template_folder=str(REPO / "web" / "templates"), static_folder=None)
# 本文件在 web/ 下，Flask 的 __name__ root_path 会算成 web/ → 模板目录会拼成 web/web/templates。
# 直接挂 FileSystemLoader 绕开它。
from jinja2 import FileSystemLoader  # noqa: E402
app.jinja_loader = FileSystemLoader(str(REPO / "web" / "templates"))


def _read(p: Path):
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _samples() -> list[dict]:
    mf = _read(REPO / "data" / "manifest.json")
    out = []
    for s in mf.get("samples", []):
        stem = Path(s.get("path", "")).stem
        if not stem:
            continue
        if not (REPO / "outputs" / f"analysis_{stem}.json").exists():
            continue
        out.append({"id": s.get("id") or stem, "stem": stem,
                    "img": str(s.get("path", "")).replace("\\", "/"),
                    "label": s.get("ground_truth", "")})
    return out


def _post_texts() -> list[dict]:
    out = []
    for p in _read(REPO / "data" / "xhs_posts.json").get("posts", []):
        out.append({"id": p.get("id"), "src": p.get("platform", ""),
                    "head": p.get("text", "").strip()[:24], "text": p.get("text", "")})
    return out


def _case_texts() -> list[dict]:
    out = []
    for c in _read(REPO / "data" / "text_cases.json").get("cases", []):
        out.append({"id": c.get("id"), "src": "自造评测文案",
                    "head": c.get("text", "").strip()[:24], "text": c.get("text", "")})
    return out


@app.get("/")
def index():
    return render_template("content.html", samples=_samples(),
                           posts=_post_texts(), cases=_case_texts())


@app.post("/check")
def check():
    payload = request.get_json(force=True, silent=True) or {}
    text = (payload.get("text") or "").strip()
    img = (payload.get("image") or "").strip()
    if not text or not img:
        return jsonify({"error": "文案和配图都要有"}), 400
    image_path = Path(img.replace("\\", "/"))
    if not image_path.exists():
        return jsonify({"error": f"找不到配图：{img}"}), 400
    try:
        import content_check
        res = content_check.check_one(text, image_path, tag=payload.get("tag", "web"))
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

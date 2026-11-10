# -*- coding: utf-8 -*-
"""Local content workbench using the shared BeautyProof audit."""
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
        recorded = str(s.get("path", "")).replace("\\", "/")
        _, sep, suffix = recorded.rpartition("/data/")
        if not sep and recorded.startswith("data/"):
            suffix = recorded[5:]
        if not suffix:
            continue
        path = (REPO / "data" / suffix).resolve()
        if path.is_relative_to(REPO / "data") and path.is_file():
            out.append({"id": s.get("id") or stem, "stem": stem,
                        "img": path.relative_to(REPO).as_posix(), "label": s.get("ground_truth", "")})
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
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "需要 JSON 对象"}), 400
    text, img, product = (payload.get(key) or "" for key in ("text", "image", "product"))
    if not all(isinstance(field, str) for field in (text, img, product)):
        return jsonify({"error": "text、image、product 应为文本"}), 400
    if not text.strip() and not img:
        return jsonify({"error": "请提供图片或正文"}), 400
    if img and img not in {item["img"] for item in _samples()}:
        return jsonify({"error": "图片不在可用示例清单中，请从上传入口选择自有图片"}), 400
    try:
        import content_check
        res = content_check.check_one(text, REPO / img if img else None, tag="web",
                                      product=product, references=payload.get("references"),
                                      llm=payload.get("llm") is True)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)

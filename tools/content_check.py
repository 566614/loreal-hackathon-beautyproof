# -*- coding: utf-8 -*-
# 原创工具 —— 赛题「种草内容核验」场景：把一段种草文案 + 一张配图丢进来，出核验结论
"""CLI and workbench adapter for the shared BeautyProof audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def _load_posts(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("posts", [])


def check_one(text: str = "", image_path: Path | None = None, tag: str = "inline",
              *, product=None, references=None, llm=False) -> dict:
    """Adapt the shared audit result for the existing content workbench."""
    import pipeline

    result = pipeline.analyze(image_path, text=text, product=product, references=references,
                              llm=llm, verbose=False, publish=False)
    evidence = result["evidence"]
    text_ev = (evidence.get("text", {}).get("evidence") or [{}])[0]
    cross_ev = (evidence.get("crossmodal", {}).get("evidence") or [{}])[0]
    findings = text_ev.get("claim_violations", [])
    conflicts = cross_ev.get("contradictions", [])
    verdict = result["verdict"]
    image = result["image"] or {}
    return {
        "tag": tag, "run_id": result["run_id"],
        "image_stem": image.get("stem"), "image_path": image.get("path"),
        "image_sha256": image.get("sha256"), "text": result["text_input"],
        "image_side_ready": any(result["audit"]["checks"][k]["status"] == "completed"
                                for k, _, _ in pipeline.TOOL_LIST),
        "aigc_score": result["audit"]["image_signals"]["origin"]["score"],
        "verdict": verdict["risk_level"], "headline": verdict["explain"]["headline"],
        "claims_high": [f for f in findings if f["severity"] == "high"],
        "claims_medium": [f for f in findings if f["severity"] == "medium"],
        "contradictions_hard": [c for c in conflicts if c["severity"] == "hard"],
        "contradictions_medium": [c for c in conflicts if c["severity"] == "medium"],
        "negated": cross_ev.get("negated_markers", []),
        "text_style": text_ev.get("ai_style", {}),
        "crossmodal_evidence": evidence.get("crossmodal", {}),
        "cannot_prove": [item["cannot_prove"] for item in evidence.values() if "cannot_prove" in item],
        "evidence": list(evidence.values()),
        "audit": result["audit"],
        "output_file": result["_output_file"],
    }


def _fmt(r: dict) -> str:
    lines = [
        f"【{r['tag']}】结论：{r['verdict']} —— {r['headline']}",
        f"  图像侧证据：{'已就绪' if r['image_side_ready'] else '未就绪'} | AIGC：{r['aigc_score']}",
    ]
    for e in r["evidence"]:
        lines.append(f"  · {e['tool']}: {e['observed']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="种草内容核验：文案 + 配图 → 核验结论")
    ap.add_argument("--text", default="", help="直接贴一段种草文案")
    ap.add_argument("--post", default="", help="用 data/xhs_posts.json 里的样本 id（如 xs_02）")
    ap.add_argument("--posts", default="", help="批量：读取某个真实文案集 json")
    ap.add_argument("--image", help="配图路径")
    ap.add_argument("--product", help="明确对应的产品名称")
    ap.add_argument("--references", type=Path, help="功效依据 JSON 数组，含 source、excerpt、可选 product")
    ap.add_argument("--llm", action="store_true", help="启用本地模型辅助抽取宣称")
    args = ap.parse_args()

    image = Path(args.image) if args.image else None
    if image and not image.is_file():
        print(f"[content-check] 找不到图片：{image}", file=sys.stderr)
        return 1

    jobs: list[tuple[str, str]] = []  # (tag, text)
    if args.text:
        jobs.append(("inline", args.text))
    elif args.post or args.posts:
        posts = _load_posts(REPO / "data" / "xhs_posts.json" if not args.posts else Path(args.posts))
        wanted = [p for p in posts if p.get("id") == args.post] if args.post else posts
        if not wanted:
            print(f"[content-check] 没找到样本：{args.post}", file=sys.stderr)
            return 1
        for p in wanted:
            jobs.append((p["id"], p.get("text", "")))
    elif image:
        jobs.append(("inline", ""))
    else:
        ap.error("至少提供 --text、--post、--posts 或 --image")

    references = json.loads(args.references.read_text(encoding="utf-8")) if args.references else None
    for tag, text in jobs:
        res = check_one(text, image, tag, product=args.product, references=references, llm=args.llm)
        print(_fmt(res))
        print(f"  结构化报告：{res['output_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
# 原创工具 —— 赛题「种草内容核验」场景：把一段种草文案 + 一张配图丢进来，出核验结论
"""
为什么要有这个
--------------
赛题说明页点名了三个场景：① 种草内容核验 ② 评论区真实性核验 ③ AI生成视觉素材鉴伪。
原来我们的流水线是「先有图、再判图」的图导向；这个工具把顺序反过来 ——
**先有文案，再把文案和配图对着看**，输出「这段内容有没有问题、问题在哪、依据是什么」。

它复用的都是已经验证过的模块，不重新造轮子：
    text_tool.build_evidence   文本侧：违禁宣称（有法律出处）+ 写作特征
    crossmodal_tool.check      图文侧：文案说的 vs 图上证据，硬/中矛盾
    crossmodal_tool.build_evidence  把矛盾整理成统一证据结构

用法
----
    # 指定样本
    python tools/content_check.py --post xs_02 --image data/clean/clean_01.png
    # 直接贴文案
    python tools/content_check.py --text "这款精华我用了两周能治疗痘痘肌" --image data/ai/ai_17.png
    # 批量：把 data/xhs_posts.json 里每条文案都拿去跟同一张配图对质
    python tools/content_check.py --posts data/xhs_posts.json --image data/clean/clean_01.png

产物：outputs/content_check_<post_id 或 inline>_<image_stem>.json

边界（不吹牛）
--------------
- 文本侧只认**词典能命中**的硬规则（违禁宣称/绝对化用语/量纲数字），不做模糊语义理解；
- 图文侧只比对「硬矛盾」（说实拍却是 AI 图、说没修图却检篡改）+「带量纲数字」；
- 没有文案、或图像侧证据没跑过 → 明确说"无法比对"，**绝不硬凑结论**；
- 输出的是**核验线索**，不是法律结论，高风险仍要人工复核。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def _load_posts(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("posts", data if isinstance(data, list) else [])


def check_one(text: str, image_path: Path, tag: str) -> dict:
    """对一条「文案 + 配图」做内容级核验，返回结构化结论（不写盘由调用方决定）。"""
    import crossmodal_tool
    import text_tool

    stem = image_path.stem

    # ---- 图像侧证据（outputs/analysis_<stem>.json），缺了就老实说缺
    img_ev = crossmodal_tool.load_image_evidence(REPO, stem)
    has_img_side = bool(img_ev)
    aigc_score = None
    _a = img_ev.get("aigc") if isinstance(img_ev, dict) else None
    if isinstance(_a, dict):
        # 分数实际挂在 aigc.evidence[0].aigc_score（见 aigc_tool.build_evidence 产出）
        for e in _a.get("evidence") or []:
            if isinstance(e, dict) and "aigc_score" in e:
                try:
                    aigc_score = float(e["aigc_score"])
                    break
                except (TypeError, ValueError):
                    continue
        for k in ("score", "aigc_score", "value", "prob"):
            if k in _a:
                try:
                    aigc_score = float(_a[k])
                    break
                except (TypeError, ValueError):
                    continue

    # ---- 文本侧
    text_ev = text_tool.build_evidence(text, source_id=stem, mode="single")
    claims = text_ev.get("claims") or []
    style = text_ev.get("style") or {}

    # ---- 图文交叉（需要图像侧证据；没有就跳过并标注）
    contradictions: list[dict] = []
    cross_ev: dict = {}
    if has_img_side:
        try:
            contradictions, negated = crossmodal_tool.check(text, img_ev)
        except Exception as exc:  # 交叉验证失败不能拖垮整份报告
            contradictions, negated = [], []
            cross_ev = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            cross_ev = crossmodal_tool.build_evidence(text, contradictions, img_ev, stem, negated=negated)
    else:
        negated = []

    hard = [c for c in contradictions if c.get("severity") == "hard"]
    medium = [c for c in contradictions if c.get("severity") == "medium"]

    high_claims = [c for c in claims if c.get("level") == "high" or c.get("severity") == "high"]
    mid_claims = [c for c in claims if c.get("level") == "medium" or c.get("severity") == "medium"]

    # ---- 人话结论（四档，与图像侧保持一致口径：不硬下"造假"结论）
    if hard:
        verdict, headline = "suspicious", f"文案与配图出现 {len(hard)} 处硬矛盾，建议人工复核（不直接定性）"
    elif high_claims or mid_claims:
        verdict, headline = "suspicious", f"文案命中 {len(high_claims)} 条高风险 + {len(mid_claims)} 条中风险表述，属合规风险而非真伪风险"
    elif contradictions:
        verdict, headline = "inconclusive", "有轻微不一致，需结合其他证据"
    else:
        verdict, headline = "credible", "未发现硬性风险线索，整体可信"

    return {
        "tag": tag,
        "image_stem": stem,
        "image_path": str(image_path),
        "text": text,
        "image_side_ready": has_img_side,
        "aigc_score": aigc_score,
        "verdict": verdict,
        "headline": headline,
        "claims_high": high_claims,
        "claims_medium": mid_claims,
        "contradictions_hard": hard,
        "contradictions_medium": medium,
        "negated": negated,
        "text_style": style,
        "crossmodal_evidence": cross_ev,
        "cannot_prove": [
            "文本侧不做模糊语义理解，只认词典能命中硬规则",
            "图像侧证据未就绪时不回答图文问题（不硬凑结论）",
            "本结论不是司法鉴定，高风险必须人工复核",
        ],
        "evidence": [
            {"tool": "text", "observed": f"命中高风险宣称 {len(high_claims)} 条 / 中风险 {len(mid_claims)} 条"},
            {"tool": "crossmodal", "observed": f"硬矛盾 {len(hard)} 处 / 中矛盾 {len(medium)} 处"},
            {"tool": "image_side", "observed": (
                f"AIGC 分数 {aigc_score}"
                if aigc_score is not None
                else ("图像侧证据已加载但未暴露 AIGC 分数" if has_img_side
                      else "图像侧证据未就绪（请先跑 pipeline 生成 outputs/analysis_<stem>.json）")
            )},
        ],
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
    ap.add_argument("--image", required=True, help="配图路径")
    args = ap.parse_args()

    image = Path(args.image)
    if not image.exists():
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
    else:
        print("[content-check] 至少要给 --text / --post / --posts 其中之一", file=sys.stderr)
        return 1

    out_dir = REPO / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for tag, text in jobs:
        if not text.strip():
            continue
        res = check_one(text, image, tag)
        results.append(res)
        print(_fmt(res))

    if results:
        stem = image.stem
        for r in results:
            f = out_dir / f"content_check_{r['tag']}_{stem}.json"
            f.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(results) > 1:
            print(f"\n[content-check] 批量完成 {len(results)} 条；单份结果已落 outputs/content_check_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

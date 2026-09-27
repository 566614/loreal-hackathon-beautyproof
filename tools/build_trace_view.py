# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队「核验过程」可视化（决策闭环可追溯），核心创新贡献
"""
核验过程可视化 —— 把 decision_trace json 渲染成一份自包含、可离线打开的 HTML 页

为什么要有这一页：
    赛题 2 明确要求"核验过程清晰"。规则引擎定完级、给出处置建议之后，
    评委 / 品牌方要看的不是最终一个结论，而是"你是怎么一步步查到这个结论的"。
    这一页把 decision_trace（工具 → 证据 → 档位 → 处置建议）画成时间线 + 工具卡，
    每句话都能点回对应的 outputs/<tool>_<图>.json 证据文件。

用法：
    python tools/build_trace_view.py outputs/decision_trace_<图名>.json
    python tools/build_trace_view.py <图名>            # 自动找 outputs/decision_trace_<图名>.json
    python tools/build_trace_view.py <trace.json> --open   # 生成后尝试用默认浏览器打开

产出：
    reports/trace_<图名>.html   内联 CSS、零外部依赖，双击即可离线查看
"""
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "outputs"
REP_DIR = REPO / "reports"

# 风险档位 → 中文标签 + 配色（红/橙/绿/灰 一眼区分）
RISK_META = {
    "high_risk": {"label": "高风险", "color": "#c0392b"},
    "suspicious": {"label": "可疑", "color": "#e67e22"},
    "credible": {"label": "可信", "color": "#27ae60"},
    "inconclusive": {"label": "无法判定", "color": "#7f8c8d"},
}

# 工具名 → 中文标签（自包含一份，避免依赖 pipeline）
TOOL_LABELS = {
    "hash": "文件指纹", "c2pa": "内容凭证", "ela": "压缩痕迹",
    "ocr": "文字识别", "aigc": "AI 生成检测", "trufor": "篡改痕迹检测",
    "text": "文案体检", "crossmodal": "图文交叉验证", "roundtable": "多 Agent 圆桌复核",
}


def _esc(text):
    return html.escape(str(text if text is not None else ""))


def _load_evidence(stem, tool):
    """尽量读回该工具的原始证据文件，用于卡片下方展开细节（失败就返回 None）。"""
    p = OUT_DIR / f"{tool}_{stem}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def build_html(trace):
    """把 decision_trace dict 渲染成自包含 HTML 字符串。

    trace 字段（见 pipeline.build_decision_trace）：
        image_name, generated_at, risk_level, reasons, reason,
        action_playbook{description/time_limit/owner_role/must_human_review}, trace[]
    """
    stem = trace.get("image_stem", "")
    image_name = trace.get("image_name", stem)
    risk_level = trace.get("risk_level", "inconclusive")
    rmeta = RISK_META.get(risk_level, RISK_META["inconclusive"])
    ap = trace.get("action_playbook", {}) or {}
    reasons = trace.get("reasons", []) or []
    trace_items = trace.get("trace", []) or []

    # ---- 工具卡（核验时间线）----
    cards = []
    for i, item in enumerate(trace_items, 1):
        tool = item.get("tool", "?")
        label = TOOL_LABELS.get(tool, tool)
        observed = item.get("observed", "（无观察记录）")
        contributed = bool(item.get("contributed_to_risk_level"))
        contrib_badge = (
            '<span class="badge contrib">✓ 直接贡献档位</span>'
            if contributed else
            '<span class="badge aux">— 辅助证据（未推动档位）</span>'
        )
        ev_ref = item.get("evidence_ref", f"outputs/{tool}_{stem}.json")

        # 展开细节：读回原始证据，给复核者看 cannot_prove 与具体数值
        detail_rows = ""
        ev = _load_evidence(stem, tool)
        if ev:
            cannot = ev.get("cannot_prove")
            if cannot:
                detail_rows += (
                    f'<div class="cannot"><b>能力边界：</b>{_esc(cannot)}</div>')
            facts = ev.get("evidence") or []
            if facts:
                detail_rows += '<ul class="facts">'
                for fobj in facts[:6]:
                    if isinstance(fobj, dict):
                        inner = " · ".join(
                            f"{_esc(k)}: {_esc(v)}"
                            for k, v in fobj.items()
                            if k not in ("heatmap", "overlay"))
                        detail_rows += f"<li>{inner}</li>"
                detail_rows += "</ul>"

        cards.append(f"""
        <div class="card {'contrib' if contributed else 'aux'}">
          <div class="card-head">
            <span class="step">{i}</span>
            <span class="tool">{_esc(label)} <code>{_esc(tool)}</code></span>
            {contrib_badge}
          </div>
          <div class="observed">{_esc(observed)}</div>
          <div class="ref">证据文件：<code>{_esc(ev_ref)}</code></div>
          {detail_rows}
        </div>""")

    cards_html = "\n".join(cards) if cards else '<p class="muted">（本次无工具证据记录）</p>'

    reasons_html = "\n".join(f"<li>{_esc(r)}</li>" for r in reasons) or '<li class="muted">（无判定依据）</li>'

    human = "是（强制人工复核）" if ap.get("must_human_review") else "否（可按 playbook 自动处置）"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>内容核验过程 · {_esc(image_name)}</title>
<style>
  :root {{ --ink:#1f2933; --muted:#6b7280; --line:#e5e7eb; --bg:#f7f8fa; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
         color:var(--ink); background:var(--bg); line-height:1.65; }}
  .wrap {{ max-width:920px; margin:0 auto; padding:28px 20px 60px; }}
  header {{ border-bottom:3px solid {rmeta['color']}; padding-bottom:14px; margin-bottom:22px; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .sub {{ color:var(--muted); font-size:13px; }}
  .block {{ background:#fff; border:1px solid var(--line); border-radius:10px;
           padding:18px 20px; margin-bottom:22px; }}
  .block h2 {{ font-size:16px; margin:0 0 12px; }}
  .risk-badge {{ display:inline-block; color:#fff; background:{rmeta['color']};
                 padding:4px 14px; border-radius:20px; font-weight:600; font-size:15px; }}
  .reasons {{ margin:10px 0 0; padding-left:20px; }}
  .reasons li {{ margin:4px 0; }}
  .playbook {{ display:grid; grid-template-columns:120px 1fr; gap:8px 14px; }}
  .playbook .k {{ color:var(--muted); font-size:13px; }}
  .playbook .v {{ font-size:14px; }}
  .card {{ border:1px solid var(--line); border-left:5px solid #cbd5e1;
          border-radius:8px; padding:14px 16px; margin-bottom:12px; background:#fff; }}
  .card.contrib {{ border-left-color:{rmeta['color']}; }}
  .card-head {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px; }}
  .step {{ width:24px; height:24px; border-radius:50%; background:var(--ink); color:#fff;
          display:inline-flex; align-items:center; justify-content:center; font-size:13px; }}
  .tool {{ font-weight:600; }}
  .tool code, .ref code {{ background:#eef1f4; padding:1px 6px; border-radius:4px; font-size:12px; }}
  .badge {{ font-size:12px; padding:2px 8px; border-radius:12px; }}
  .badge.contrib {{ background:#fdecea; color:#c0392b; }}
  .badge.aux {{ background:#eef1f4; color:#6b7280; }}
  .observed {{ font-size:14px; }}
  .ref {{ font-size:12px; color:var(--muted); margin-top:4px; }}
  .cannot {{ font-size:12px; color:#8a6d3b; background:#fdf6e3; border-radius:6px;
             padding:8px 10px; margin-top:8px; }}
  .facts {{ margin:8px 0 0; padding-left:18px; font-size:12px; color:#374151; }}
  .muted {{ color:var(--muted); }}
  footer {{ color:var(--muted); font-size:12px; text-align:center; margin-top:30px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>内容核验过程 · {_esc(image_name)}</h1>
    <div class="sub">BeautyProof 内容核验 Agent · 决策闭环可追溯 · 生成于 {_esc(trace.get('generated_at', ''))}</div>
  </header>

  <div class="block">
    <h2>最终结论</h2>
    <span class="risk-badge">{_esc(rmeta['label'])}（{_esc(risk_level)}）</span>
    <ul class="reasons">{reasons_html}</ul>
  </div>

  <div class="block">
    <h2>处置建议（分级 playbook）</h2>
    <div class="playbook">
      <div class="k">处置动作</div><div class="v">{_esc(ap.get('disposition', '—'))}</div>
      <div class="k">建议时限</div><div class="v">{_esc(ap.get('time_limit', '—'))}</div>
      <div class="k">责任人角色</div><div class="v">{_esc(ap.get('owner_role', '—'))}</div>
      <div class="k">强制人工</div><div class="v">{_esc(human)}</div>
    </div>
  </div>

  <div class="block">
    <h2>核验时间线（工具 → 证据 → 档位）</h2>
    {cards_html}
  </div>

  <footer>
    本页由 tools/build_trace_view.py 生成，内联样式、可离线打开；证据文件见各卡片「证据文件」路径。
  </footer>
</div>
</body>
</html>"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python tools/build_trace_view.py <trace.json 或 图名> [--open]")
        return 1

    open_after = "--open" in argv
    arg = argv[0]

    # 参数既可以是 trace json 路径，也可以直接是图名（自动找）
    trace_path = Path(arg)
    if not trace_path.exists() and not arg.endswith(".json"):
        trace_path = OUT_DIR / f"decision_trace_{arg}.json"
    if not trace_path.exists():
        print(f"找不到 trace 文件：{trace_path}")
        return 1

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    html_text = build_html(trace)
    if not html_text.strip():
        print("生成的 HTML 为空，异常中止")
        return 1

    stem = trace.get("image_stem") or trace_path.stem.replace("decision_trace_", "")
    REP_DIR.mkdir(exist_ok=True)
    out_html = REP_DIR / f"trace_{stem}.html"
    out_html.write_text(html_text, encoding="utf-8")
    print(f"已生成核验过程页：{out_html}（{len(html_text)} 字节）")

    if open_after:
        import webbrowser
        webbrowser.open(str(out_html))
    return 0


if __name__ == "__main__":
    sys.exit(main())

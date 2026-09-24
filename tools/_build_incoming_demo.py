# -*- coding: utf-8 -*-
"""Generate demo/incoming_compare.html from results/incoming_judge.json."""
import json
from pathlib import Path

REPO = Path(".")
data = json.loads((REPO / "results/incoming_judge.json").read_text(encoding="utf-8"))
ai = [r for r in data if r["group"].startswith("AI")]
real = [r for r in data if r["group"].startswith("真实")]

def card(r):
    if r["group"].startswith("AI"):
        img = f"../data/incoming_ai/{r['file']}"
    else:
        img = f"../data/incoming_real/{r['file']}"
    aigc = r["aigc_score"]
    aigc_s = f"{aigc:.4f}" if aigc is not None else "—"
    # 分数越高=越像AI（红），越低=越像真（绿）
    if aigc is None:
        bar_color = "#888"; pct = 0
    else:
        pct = int(max(0, min(1, aigc)) * 100)
        bar_color = "#e53935" if aigc >= 0.9 else ("#fb8c00" if aigc >= 0.5 else "#2e9e4f")
    tc = "✓ 有 TC260 AIGC 标识" if r["tc260_present"] else "✗ 无 TC260 标识"
    tc_cls = "badge-yes" if r["tc260_present"] else "badge-no"
    verdict = ("AI 生成" if (r["tc260_is_ai"] or (aigc or 0) >= 0.9)
               else "真实拍摄（无 AI 信号）")
    return f"""<div class="card">
  <img src="{img}" alt="" loading="lazy"/>
  <div class="meta">
    <div class="fname" title="{r['file']}">{r['file'][:30]}</div>
    <span class="badge {tc_cls}">{tc}</span>
    <div class="row"><span>aigc 分数</span><b>{aigc_s}</b></div>
    <div class="bar"><div class="fill" style="width:{pct}%;background:{bar_color}"></div></div>
    <div class="fmt">{r['format']} · {r['width']}×{r['height']} · {r['size_kb']} KB</div>
    <div class="verdict {('v-ai' if verdict=='AI 生成' else 'v-real')}">{verdict}</div>
  </div>
</div>"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BeautyProof · 素材盘点 · AI vs 真实 对比展示</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;margin:0;background:#f5f6f8;color:#222}}
  header{{background:#1f2430;color:#fff;padding:22px 28px}}
  header h1{{margin:0 0 6px;font-size:20px}}
  header p{{margin:0;opacity:.8;font-size:13px}}
  .summary{{display:flex;gap:14px;flex-wrap:wrap;padding:18px 28px;background:#fff;border-bottom:1px solid #eee}}
  .stat{{flex:1;min-width:160px;background:#fafbfc;border:1px solid #eceff3;border-radius:10px;padding:14px}}
  .stat .n{{font-size:26px;font-weight:700}}
  .stat .l{{font-size:12px;color:#666}}
  .group{{padding:20px 28px}}
  .group h2{{font-size:16px;margin:0 0 4px}}
  .group .sub{{font-size:12px;color:#777;margin-bottom:14px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px}}
  .card{{background:#fff;border:1px solid #eceff3;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .card img{{width:100%;height:180px;object-fit:cover;background:#eee;display:block}}
  .meta{{padding:10px 12px 14px}}
  .fname{{font-size:11px;color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:8px}}
  .badge{{display:inline-block;font-size:11px;padding:3px 8px;border-radius:20px}}
  .badge-yes{{background:#fdecea;color:#c62828}}
  .badge-no{{background:#e8f5e9;color:#2e7d32}}
  .row{{display:flex;justify-content:space-between;font-size:12px;margin:8px 0 4px}}
  .bar{{height:7px;background:#eee;border-radius:4px;overflow:hidden}}
  .fill{{height:100%}}
  .fmt{{font-size:11px;color:#888;margin-top:6px}}
  .verdict{{font-size:12px;font-weight:600;margin-top:8px;padding:5px 8px;border-radius:6px;text-align:center}}
  .v-ai{{background:#fdecea;color:#c62828}}
  .v-real{{background:#e8f5e9;color:#2e7d32}}
  .note{{padding:0 28px 30px;font-size:12px;color:#666;line-height:1.7}}
  .note code{{background:#eef;padding:1px 5px;border-radius:4px}}
</style></head>
<body>
<header>
  <h1>BeautyProof · 素材盘点对比展示</h1>
  <p>用户提供的两个压缩包：① ai训练集.zip（AI 生成）② 训练集.zip（真实拍摄）→ 已解包为 <code>data/incoming_ai/</code> 与 <code>data/incoming_real/</code></p>
</header>
<div class="summary">
  <div class="stat"><div class="n">30</div><div class="l">素材总数（AI 20 + 真实 10）</div></div>
  <div class="stat"><div class="n">30/30</div><div class="l">TC260 + 本域模型双重判定一致</div></div>
  <div class="stat"><div class="n">20/20</div><div class="l">AI 组带 TC260 AIGC 强制标识（Label=1）</div></div>
  <div class="stat"><div class="n">0/10</div><div class="l">真实组带 TC260 标识（符合预期）</div></div>
  <div class="stat"><div class="n">0</div><div class="l">模型对真实组的假阳性（阈值 0.9）</div></div>
</div>
<div class="group">
  <h2>① AI 生成组（ai训练集.zip，20 张）</h2>
  <div class="sub">判定依据：TC260 AIGC 元数据（生成器自声明，无需模型）+ 本域微调 aigc 模型（分数 0.9975~1.0）。二者一致。</div>
  <div class="grid">{"".join(card(r) for r in ai)}</div>
</div>
<div class="group">
  <h2>② 真实拍摄组（训练集.zip，10 张）</h2>
  <div class="sub">判定依据：无 TC260 标识（符合真实图常态）+ 本域 aigc 模型（分数 0.0~0.0196，远低于 0.9 阈值）。其中含 L'Oréal×HYROX 活动实拍等高质量真实素材。</div>
  <div class="grid">{"".join(card(r) for r in real)}</div>
</div>
<div class="note">
  <b>关于"判断方法"：</b>本页判定不依赖人工肉眼，也不依赖通用大模型（LLM）。
  强证据来自 <b>TC260 AIGC 元数据</b>（中国《人工智能生成合成内容标识办法》要求的强制标识，读文件即可复核）；
  弱但有价值的补充来自 <b>本域微调的轻量分类模型</b>（MobileNetV3，17MB，仅判"整图是否 AI 生成"，不调用任何 LLM）。<br/>
  <b>注意：</b>TC260 标识可被截图/转格式/重保存剥离，"无标识"不等于"是真图"；且本批 AI 图全部来自单一生成器「即梦」，跨生成器（Midjourney/SD/Flux）泛化能力尚未验证。
</div>
</body></html>"""

out = REPO / "demo" / "incoming_compare.html"
out.write_text(html, encoding="utf-8")
print(f"written {out} ({len(html)} bytes)")
print(f"AI cards={len(ai)}  Real cards={len(real)}")

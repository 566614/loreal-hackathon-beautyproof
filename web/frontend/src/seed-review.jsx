import React, { useState } from "react";
import { Button, Tag } from "antd";
import { ArrowRightOutlined, CheckCircleOutlined, FileSearchOutlined, PictureOutlined } from "@ant-design/icons";
import "./seed-review.css";

const categories = { all: "全部线索", promotion: "推广", efficacy: "功效参数", suitability: "适配",
  comments: "评论导流", logic: "逻辑", image: "图片", credible: "可参考" };
const levels = { warning: ["待核对", "warning"], positive: ["有限可信", "success"], neutral: ["观察", "default"] };
const sourceNames = { body: "正文", image_text: "图中文字", comment: "评论", image: "画面观察" };

function EvidenceView({ asset, evidence }) {
  const [full, setFull] = useState(false);
  const box = evidence?.box;
  if (!asset) return <div className="seed-quote-only">{evidence?.text || "选择一条线索查看原文。"}</div>;
  const { width: w, height: h } = asset;
  const x = box ? box[0] * w : 0, y = box ? box[1] * h : 0;
  const bw = box ? (box[2] - box[0]) * w : w, bh = box ? (box[3] - box[1]) * h : h;
  const px = w * .025, py = Math.min(h * .012, 80);
  const view = full || !box ? `0 0 ${w} ${h}` : `${Math.max(0, x-px)} ${Math.max(0, y-py)} ${Math.min(w, bw+px*2)} ${Math.min(h, bh+py*2)}`;
  return <figure className="seed-viewer">
    <div className="seed-viewer-toolbar"><span><PictureOutlined /> {asset.kind || "送审图片"}</span>
      <Button size="small" onClick={() => setFull(v => !v)}>{full ? "聚焦证据" : "查看整张"}</Button></div>
    <div className={"seed-image-frame" + (full ? " seed-image-full" : "")}>
      <svg viewBox={view} role="img" aria-label={evidence ? "原图证据位置：" + evidence.text : asset.name}>
        <image href={asset.preview} width={w} height={h} />
        {box && <rect x={x} y={y} width={bw} height={bh} fill="#ad763015" stroke="#a65320"
          strokeWidth="2" vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>
    <figcaption>框选对应当前引用，表示文字或画面位置。</figcaption>
  </figure>;
}

export default function SeedReport({ result }) {
  const [category, setCategory] = useState("all");
  const [activeId, setActiveId] = useState(result.findings[0]?.id);
  const [evidenceId, setEvidenceId] = useState(null);
  const visible = result.findings.filter(f => category === "all" || f.category === category);
  const active = visible.find(f => f.id === activeId) || visible[0];
  const anchors = active ? active.evidence_ids.map(id => result.evidence.find(e => e.id === id)).filter(Boolean) : [];
  const evidence = anchors.find(e => e.id === evidenceId) || anchors[0];
  const asset = result.assets.find(a => a.id === evidence?.asset_id);
  const counts = result.source_counts;
  return <article className="review-document seed-document report-surface">
    <header className="report-titlebar"><div><FileSearchOutlined aria-hidden="true" /><h2>种草核验</h2></div>
      <Tag color="warning">{result.findings.some(f => f.category === "promotion") ? "留意推广与证据缺口" : "按线索核对"}</Tag></header>
    <section className="seed-summary"><h3>{result.product || result.title}</h3><p>{result.summary}</p>
      <div className="seed-source-strip">
        <span>{result.assets.length} 张图片</span><span>{counts.body || 0} 段正文</span>
        <span>{counts.image_text || 0} 段图片文字</span><span>{counts.comment || 0} 条可见评论线索</span>
      </div>
    </section>
    <nav className="seed-filters" aria-label="分析线索分类">
      {Object.entries(categories).filter(([key]) => key === "all" || result.findings.some(f => f.category === key))
        .map(([key, label]) => <button key={key} type="button" aria-pressed={category === key}
          onClick={() => { setCategory(key); setEvidenceId(null); }}>{label}</button>)}
    </nav>
    <div className="seed-findings" aria-label="分析结果">
      {visible.map(f => <button className={"seed-finding" + (active?.id === f.id ? " is-active" : "")}
        aria-pressed={active?.id === f.id} type="button" key={f.id}
        onClick={() => { setActiveId(f.id); setEvidenceId(null); }}>
        <span><Tag color={levels[f.level][1]}>{categories[f.category]}</Tag><strong>{f.title}</strong></span>
        <ArrowRightOutlined aria-hidden="true" />
      </button>)}
    </div>
    {active && <section className="seed-detail" aria-live="polite">
      <h3>{active.title}</h3><p>{active.analysis}</p>
      <div className="seed-anchor-tabs" aria-label="原文证据">
        {anchors.map((e, index) => <button type="button" key={e.id} aria-pressed={evidence?.id === e.id}
          onClick={() => setEvidenceId(e.id)}>{sourceNames[e.source]} {index+1}</button>)}
      </div>
      {evidence && <blockquote className="seed-source-quote">{evidence.text}</blockquote>}
      <EvidenceView key={evidence?.id || active.id} asset={asset} evidence={evidence} />
      <div className="seed-action"><ArrowRightOutlined /><p>{active.action}</p></div>
    </section>}
    {!result.findings.length && <p className="seed-empty">请补充清晰文字或正文、评论，继续核验。</p>}
    <section className="seed-next"><h3><CheckCircleOutlined /> 你可以怎么判断</h3>
      <ul>{result.next_steps.map(step => <li key={step}>{step}</li>)}</ul>
    </section>
    <details className="seed-references"><summary>查看知识库参考与使用边界</summary>
      <p>推广规则与社区语料用于提示线索。具体产品功效，等待品牌提供适用的试验资料。</p>
      {result.references.map(ref => <section key={ref.id}><strong>{ref.locator}</strong>
        <small>{ref.source} · {ref.kind === "community_example" ? "其他笔记的用语参考" : "编辑参考"}</small>
        <p>{ref.excerpt.slice(0, 450)}</p></section>)}
    </details>
  </article>;
}

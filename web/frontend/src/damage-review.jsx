import { Button, Space } from "antd";
import { ArrowRightOutlined, FileSearchOutlined } from "@ant-design/icons";
import React, { useRef, useState } from "react";

const SOURCE_LABELS = {
  declared_ai: "文件元数据声明为 AI 生成，来源仍可复核。",
  unverified_marker: "发现来源标识，等待核验。",
  not_found: "文件中未读到 AI 来源标识。",
};

const STATUS_LABELS = {
  similar: "发现相似生成线索", weak: "未见明显相似生成线索", insufficient: "证据不足",
};

function Assessment({ title, data, prominent = false }) {
  return <article className={"claim" + (prominent ? " damage-conclusion" : "")}>
    <h3>{prominent ? STATUS_LABELS[data.status] : title}</h3>
    {prominent ? <p className="damage-score">检测分 <strong>{data.score == null ? "—" : data.score.toFixed(3)}</strong>
      {data.threshold != null && <span>复核阈值 {data.threshold.toFixed(3)}</span>}</p> :
    <p><strong>{STATUS_LABELS[data.status]}</strong>{data.score != null && " · 检测分 " + data.score.toFixed(3)}
      {data.threshold != null && " · 复核阈值 " + data.threshold.toFixed(3)}</p>}
    <p>{data.reason}</p>
    {prominent && <p className="damage-score-note">检测分用于复核，尚未校准为生成概率。</p>}
    {data.cues?.length > 0 && <ul className="detail-list">{data.cues.map(cue =>
      <li key={cue.label}>{cue.label} · {cue.direction === "raises" ? "推高评分" : "拉低评分"}</li>
    )}</ul>}
  </article>;
}

function normalizeBox(start, end) {
  return [Math.min(start[0], end[0]), Math.min(start[1], end[1]),
    Math.max(start[0], end[0]), Math.max(start[1], end[1])];
}

function pointerPosition(event) {
  const { left, top, width, height } = event.currentTarget.getBoundingClientRect();
  const clamp = value => Math.min(1, Math.max(0, value));
  return [clamp((event.clientX - left) / width), clamp((event.clientY - top) / height)];
}

export function DamageRegion({ preview, roi, onChange, busy }) {
  const [selecting, setSelecting] = useState(false);
  const [draft, setDraft] = useState(null);
  const drag = useRef(null);
  const box = draft || roi;
  return <div className="damage-selection">
    <Space wrap>
      <Button disabled={busy} type={selecting ? "primary" : "default"}
        onClick={() => setSelecting(value => !value)} aria-pressed={selecting}>框选破损</Button>
      {roi && <Button disabled={busy} type="text" onClick={() => onChange(null)}>清除选区</Button>}
    </Space>
    {selecting && <p className="muted">在图片中拖动，圈出裂痕、碎粉或包装破口。</p>}
    {preview && (roi || selecting) && <div className={"damage-photo" + (selecting && !busy ? " damage-selecting" : "")}>
      <img src={preview} alt="待框选的送审图片" draggable={false} />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"
        onPointerDown={event => {
          if (!selecting || busy || drag.current || event.button !== 0) return;
          const start = pointerPosition(event);
          drag.current = { start, id: event.pointerId };
          event.currentTarget.setPointerCapture(event.pointerId);
          setDraft([...start, ...start]);
        }}
        onPointerMove={event => {
          if (drag.current?.id === event.pointerId) setDraft(normalizeBox(drag.current.start, pointerPosition(event)));
        }}
        onPointerUp={event => {
          if (drag.current?.id !== event.pointerId) return;
          const selected = normalizeBox(drag.current.start, pointerPosition(event));
          drag.current = null;
          setDraft(null);
          if (selected[2] - selected[0] >= 0.03 && selected[3] - selected[1] >= 0.03) {
            onChange(selected);
            setSelecting(false);
          }
        }}
        onLostPointerCapture={() => {
          if (drag.current) {
            drag.current = null;
            setDraft(null);
          }
        }}>
        {box && <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]}
          className="damage-roi" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>}
  </div>;
}

export function DamageReport({ result }) {
  const { preview, roi, edge_preview_png: overlay, whole, region } = result;
  const wholeRef = useRef(null), regionRef = useRef(null), imageRef = useRef(null), nextRef = useRef(null);
  const locate = ref => ref.current?.scrollIntoView({ block: "start", behavior: "smooth" });
  return <article className="review-document damage-document report-surface">
    <header className="report-titlebar"><div><FileSearchOutlined aria-hidden="true" /><h2>退款破损审核</h2></div></header>
    <div className="result-stack">
      {whole && <div ref={wholeRef}><Assessment title="整图" data={whole} prominent /></div>}
      <nav className="damage-directory" aria-label="破损线索目录"><h3>线索目录</h3>
        <button type="button" onClick={() => locate(wholeRef)}>整图检测依据 <ArrowRightOutlined /></button>
        {region && <button type="button" onClick={() => locate(regionRef)}>框选区域线索 <ArrowRightOutlined /></button>}
        {preview && <button type="button" onClick={() => locate(imageRef)}>送审原图与来源 <ArrowRightOutlined /></button>}
        <button type="button" onClick={() => locate(nextRef)}>人工复核建议 <ArrowRightOutlined /></button>
      </nav>
      {region && <div ref={regionRef}><Assessment title="选区" data={region} /></div>}
      <div ref={nextRef} className="report-next damage-anchor"><ArrowRightOutlined aria-hidden="true" />
        <p><b>下一步</b><span>结合原始图片和售后凭证人工复核。</span></p></div>
      {preview && <figure ref={imageRef} className="damage-edge damage-anchor">
        <div className="damage-photo">
          <img src={preview} alt="退款破损送审图片" />
          {overlay && roi && <img className="damage-edge-overlay" alt="选区候选边缘"
            src={"data:image/png;base64," + overlay} style={{
              left: (roi[0] * 100) + "%", top: (roi[1] * 100) + "%",
              width: ((roi[2] - roi[0]) * 100) + "%", height: ((roi[3] - roi[1]) * 100) + "%",
            }} />}
        </div>
        {overlay && roi && <figcaption>橙线是选区候选边缘，可能包含接缝、文字或反光。</figcaption>}
      </figure>}
      {SOURCE_LABELS[result.source_marker] && <p className="muted">{SOURCE_LABELS[result.source_marker]}</p>}
      <details className="evidence-detail">
        <summary>检查记录</summary>
        <p className="muted">{result.model ? result.model + " · CPU" : "本地 CPU 检查未加载破损模型。"}</p>
      </details>
    </div>
  </article>;
}

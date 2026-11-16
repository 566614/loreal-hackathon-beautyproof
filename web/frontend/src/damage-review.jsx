import { Alert, Button, Card, Space, Upload } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import React, { useEffect, useRef, useState } from "react";

const SOURCE_LABELS = {
  declared_ai: "文件元数据声明为 AI 生成，来源仍可复核。",
  unverified_marker: "发现来源标识，等待核验。",
  not_found: "文件中未读到 AI 来源标识。",
};

function Assessment({ title, data }) {
  if (!data) return null;
  const status = { similar: "疑似 AI 生成破损", weak: "未见明显相似线索", insufficient: "证据不足" }[data.status];
  return <article className="claim">
    <h3>{title}</h3>
    <p><strong>{status}</strong>{data.score != null && " · 检测分 " + data.score.toFixed(3)}
      {data.threshold != null && " · 复核阈值 " + data.threshold.toFixed(3)}</p>
    <p>{data.reason}</p>
    {data.cues.length > 0 && <ul>{data.cues.map(cue =>
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

export default function DamageReview({ readJson }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [selecting, setSelecting] = useState(false);
  const [roi, setRoi] = useState(null);
  const [draft, setDraft] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const drag = useRef(null);
  const request = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => request.current?.abort(), []);

  function changeFile(next) {
    setFile(next);
    setRoi(null);
    setDraft(null);
    setSelecting(false);
    setResult(null);
    setError("");
    drag.current = null;
  }

  async function analyze() {
    if (request.current || !file) return;
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setResult(null);
    setError("");
    const form = new FormData();
    form.set("file", file);
    if (roi) form.set("roi", JSON.stringify(roi));
    try {
      const data = await readJson("/api/damage-review", { method: "POST", body: form, signal: controller.signal });
      if (!controller.signal.aborted) setResult(data);
    } catch (err) {
      if (!controller.signal.aborted) setError(err.message);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      request.current = null;
    }
  }

  const box = draft || roi;
  const overlay = result?.edge_preview_png;
  return <>
    <div className="intro">
      <h1>退款破损审核</h1>
      <p>上传送审图查看生成线索，可框选裂痕、碎粉或包装破口。优先检出疑似图片，结果用于人工复核。</p>
    </div>
    <div className="workbench-grid damage-grid">
      <Card className="surface intake-surface" title="送审图片">
        <Upload.Dragger name="file" accept=".png,.jpg,.jpeg" maxCount={1} multiple={false}
          disabled={busy} fileList={file ? [file] : []} onRemove={() => { changeFile(null); return true; }}
          beforeUpload={picked => {
            if (!/\.(png|jpe?g)$/i.test(picked.name) || picked.size > 32 * 1024 * 1024) {
              setError("请选择不超过 32 MB 的 JPG 或 PNG 图片。");
              return Upload.LIST_IGNORE;
            }
            changeFile(picked);
            return false;
          }}>
          <span className="upload-icon"><UploadOutlined /></span>
          <strong>点击或拖入图片</strong>
          <small>JPG / PNG · 最大 32 MB</small>
        </Upload.Dragger>
        {preview && <div className="damage-selection">
          <Space wrap>
            <Button disabled={busy} type={selecting ? "primary" : "default"}
              onClick={() => setSelecting(value => !value)} aria-pressed={selecting}>框选破损</Button>
            {roi && <Button disabled={busy} onClick={() => { setRoi(null); setResult(null); }}>清除选区</Button>}
          </Space>
          {selecting && <p className="muted">在图片上拖动，画出需要检查的区域。</p>}
          <div className={"damage-photo" + (selecting && !busy ? " damage-selecting" : "")}>
            <img src={preview} alt="待审核商品图片" draggable={false} />
            <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"
              onPointerDown={event => {
                if (!selecting || busy || drag.current || event.button !== 0) return;
                const start = pointerPosition(event);
                drag.current = { start, id: event.pointerId };
                event.currentTarget.setPointerCapture(event.pointerId);
                setDraft([...start, ...start]);
              }}
              onPointerMove={event => {
                if (drag.current?.id !== event.pointerId) return;
                setDraft(normalizeBox(drag.current.start, pointerPosition(event)));
              }}
              onPointerUp={event => {
                if (drag.current?.id !== event.pointerId) return;
                const selected = normalizeBox(drag.current.start, pointerPosition(event));
                drag.current = null;
                setDraft(null);
                if (selected[2] - selected[0] >= 0.03 && selected[3] - selected[1] >= 0.03) {
                  setRoi(selected);
                  setResult(null);
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
                fill="rgba(15, 168, 163, 0.12)" stroke="#006e68" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
            </svg>
          </div>
        </div>}
        <Button className="damage-submit" type="primary" disabled={!file} loading={busy} onClick={analyze}>开始检查</Button>
        {error && <Alert className="damage-message" type="error" showIcon title={error} />}
      </Card>
      <Card className="surface" title="检查结果" aria-live="polite">
        {result ? <div className="result-stack">
          <Assessment title="整图" data={result.whole} />
          <Assessment title="选区" data={result.region} />
          {overlay && result.roi && <figure className="damage-edge">
            <div className="damage-photo">
              <img src={preview} alt="送审图片，已覆盖选区候选边缘" />
              <img className="damage-edge-overlay" alt="" src={"data:image/png;base64," + overlay}
                style={{
                  left: (result.roi[0] * 100) + "%",
                  top: (result.roi[1] * 100) + "%",
                  width: ((result.roi[2] - result.roi[0]) * 100) + "%",
                  height: ((result.roi[3] - result.roi[1]) * 100) + "%",
                }} />
            </div>
            <figcaption>橙线是选区候选边缘，可能包含接缝、文字或反光。</figcaption>
          </figure>}
          <p>{SOURCE_LABELS[result.source_marker]}</p>
          {result.model && <small className="muted">{result.model} · CPU</small>}
        </div> : <p className="muted">{busy ? "正在检查图片…" : "上传图片后开始检查。"}</p>}
      </Card>
    </div>
  </>;
}

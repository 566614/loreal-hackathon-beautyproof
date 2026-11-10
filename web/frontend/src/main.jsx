import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert, Button, Card, Checkbox, Collapse, ConfigProvider, Form, Image, Input,
  Progress, Select, Space, Tabs, Tag, Upload,
} from "antd";
import {
  ApartmentOutlined, ArrowRightOutlined, CopyOutlined, DownloadOutlined,
  FileTextOutlined, PictureOutlined, SafetyCertificateOutlined, ScanOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import zhCN from "antd/locale/zh_CN";
import "antd/dist/reset.css";
import "./workbench.css";

const toolLabels = {
  hash: "文件指纹", c2pa: "来源凭证", ela: "压缩痕迹", spectral: "频域取证",
  ocr: "图中文字", ai_label: "AI 来源标识", aigc: "生成倾向", trufor: "局部异常",
  text: "文案检查", crossmodal: "图文对照", claim_model: "本地文案模型",
};
const risks = {
  high_risk: ["高风险", "error"], suspicious: ["待复核", "warning"],
  credible: ["来源可核对", "success"], inconclusive: ["证据不足", "default"],
};
const states = {
  completed: ["已完成", "success"], insufficient_evidence: ["证据不足", "default"],
  model_unavailable: ["模型不可用", "warning"],
  running: ["进行中", "processing"], done: ["已完成", "success"],
  skipped: ["已跳过", "default"], failed: ["不可用", "warning"],
};
const claimColors = { "有证据支持": "success", "存在冲突": "error", "证据不足": "warning" };
const theme = {
  token: {
    colorPrimary: "#14736a", colorLink: "#14736a", colorText: "#172f2b", colorTextSecondary: "#4e6661",
    colorBorder: "#d3e0da", colorBgLayout: "#f3f7f4", borderRadius: 10,
    fontFamily: '"Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif',
    controlHeight: 40,
  },
};

async function readJson(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload || payload.error) {
    throw new Error(payload?.error || (response.status === 413 ? "文件太大，请选择不超过 32 MB 的图片。" : "无法读取审核服务的响应，请稍后重试。"));
  }
  return payload;
}

function AuditStatus({ value }) {
  const [label, color] = states[value] || ["待检查", "default"];
  return <Tag color={color}>{label}</Tag>;
}

function EmptyReport() {
  return <div className="empty-report">
    <div className="empty-index"><span className="status-dot" /> 等待送审</div>
    <div className="inspection" aria-hidden="true">
      <div className="inspection-inner"><ScanOutlined /></div>
      <span className="inspection-label">EVIDENCE VIEW</span>
    </div>
    <h2>核验报告</h2>
    <p>提交图片或正文后，查看引用依据、图像线索和待复核事项。</p>
    <div className="evidence-lanes">
      <div><PictureOutlined /> <span>图片来源</span><small>生成倾向与局部异常</small></div>
      <div><FileTextOutlined /> <span>宣传宣称</span><small>承诺原句与功效依据</small></div>
      <div><ApartmentOutlined /> <span>图文对照</span><small>可比指标与冲突点</small></div>
    </div>
  </div>;
}

function RunningReport({ steps }) {
  const latest = [...new Map(steps.map(step => [step.tool, step])).values()];
  const total = Math.max(0, ...latest.map(step => step.total || 0));
  const finished = latest.filter(step => ["done", "failed", "skipped"].includes(step.state)).length;
  return <Card className="surface report-surface" title={<span>正在核验</span>}>
    <div role="status" aria-live="polite">
      <Progress percent={total ? Math.min(100, Math.round(finished / total * 100)) : 0} status="active" />
      <p className="muted">{latest.length ? "正在汇总本次检查记录。" : "正在准备送审内容…"}</p>
    </div>
    <ul className="check-list">
      {latest.map(step => <li key={step.tool}><span>{toolLabels[step.tool] || step.tool}</span><AuditStatus value={step.state} /></li>)}
    </ul>
  </Card>;
}

function ImageRegions({ image, regions }) {
  if (!image?.preview || !image.width || !image.height || !regions?.length) return null;
  return <details className="evidence-detail">
    <summary>查看图片中的位置</summary>
    <svg role="img" aria-label="原图与文字证据位置" viewBox={`0 0 ${image.width} ${image.height}`}>
      <image href={image.preview} width={image.width} height={image.height} />
      {regions.map(({ bbox: [x1, y1, x2, y2] }, i) =>
        <rect key={i} x={x1} y={y1} width={x2 - x1} height={y2 - y1}
          fill="none" stroke="#ec6044" strokeWidth="3" vectorEffect="non-scaling-stroke" />)}
    </svg>
  </details>;
}

function ClaimResults({ result }) {
  const claims = result.audit?.claims || [];
  const reasons = result.verdict.reasons || [];
  return <div className="result-stack">
    {result.audit?.ai_illustration_disclosed && <Alert type="info" showIcon title="素材已标注为 AI 示意" />}
    {claims.length ? claims.map((claim, index) => <article className="claim" key={index}>
      <div className="claim-heading">
        <Tag color={claimColors[claim.status]}>{claim.status}</Tag>
        <span>{claim.source === "image" ? "图片原句" : "正文原句"}</span>
      </div>
      <blockquote><mark>{claim.quote}</mark></blockquote>
      <ImageRegions image={result.image} regions={claim.regions} />
      {claim.references?.map((reference, i) => <details className="evidence-detail" key={i}>
        <summary>{reference.source} · {reference.verification === "reviewer_verified"
          ? "已核验来源" : reference.matches ? "摘录内容一致，来源待核验" : "摘录内容不一致，来源待核验"}</summary>
        <blockquote>{reference.quote}</blockquote>
        {reference.version && <p>版本 {reference.version} · {reference.locator} · 审核人 {reference.reviewer}</p>}
      </details>)}
      {claim.conflicts?.map((conflict, i) => <p key={i}>冲突　{conflict.text_side} / {conflict.image_side}</p>)}
      {!!claim.missing?.length && <p className="muted">待补　{claim.missing.join("、")}</p>}
    </article>) : <p className="tab-empty">本次报告没有可逐条核对的宣称。</p>}
    {!!reasons.length && <Collapse items={[{ key: "reasons", label: "查看综合判定依据", children: <ul className="detail-list">{reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul> }]} />}
  </div>;
}

function CrossmodalResults({ result }) {
  const cross = result.audit?.crossmodal || result.evidence?.crossmodal?.evidence?.[0] || {};
  if (!result.image || !result.text_input) {
    return <p className="tab-empty">上传图片并填写正文后可比对图文。</p>;
  }
  const findings = (cross.contradictions || []).filter(item => !item.body_claim);
  const alignment = cross.alignment || (cross.contradictions || []).filter(item => item.body_claim)
    .map(item => ({ body: item.body_claim, image: item.image_claim, status: "数值冲突" }));
  return <div className="result-stack">
    {cross.ai_illustration_disclosed && <Alert type="info" showIcon title="素材已声明为 AI 示意" />}
    {!!cross.visual_observation?.regions?.length && <article className="claim">
      <Tag>{cross.visual_observation.observation}</Tag>
      <ImageRegions image={result.image} regions={cross.visual_observation.regions} />
    </article>}
    {findings.map((item, i) => <article className="claim" key={i}>
      <Tag color="orange">{item.type}</Tag>
      <blockquote>正文 · {item.text_side}</blockquote>
      <blockquote>图片 · {item.image_side}</blockquote>
      <p>{item.why}</p>
      <ImageRegions image={result.image} regions={item.regions} />
    </article>)}
    {alignment.map((row, i) => <article className="claim" key={i}>
      <Tag color={row.status === "数值冲突" ? "orange" : undefined}>{row.status}</Tag>
      <blockquote>正文 · {row.body.quote}</blockquote>
      {row.image && <blockquote>图片 · {row.image.quote}</blockquote>}
      {row.missing && <p>{row.missing}</p>}
      <ImageRegions image={result.image} regions={row.image?.regions} />
    </article>)}
    {!alignment.length && !findings.length && <p className="tab-empty">本次未找到可对齐的图文宣称，请核对产品、指标与图片中的文字。</p>}
  </div>;
}

function ImageResults({ result }) {
  const signal = result.audit?.image_signals;
  const evidence = result.evidence || {};
  const origin = signal?.origin || { score: evidence.aigc?.evidence?.[0]?.aigc_score };
  const editing = signal?.editing || { score: evidence.trufor?.evidence?.[0]?.trufor_score };
  const heatmap = result.visuals?.overlay || result.visuals?.heatmap;
  return <div className="result-stack">
    {result.image ? <>
      <div className="score-row">
        <div><span>生成倾向分</span><b>{origin.score ?? "—"}</b><small>{[origin.model, origin.status || "待复核线索"].filter(Boolean).join(" · ")}</small></div>
        <div><span>局部异常分</span><b>{editing.score ?? "—"}</b><small>{[editing.model, editing.region || "区域线索需对照原图"].filter(Boolean).join(" · ")}</small></div>
      </div>
      {result.audit?.ai_illustration_disclosed && <Alert type="info" showIcon title="素材已声明为 AI 示意" />}
      {origin.verified_ai ? <p>已验证来源凭证声明素材由 AI 生成。</p>
        : origin.visible_ai_label && <p>图中有可见 AI 标识，需核对原始来源。</p>}
      <p className="muted">局部热力图提示异常位置。真实上妆、美颜和实际修图需结合原片或处理记录复核。</p>
      {(result.image.preview || heatmap) && <div className="image-grid">
        <Image.PreviewGroup>
          {[[result.image.preview, "送审原图"], [heatmap, "可疑区域"]].filter(([src]) => src).map(([src, label]) =>
            <figure key={label}><Image src={src} alt={label} width="100%" /><figcaption>{label}</figcaption></figure>)}
        </Image.PreviewGroup>
      </div>}
      {result.image.sha256 && <p className="file-hash">SHA-256 <code>{result.image.sha256}</code></p>}
    </> : <p className="tab-empty">本次仅审核正文。</p>}
  </div>;
}

function EvidenceResults({ result }) {
  const evidence = result.evidence || {};
  const checks = result.audit?.checks || {};
  const keys = [...new Set([...Object.keys(checks), ...Object.keys(evidence)])];
  const actions = [...(result.agent?.decisions || []), ...(result.agent?.actions || [])];
  return <div className="result-stack">
    <Collapse items={keys.map(key => ({
      key,
      label: result.tool_meta?.[key]?.label || toolLabels[key] || key,
      extra: <AuditStatus value={checks[key]?.decision_status || checks[key]?.status ||
        (evidence[key]?.evidence?.[0]?.available === false ? "model_unavailable" : "completed")} />,
      children: <>
        <p>{evidence[key]?.observed || "本次没有可引用的检查结果。"}</p>
        {evidence[key]?.cannot_prove && <p className="muted">{evidence[key].cannot_prove}</p>}
      </>,
    }))} />
    {!!actions.length && <Collapse items={[{ key: "agent", label: "查看 Agent 决策记录", children: <ul className="detail-list">
      {actions.map((action, i) => <li key={i}><b>{toolLabels[action.tool] || action.tool || action.action}</b>　{action.reason}</li>)}
    </ul> }]} />}
    {result.llm_explanation && <p><b>本地模型辅助解读</b><br />{result.llm_explanation}</p>}
  </div>;
}

function reportText(result) {
  const { image, verdict, audit, evidence = {}, agent } = result;
  const explanation = verdict.explain || {};
  const label = (risks[verdict.risk_level] || risks.inconclusive)[0];
  return [
    "# BeautyProof 美妆宣传核验报告", "",
    "## 核验结论", "", "**" + label + "**", explanation.headline, explanation.summary,
    explanation.what_to_do && "下一步　" + explanation.what_to_do,
    explanation.caveat, "",
    "## 送审信息", "", image?.name || "仅正文", image?.sha256 && "SHA-256 " + image.sha256, "",
    "## 判定依据", "", ...(verdict.reasons || []).map(reason => "- " + reason), "",
    "## 宣称核验", "", ...(audit?.claims || []).flatMap(claim => [
      "- " + claim.status + "｜" + claim.quote,
      ...(claim.references || []).map(ref => "  - " + ref.source + "｜" + ref.quote),
      ...(claim.conflicts || []).map(c => "  - 冲突｜" + c.text_side + " / " + c.image_side),
      ...(claim.missing || []).map(missing => "  - 待补 " + missing),
    ]), "",
    "## 取证记录", "", ...Object.entries(evidence).flatMap(([key, entry]) => [
      "### " + (toolLabels[key] || key), entry.observed, entry.cannot_prove, "",
    ]), ...(agent ? ["## Agent 决策过程", "", agent.summary,
      ...[...(agent.decisions || []), ...(agent.actions || [])].map(row => "- " + row.action + "｜" + row.reason),
    ] : []),
  ].filter(line => line != null && line !== false).join("\n");
}

function FinishedReport({ result, isSample }) {
  const [copy, setCopy] = useState("");
  const verdict = result.verdict;
  const [label, color] = risks[verdict.risk_level] || risks.inconclusive;
  const explanation = verdict.explain || {};
  const hasClaims = !!result.audit?.claims?.length;
  function download() {
    const url = URL.createObjectURL(new Blob([reportText(result)], { type: "text/markdown;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = (result.image?.name?.replace(/\.[^.]+$/, "") || "beautyproof_report") + ".md";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function copySummary() {
    try {
      await navigator.clipboard.writeText([label, explanation.headline, explanation.summary].filter(Boolean).join("\n"));
      setCopy("已复制");
    } catch {
      setCopy("复制失败，可下载报告");
    }
  }
  return <Card className="surface report-surface" title="核验报告">
    <div className="report-heading">
      <Space size={6} wrap><Tag color={color}>{label}</Tag>{isSample && <Tag>预置样本报告</Tag>}</Space>
      <h2>{explanation.headline || label}</h2>
      {explanation.summary && <p>{explanation.summary}</p>}
      {explanation.what_to_do && <p className="next-action"><b>下一步</b>　{explanation.what_to_do}</p>}
      <Space size={8} wrap>
        <Button size="small" icon={<DownloadOutlined />} onClick={download}>下载 Markdown</Button>
        <Button size="small" icon={<CopyOutlined />} onClick={copySummary}>复制结论</Button>
        {copy && <span role="status" className="muted">{copy}</span>}
      </Space>
    </div>
    <Tabs defaultActiveKey={hasClaims || !result.image ? "claims" : "image"} items={[
      { key: "image", label: "图片审核", children: <ImageResults result={result} /> },
      { key: "claims", label: hasClaims ? "文字审核（" + result.audit.claims.length + "）" : "文字审核", children: <ClaimResults result={result} /> },
      { key: "crossmodal", label: "图文匹配", children: <CrossmodalResults result={result} /> },
      { key: "tools", label: "检查记录", children: <EvidenceResults result={result} /> },
    ]} />
    {explanation.caveat && <p className="report-caveat">{explanation.caveat}</p>}
  </Card>;
}

function Workbench() {
  const [form] = Form.useForm();
  const [file, setFile] = useState(null);
  const [samples, setSamples] = useState([]);
  const [references, setReferences] = useState([]);
  const [report, setReport] = useState(null);
  const [steps, setSteps] = useState([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const request = useRef(null);
  const reportRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    readJson("/api/samples", { signal: controller.signal }).then(data => setSamples(data.samples || [])).catch(() => {});
    readJson("/api/reference-catalog", { signal: controller.signal }).then(data => setReferences(data.references || [])).catch(() => {});
    return () => {
      controller.abort();
      request.current?.abort();
    };
  }, []);

  async function run(task, operation) {
    if (request.current) return;
    const controller = new AbortController();
    request.current = controller;
    setError("");
    setReport(null);
    setSteps([]);
    setBusy(task);
    try {
      setReport({ data: await operation(controller.signal), isSample: task === "sample" });
      if (window.matchMedia("(max-width: 800px)").matches) {
        reportRef.current?.scrollIntoView({ block: "start" });
      }
    } catch (err) {
      if (!controller.signal.aborted) setError(err.message);
    } finally {
      request.current = null;
      if (!controller.signal.aborted) setBusy("");
    }
  }

  function submit(values) {
    const text = values.text?.trim();
    const source = values.referenceSource?.trim(), excerpt = values.referenceExcerpt?.trim();
    if (!text && !file) {
      setError("请上传图片或填写正文。");
      return;
    }
    if (Boolean(source) !== Boolean(excerpt)) {
      setError("请同时填写依据名称和依据原文。");
      return;
    }
    const data = new FormData();
    if (file) data.set("file", file);
    if (text) data.set("text", text);
    if (values.product?.trim()) data.set("product", values.product.trim());
    if (source) data.set("references", JSON.stringify([{ source, excerpt, product: values.referenceProduct?.trim() || null }]));
    if (values.referenceIds?.length) data.set("reference_ids", JSON.stringify(values.referenceIds));
    if (values.fast) data.set("fast", "1");
    if (values.llm) data.set("llm", "1");
    run("analysis", async signal => {
      const { job_id } = await readJson("/api/analyze", { method: "POST", body: data, signal });
      const deadline = Date.now() + 10 * 60 * 1000;
      while (!signal.aborted && Date.now() < deadline) {
        const job = await readJson("/api/job/" + encodeURIComponent(job_id), { signal });
        setSteps(job.steps || []);
        if (job.status === "done") return job.result;
        if (job.status === "error" || job.status === "not_found") throw new Error(job.error || "本次核验未完成。");
        await new Promise(resolve => setTimeout(resolve, 850));
      }
      throw new Error("等待超时，请重新提交。");
    });
  }

  return <>
    <header className="topbar"><div className="page-width topbar-inner">
      <div className="brand"><div className="brand-mark"><SafetyCertificateOutlined /></div>
        <span>BeautyProof</span>
      </div>
      <span className="topbar-caption">内容审核工作台</span>
    </div></header>
    <main className="page-width">
      <div className="intro">
        <span className="eyebrow"><span /> 审核工作台</span>
        <h1>美妆宣传核验台</h1>
        <p>核对宣传图片、文案和功效依据，把需要人工复核的地方标出来。</p>
      </div>
      <div className="workbench-grid">
        <Card className="surface intake-surface" title="送审内容" extra={<Button type="link" size="small" disabled={!!busy} onClick={() => {
          form.setFieldValue("text", "【青蓝精华】连续使用 7 天，水分提升 20%。");
          setError("");
        }}>试填示例正文</Button>}>
          <Form form={form} name="workbench" layout="vertical" disabled={!!busy} onFinish={submit}
            requiredMark={false}>
            {!!samples.length && <Form.Item label="已有样本">
              <Select placeholder="选择样本查看预置报告" options={samples.map(sample => ({
                value: sample.stem, label: sample.name + (sample.risk_zh ? " · " + sample.risk_zh : ""),
              }))} onChange={stem => run("sample", signal => readJson("/api/sample/" + encodeURIComponent(stem), { signal }))} />
            </Form.Item>}
            <Form.Item label="宣传图片">
              <Upload.Dragger name="file" accept=".png,.jpg,.jpeg" multiple={false} maxCount={1}
                fileList={file ? [file] : []} onRemove={() => { setFile(null); return true; }}
                beforeUpload={picked => {
                  if (!/\.(png|jpe?g)$/i.test(picked.name) || picked.size > 32 * 1024 * 1024) {
                    setError("请选择不超过 32 MB 的 PNG 或 JPG 图片。");
                    return Upload.LIST_IGNORE;
                  }
                  setFile(picked);
                  setError("");
                  return false;
                }}>
                <span className="upload-icon"><UploadOutlined /></span>
                <strong>点击或拖入图片</strong>
                <small>PNG / JPG · 最大 32 MB</small>
              </Upload.Dragger>
            </Form.Item>
            <Form.Item name="text" label="宣传正文">
              <Input.TextArea rows={4} placeholder="粘贴种草正文、产品功效宣称或图片配文…" />
            </Form.Item>
            <Collapse className="options" items={[{
              key: "context", label: "补充产品与功效依据", extra: <span className="muted">选填</span>,
              children: <>
                <Form.Item name="product" label="宣传产品"><Input placeholder="例如青蓝精华" /></Form.Item>
                {!!references.length && <Form.Item name="referenceIds" label="选择已核验资料">
                  <Select mode="multiple" options={references.map(ref => ({ value: ref.id, label: `${ref.source} · ${ref.product} · ${ref.locator}` }))} />
                </Form.Item>}
                <Form.Item name="referenceSource" label="依据名称"><Input placeholder="例如试验记录或原始报告名称" /></Form.Item>
                <Form.Item name="referenceProduct" label="依据对应产品"><Input placeholder="按依据原文填写" /></Form.Item>
                <Form.Item name="referenceExcerpt" label="待核验摘录"><Input.TextArea rows={3} /></Form.Item>
              </>,
            }]} />
            <div className="form-options">
              <Form.Item name="fast" valuePropName="checked" noStyle><Checkbox title="跳过耗时的图像检测模型">快速预览</Checkbox></Form.Item>
              <Form.Item name="llm" valuePropName="checked" noStyle><Checkbox title="使用已配置的本地模型辅助抽取宣称">本地模型辅助</Checkbox></Form.Item>
            </div>
            {error && <Alert className="form-error" title={error} type="error" showIcon />}
            <Button type="primary" size="large" block htmlType="submit" loading={busy === "analysis"}
              icon={<ArrowRightOutlined />} iconPlacement="end">开始核验</Button>
            <p className="input-note">图片和正文至少提供一项。</p>
          </Form>
        </Card>
        <section className="report-region" ref={reportRef} aria-label="核验结果">
          {busy ? <RunningReport steps={steps} /> : report ? <FinishedReport key={report.data.run_id || report.data.image?.stem}
            result={report.data} isSample={report.isSample} /> : <EmptyReport />}
        </section>
      </div>
    </main>
    <footer className="page-width footer"><span>BeautyProof</span><span>美妆内容审核 · 证据辅助人工复核</span></footer>
  </>;
}

createRoot(document.getElementById("workbench")).render(
  <ConfigProvider locale={zhCN} theme={theme}><Workbench /></ConfigProvider>,
);

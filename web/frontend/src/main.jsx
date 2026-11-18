import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert, Avatar, Button, Card, Checkbox, Collapse, ConfigProvider, Drawer, Dropdown, Form, Image, Input, Modal,
  Select, Spin, Tabs, Tag, Tooltip, Upload,
} from "antd";
import {
  AppstoreOutlined, ArrowLeftOutlined, ArrowRightOutlined, ArrowUpOutlined, AuditOutlined,
  CloseOutlined, CopyOutlined, DownloadOutlined, FileSearchOutlined, MessageOutlined, PaperClipOutlined,
  PictureOutlined, PlusOutlined, ScanOutlined, SwapOutlined, UploadOutlined, UserOutlined,
} from "@ant-design/icons";
import zhCN from "antd/locale/zh_CN";
import "antd/dist/reset.css";
import "./workbench.css";
import { DamageRegion, DamageReport } from "./damage-review.jsx";
import Portal from "./portal.jsx";

const toolLabels = {
  hash: "文件指纹", c2pa: "来源凭证", ela: "压缩痕迹", spectral: "频域取证",
  ocr: "图中文字", ai_label: "AI 来源标识", aigc: "生成倾向", trufor: "局部异常",
  text: "文案检查", crossmodal: "图文对照", claim_model: "DeepSeek 文案模型", semantic_review: "语义复核",
  router: "识别审核需求", publicity: "宣传核验", damage: "退款破损审核", image_agent: "图像证据复核",
  evidence_followup: "补证后核验",
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
const referenceTypes = { efficacy_study: "产品功效试验", product_label: "产品标签", brand_claim: "品牌宣称", method: "技术方法" };
const theme = {
  token: {
    colorPrimary: "#8e343d", colorLink: "#8a3039", colorText: "#261d20", colorTextSecondary: "#65595c",
    colorBorder: "#e3d9d6", colorBgLayout: "#f5efeb", colorWarning: "#946023",
    colorWarningText: "#7b4210", borderRadius: 10,
    fontFamily: 'Geist, "Segoe UI", "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif',
    controlHeight: 40,
    motionEaseOutBack: "cubic-bezier(0.22, 1, 0.36, 1)",
    motionEaseInBack: "cubic-bezier(0.64, 0, 0.78, 0)",
  },
};

async function readJson(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload || payload.error) {
    const error = new Error(payload?.error || (response.status === 413 ? "文件太大，两张图片合计须小于 32 MB。" : "无法读取审核服务的响应，请稍后重试。"));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function AuditStatus({ value }) {
  const [label, color] = states[value] || ["待检查", "default"];
  return <Tag color={color}>{label}</Tag>;
}

function EmptyReport() {
  return <div className="empty-report">
    <FileSearchOutlined aria-hidden="true" />
    <p>送审后，可在这里查看结论和对应证据。</p>
  </div>;
}

function RunningReport({ steps }) {
  const latest = [...new Map(steps.map(step => [step.tool, step])).values()];
  const current = latest.filter(step => step.state === "running").at(-1);
  return <div className="running-report">
    <p role="status">{current ? toolLabels[current.tool] || current.tool : "正在处理本轮消息"}…</p>
    {!!latest.length && <details className="evidence-detail">
      <summary>检查进度</summary>
      <ul className="check-list">
        {latest.map(step => <li key={step.tool}><span>{toolLabels[step.tool] || step.tool}</span><AuditStatus value={step.state} /></li>)}
      </ul>
    </details>}
  </div>;
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
  const contextConflicts = result.audit?.context_conflicts || [];
  const reviewWords = (result.evidence?.text?.evidence?.[0]?.claim_violations || []).filter(item => item.severity === "review");
  const reasons = result.verdict.reasons || [];
  return <div className="result-stack">
    {result.audit?.ai_illustration_disclosed && <Alert type="info" showIcon title="素材已标注为 AI 示意" />}
    {result.audit?.checks?.semantic_review?.status === "completed" && result.audit.checks.semantic_review.reference_count === 0 &&
      <Alert type="info" showIcon title="未匹配到同产品、同市场资料" description="语义审核仅检查了正文上下文，请补全准确产品名称和市场，或核对知识库资料。" />}
    {result.audit?.checks?.semantic_review?.status === "completed" && result.audit.checks.semantic_review.reference_total > result.audit.checks.semantic_review.reference_count &&
      <Alert type="info" showIcon title="部分资料未纳入语义审核" description={result.audit.checks.semantic_review.reason} />}
    {reviewWords.map((item, index) => <article className="claim" key={"review-" + index}>
      <Tag color="orange">词语线索 · 待人工复核</Tag><blockquote>{item.context}</blockquote>
      <p>{item.review_reason}</p>
    </article>)}
    {contextConflicts.map((item, index) => <article className="claim" key={"context-" + index}>
      <Tag color="orange">{item.type}</Tag>
      <blockquote><mark>{item.first.quote}</mark></blockquote>
      <blockquote><mark>{item.second.quote}</mark></blockquote>
      <p>{item.reason}</p>
    </article>)}
    {(result.audit?.semantic_findings || []).map((item, index) => <article className="claim" key={"semantic-" + index}>
      <Tag color="orange">{item.type || "语义线索"} · 待人工复核</Tag>
      <blockquote><mark>{item.quote}</mark></blockquote>
      <blockquote>{item.source} · {item.other_quote}</blockquote>
      {item.reference_id && <p>{referenceTypes[item.source_type] || "资料类型待核验"}{item.locator && ` · ${item.locator}`}</p>}
      {item.reference_id && !item.reference_conditions?.product_version && <p>资料的具体产品版本待确认，比较前请核对。</p>}
      {item.reference_id && /^https?:\/\//i.test(item.source) && <p><a href={item.source} target="_blank" rel="noreferrer">查看来源原文</a></p>}
      {item.reference_conditions && Object.keys(item.reference_conditions).length > 0 && <p>
        {Object.entries(item.reference_conditions).map(([key, value]) => `${({ version: "资料版本", product_version: "产品版本", market: "市场", method: "方法", sample: "样本", population: "人群", period: "周期", conditions: "条件" })[key] || key}：${value}`).join("；")}
      </p>}
      <p>{item.reason}</p>
    </article>)}
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
        {/^https?:\/\//i.test(reference.source) && <p><a href={reference.source} target="_blank" rel="noreferrer">查看来源原文</a></p>}
        {reference.source_type && <p>资料类型 {referenceTypes[reference.source_type]}{reference.market && ` · ${reference.market}`}</p>}
        {reference.product_version && <p>产品版本 {reference.product_version}</p>}
        {!reference.source_type && reference.verification === "reviewer_verified" && <p>资料类型待核验</p>}
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
  if (!result.image) {
    return <p className="tab-empty">上传图片后可查看图文或包装线索。</p>;
  }
  const findings = (cross.contradictions || []).filter(item => !item.body_claim);
  const alignment = cross.alignment || (cross.contradictions || []).filter(item => item.body_claim)
    .map(item => ({ body: item.body_claim, image: item.image_claim, status: "数值冲突" }));
  return <div className="result-stack">
    {cross.ai_illustration_disclosed && <Alert type="info" showIcon title="素材已声明为 AI 示意" />}
    {cross.package_reference && <article className="claim">
      <Tag color="orange">包装参考图 · {cross.package_reference.status}</Tag>
      <p>{cross.package_reference.reason}</p>
      <ImageRegions image={result.image} regions={cross.package_reference.regions} />
    </article>}
    {!!result.audit?.uncertain_ocr?.length && <article className="claim">
      <Tag color="orange">图片文字不清晰，待人工核对</Tag>
      {result.audit.uncertain_ocr.map((line, i) => <p key={i}>{line.text}</p>)}
      <ImageRegions image={result.image} regions={result.audit.uncertain_ocr} />
    </article>}
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
      {!!result.audit?.image_review_checks?.length && <p className="next-action">
        <b>优先复核</b>　{result.audit.image_review_checks.map(key => toolLabels[key] || key).join("、")}
      </p>}
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

const harnessActions = {
  request_metadata: "补充产品、版本或市场", request_original: "补充清晰原图",
  request_study: "补充已核验功效试验", human_review: "转人工复核", finish: "结束取证",
};

function downloadJson(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function HarnessResults({ result, references, busy, onFollowup, onAddReference }) {
  const [selected, setSelected] = useState([]);
  const trace = result.harness_trace;
  if (!trace) return null;
  return <div className="result-stack">
    <p><b>下一步</b>　{trace.decision.label}</p>
    <p className="muted">策略 {trace.policy_version} · {trace.decision.reason_code} · {trace.elapsed_ms} ms</p>
    <p className="muted">依据编号　{trace.decision.evidence_ids.join("、") || "尚无可用观测"}</p>
    {trace.state === "waiting_evidence" && <>
      <p>核对功效试验原文、产品版本和市场后，选择资料继续核验。</p>
      <Button onClick={onAddReference} disabled={busy}>添加已核验资料</Button>
      <Select mode="multiple" aria-label="补证资料" placeholder="选择已核验资料" value={selected}
        onChange={setSelected} disabled={busy} options={references.map(ref => ({
          value: ref.id, label: `${ref.source} · ${ref.product} · ${ref.locator}`,
        }))} />
      <Button type="primary" disabled={busy || !selected.length} onClick={() => onFollowup(selected)}>
        补证后核验
      </Button>
    </>}
    {!!result.evidence_comparison?.length && <>
      <p>已沿用原始观测完成一次补证核验。</p>
      <div className="claim-table-wrap"><table className="claim-table">
        <thead><tr><th scope="col">宣称</th><th scope="col">补证前</th><th scope="col">补证后</th></tr></thead>
        <tbody>{result.evidence_comparison.map(row => <tr key={row.claim_id}>
          <td>{row.quote}</td><td>{row.before}</td><td>{row.after}</td>
        </tr>)}</tbody>
      </table></div>
      {result.evidence_comparison.flatMap(row => row.added_references.map(ref => <p key={`${row.claim_id}-${ref.id}`}>
        <b>新增引用</b>　{ref.source} · {ref.locator}<br />{ref.quote}
      </p>))}
    </>}
    <Button onClick={() => downloadJson({ schema: "beautyproof/harness@1", sample_id: result.run_id, ...trace }, "harness-trace.json")}>
      下载决策轨迹
    </Button>
    <Collapse items={[{ key: "feedback", label: "标注取证顺序并导出回放样本", children:
      <Form layout="vertical" requiredMark={false} initialValues={{ sample_id: result.run_id, split: "development" }}
        onFinish={values => downloadJson({
          sample_id: values.sample_id, group: values.group, split: values.split, kind: "reviewer_labeled",
          observation: trace.observation,
          feedback: { correct_action: values.correct_action, reason: values.reason },
        }, "harness-feedback.json")}>
        <Form.Item name="sample_id" label="样本编号" rules={[{ required: true, whitespace: true }]}><Input /></Form.Item>
        <Form.Item name="group" label="来源组（同一宣传或试验使用同组）" rules={[{ required: true, whitespace: true }]}><Input /></Form.Item>
        <Form.Item name="split" label="分区" rules={[{ required: true }]}><Select options={[
          { value: "development", label: "开发集" }, { value: "holdout", label: "冻结留出集" },
        ]} /></Form.Item>
        <Form.Item name="correct_action" label="正确下一步" rules={[{ required: true }]}>
          <Select options={Object.entries(harnessActions).map(([value, label]) => ({ value, label }))} />
        </Form.Item>
        <Form.Item name="reason" label="纠正理由" rules={[{ required: true, whitespace: true }]}><Input.TextArea rows={2} /></Form.Item>
        <Button htmlType="submit">导出反馈与观测</Button>
      </Form>,
    }]} />
  </div>;
}

function EvidenceResults({ result, ...harnessProps }) {
  const evidence = result.evidence || {};
  const checks = result.audit?.checks || {};
  const keys = [...new Set([...Object.keys(checks), ...Object.keys(evidence)])];
  const actions = [...(result.agent?.decisions || []), ...(result.agent?.actions || [])];
  return <div className="result-stack">
    <HarnessResults result={result} {...harnessProps} />
    <Collapse items={keys.map(key => ({
      key,
      label: result.tool_meta?.[key]?.label || toolLabels[key] || key,
      extra: <AuditStatus value={checks[key]?.decision_status || checks[key]?.status ||
        (evidence[key]?.evidence?.[0]?.available === false ? "model_unavailable" : "completed")} />,
      children: <>
        <p>{evidence[key]?.observed || "本次没有可引用的检查结果。"}</p>
        {checks[key]?.model && <p className="muted">模型　{checks[key].model}</p>}
        {checks[key]?.reason && <p className="muted">原因　{checks[key].reason}</p>}
        {evidence[key]?.cannot_prove && <p className="muted">{evidence[key].cannot_prove}</p>}
      </>,
    }))} />
    {!!actions.length && <Collapse items={[{ key: "agent", label: "查看 Agent 决策记录", children: <ul className="detail-list">
      {actions.map((action, i) => <li key={i}><b>{toolLabels[action.tool] || action.tool || action.action}</b>　{action.reason}</li>)}
    </ul> }]} />}
    {result.llm_explanation && <p><b>本地模型辅助解读</b><br />{result.llm_explanation}</p>}
  </div>;
}

function publicityExplanation(result) {
  const explanation = result.verdict.explain || {};
  if (!result.image && result.verdict.risk_level === "suspicious" && result.audit?.context_conflicts?.length) {
    return {
      ...explanation,
      headline: "同一宣称出现不同数值",
      summary: "相同产品和使用条件下，正文给出了不同的数值。",
      what_to_do: "核对原始功效资料和对应宣传文案。",
    };
  }
  return explanation;
}

function reportText(result) {
  const { image, verdict, audit, evidence = {}, agent } = result;
  const explanation = publicityExplanation(result);
  const label = (risks[verdict.risk_level] || risks.inconclusive)[0];
  return [
    "# BeautyProof 美妆宣传核验报告", "",
    "## 核验结论", "", "**" + label + "**", explanation.headline, explanation.summary,
    explanation.what_to_do && "下一步　" + explanation.what_to_do,
    explanation.caveat, "",
    "## 送审信息", "", image?.name || "仅正文", image?.sha256 && "SHA-256 " + image.sha256, "",
    "## 判定依据", "", ...(verdict.reasons || []).map(reason => "- " + reason), "",
    "## 正文前后核对", "", ...(audit?.context_conflicts || []).map(item =>
      "- " + item.type + "｜" + item.first.quote + " / " + item.second.quote), "",
    ...(audit?.semantic_findings || []).map(item => "- 语义线索，待人工复核｜" + item.quote + " / " + item.source + " " + item.other_quote + "｜" + item.reason),
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

function FinishedReport({ result, ...harnessProps }) {
  const [copy, setCopy] = useState("");
  const verdict = result.verdict;
  const [label, color] = risks[verdict.risk_level] || risks.inconclusive;
  const explanation = publicityExplanation(result);
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
  return <Card className="surface report-surface" title="宣传核验" extra={<Dropdown menu={{ items: [
    { key: "download", label: "下载 Markdown", icon: <DownloadOutlined />, onClick: download },
    { key: "copy", label: "复制结论", icon: <CopyOutlined />, onClick: copySummary },
  ] }} trigger={["click"]}><Button size="small" type="text">报告操作</Button></Dropdown>}>
    <div className="report-heading">
      <Tag color={color}>{label}</Tag>
      <h2>{explanation.headline || label}</h2>
      {explanation.summary && <p>{explanation.summary}</p>}
      {explanation.what_to_do && <p className="next-action"><b>下一步</b>　{explanation.what_to_do}</p>}
      {copy && <span role="status" className="muted">{copy}</span>}
    </div>
    <Tabs defaultActiveKey={hasClaims || !result.image ? "claims" : "image"} items={[
      { key: "image", label: "图片审核", children: <ImageResults result={result} /> },
      { key: "claims", label: hasClaims ? "文字审核（" + result.audit.claims.length + "）" : "文字审核", children: <ClaimResults result={result} /> },
      { key: "crossmodal", label: "图文匹配", children: <CrossmodalResults result={result} /> },
      { key: "tools", label: "检查记录", children: <EvidenceResults result={result} {...harnessProps} /> },
    ]} />
    {explanation.caveat && <p className="report-caveat">{explanation.caveat}</p>}
  </Card>;
}

const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const reviewLabels = { publicity: "宣传核验", damage: "退款破损审核" };

const workspaceCopy = {
  consumer: {
    roleName: "消费者", title: "美妆助手", greeting: "今天想核对什么？",
    lead: "发来你看到的美妆文案或宣传图片，一起核对产品说法和依据。",
    placeholder: "问问功效宣传，或附一张宣传图片…",
    prompts: [
      { title: "看看功效宣传", detail: "核对说法和出处", icon: FileSearchOutlined,
        text: "请核验宣传文案：这款精华连续使用 7 天，水分提升 20%。" },
      { title: "核对宣传图片", detail: "看看图片和文案是否一致", icon: PictureOutlined,
        text: "请核验这张宣传图片，看看图片上的功效宣称有哪些依据。" },
    ],
  },
  reviewer: {
    roleName: "企业审核人员", title: "企业审核助手", greeting: "今天要复核哪份材料？",
    lead: "发来宣传文案、商品图片或售后凭证，让助手整理审核依据。",
    placeholder: "输入审核需求，或拖入一张送审图片…",
    prompts: [
      { title: "核验宣传宣称", detail: "找出文案冲突和缺失依据", icon: FileSearchOutlined,
        text: "核验宣传文案：【示例精华】2025版14天水分提升20%。【示例精华】2025版14天水分提升40%。" },
      { title: "审核退款破损", detail: "核对整图和可疑区域", icon: PictureOutlined,
        text: "请审核退款申请中的商品破损图片，并指出值得复核的证据。" },
    ],
  },
};

function Workbench({ role, onReset }) {
  const workspace = workspaceCopy[role];
  const [form] = Form.useForm();
  const [referenceForm] = Form.useForm();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [roi, setRoi] = useState(null);
  const [packageReference, setPackageReference] = useState(null);
  const [references, setReferences] = useState([]);
  const [referenceOpen, setReferenceOpen] = useState(false);
  const [referenceError, setReferenceError] = useState("");
  const [referenceSaving, setReferenceSaving] = useState(false);
  const [deepseekReady, setDeepseekReady] = useState(false);
  const [useModel, setUseModel] = useState(true);
  const [messages, setMessages] = useState([]);
  const [review, setReview] = useState(null);
  const [steps, setSteps] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [materialsOpen, setMaterialsOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [compact, setCompact] = useState(() => window.matchMedia("(max-width: 1100px)").matches);
  const previousJob = useRef(null);
  const request = useRef(null);
  const logRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 1100px)");
    const resize = () => setCompact(query.matches);
    query.addEventListener("change", resize);
    return () => query.removeEventListener("change", resize);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    readJson("/api/reference-catalog", { signal: controller.signal })
      .then(data => setReferences(data.references || [])).catch(() => {});
    readJson("/api/config", { signal: controller.signal })
      .then(data => setDeepseekReady(data.deepseek_configured)).catch(() => {});
    return () => {
      controller.abort();
      request.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [messages, busy]);

  function attach(picked) {
    if (picked && (!/\.(png|jpe?g)$/i.test(picked.name) || picked.size >= MAX_IMAGE_BYTES)) {
      setError("请选择小于 32 MB 的 JPG 或 PNG 图片。");
      return;
    }
    setFile(picked);
    setRoi(null);
    if (picked !== file) setReview(null);
    if (file && picked !== file) {
      setPackageReference(null);
      form.resetFields(["copy", "product", "market", "referenceIds", "referenceSource", "referenceProduct", "referenceExcerpt"]);
    }
    setError("");
  }

  async function addReference(values) {
    setReferenceError("");
    setReferenceSaving(true);
    try {
      const data = await readJson("/api/reference-catalog", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      setReferences(previous => [...previous, data.reference]);
      referenceForm.resetFields();
      setReferenceOpen(false);
    } catch (err) {
      setReferenceError(err.message);
    } finally {
      setReferenceSaving(false);
    }
  }

  function submit(values) {
    if (request.current) return;
    const message = values.message?.trim();
    const copy = values.copy?.trim();
    const source = values.referenceSource?.trim();
    const excerpt = values.referenceExcerpt?.trim();
    if (!message && !file && !copy) {
      setError("请填写审核需求或附上图片。");
      return;
    }
    if (packageReference && !file) {
      setError("请同时附上要核验的宣传图片。");
      return;
    }
    if ((file?.size || 0) + (packageReference?.size || 0) >= MAX_IMAGE_BYTES) {
      setError("两张图片合计须小于 32 MB。");
      return;
    }
    if (Boolean(source) !== Boolean(excerpt)) {
      setError("请同时填写依据名称和依据原文。");
      return;
    }
    const data = new FormData();
    data.set("role", role);
    if (message) data.set("message", message);
    if (copy) data.set("copy", copy);
    if (file) data.set("file", file);
    if (packageReference) data.set("reference_image", packageReference);
    if (roi) data.set("roi", JSON.stringify(roi));
    if (values.product?.trim()) data.set("product", values.product.trim());
    if (values.market?.trim()) data.set("market", values.market.trim());
    if (source) data.set("references", JSON.stringify([{ source, excerpt, product: values.referenceProduct?.trim() || null }]));
    if (values.referenceIds?.length) data.set("reference_ids", JSON.stringify(values.referenceIds));
    if (values.fast) data.set("fast", "1");
    data.set("llm", deepseekReady && useModel ? "1" : "0");
    if (previousJob.current) data.set("previous_job_id", previousJob.current);
    runTurn(data, { kind: "user", text: message || copy, attachment: file?.name });
  }

  function followup(referenceIds) {
    if (request.current || !review) return;
    const data = new FormData();
    data.set("role", role);
    data.set("evidence_followup", "1");
    data.set("previous_job_id", review.id);
    data.set("reference_ids", JSON.stringify(referenceIds));
    data.set("llm", "0");
    runTurn(data, { kind: "user", text: "使用已核验资料补证后核验。" });
  }

  async function runTurn(data, entry) {
    const controller = new AbortController();
    request.current = controller;
    const { signal } = controller;
    setBusy(true);
    setError("");
    setSteps([]);
    try {
      const { job_id: jobId } = await readJson("/api/agent/turn", { method: "POST", body: data, signal });
      setMessages(previous => [...previous, entry]);
      form.setFieldValue("message", "");
      while (!signal.aborted) {
        const turn = await readJson("/api/job/" + encodeURIComponent(jobId), { signal });
        setSteps(turn.steps || []);
        if (turn.status === "done") {
          const result = turn.result;
          previousJob.current = jobId;
          setMessages(previous => [...previous, { kind: "agent", text: result.message, evidence: result.evidence,
            reviewId: result.type === "review" ? jobId : null }]);
          if (result.type === "review") {
            setReview({ ...result, id: jobId });
            if (!compact) setReportOpen(true);
          }
          return;
        }
        if (turn.status === "error") throw new Error(turn.error || "本次审核未完成，请重新发送。");
        await new Promise(resolve => setTimeout(resolve, 750));
      }
    } catch (err) {
      if (!signal.aborted) {
        if (err.status === 409) previousJob.current = null;
        setError(err.message);
      }
    } finally {
      request.current = null;
      if (!signal.aborted) setBusy(false);
    }
  }

  const RoleIcon = role === "consumer" ? UserOutlined : AuditOutlined;
  const threadTitle = messages.find(entry => entry.kind === "user")?.text || "新的审核对话";
  const reportContent = <>
    {busy && <RunningReport steps={steps} />}
    {review ? <div className="review-results" key={review.id}>
      {review.reports.publicity && <FinishedReport result={review.reports.publicity}
        references={references} busy={busy} onFollowup={followup} onAddReference={() => setReferenceOpen(true)} />}
      {review.reports.damage && <DamageReport result={review.reports.damage} />}
      {Object.entries(review.errors).map(([key, value]) => <Alert key={key} type="error" showIcon
        title={reviewLabels[key]} description={value} />)}
    </div> : !busy && <EmptyReport />}
  </>;

  return <div className="desk">
    <aside className="desk-sidebar" aria-label="会话导航">
      <a className="desk-brand" href="#"><ScanOutlined aria-hidden="true" /> BeautyProof</a>
      <Button className="desk-new" block icon={<PlusOutlined aria-hidden="true" />} onClick={onReset} disabled={busy}>新建对话</Button>
      <div className="desk-session">
        <small>当前会话</small>
        <div className="desk-thread"><MessageOutlined aria-hidden="true" /><span title={threadTitle}>{threadTitle}</span></div>
      </div>
      <div className="desk-role">
        <Avatar shape="square" size={36} icon={<RoleIcon />} />
        <strong>{workspace.roleName}</strong>
        <Tooltip title="切换身份"><Button type="text" href="#" className="desk-switch" aria-label="切换身份"
          icon={<SwapOutlined aria-hidden="true" />} /></Tooltip>
      </div>
    </aside>

    <main className="desk-main">
      <header className="desk-topbar">
        <div className="desk-top-title">
          <Button className="desk-mobile-home" type="text" size="small" href="#" aria-label="返回首页"
            icon={<ArrowLeftOutlined aria-hidden="true" />} />
          <span>{workspace.title}</span>
        </div>
        <div className="desk-top-actions">
          <Button className="desk-mobile-new" type="text" size="small" onClick={onReset} disabled={busy}
            aria-label="新建对话" icon={<PlusOutlined aria-hidden="true" />} />
          {(review || busy) && <Button size="small" type={reportOpen ? "default" : "text"} icon={<FileSearchOutlined aria-hidden="true" />}
            aria-expanded={reportOpen} onClick={() => setReportOpen(value => !value)}>审核报告</Button>}
        </div>
      </header>

      <div className={"desk-body" + (reportOpen && !compact ? " desk-with-report" : "")}>
        <section className={"conversation" + (messages.length ? " conversation-active" : "")} aria-label="审核对话">
          <div ref={logRef} className="conversation-log" role="log" aria-label="审核消息">
            {!messages.length && <div className="welcome">
              <div className="welcome-insignia"><ScanOutlined aria-hidden="true" /></div>
              <h1>{workspace.greeting}</h1>
              <p>{workspace.lead}</p>
              <div className="conversation-examples">
                {workspace.prompts.map(({ title, detail, text, icon: Icon }) => <Button disabled={busy}
                  key={title} className="example-action" onClick={() => {
                    form.setFieldValue("message", text);
                    inputRef.current?.focus();
                  }}>
                  <span className="example-icon"><Icon aria-hidden="true" /></span>
                  <span className="example-copy"><strong>{title}</strong><small>{detail}</small></span>
                  <ArrowRightOutlined className="example-arrow" aria-hidden="true" />
                </Button>)}
              </div>
            </div>}
            {!!messages.length && <div className="message-stream">
              {messages.map((entry, index) => <article key={index} className={"message message-" + entry.kind}>
                {entry.kind === "agent" && <Avatar className="message-avatar" shape="square" size={29} icon={<ScanOutlined />} />}
                <div className="message-content">
                  <h2>{entry.kind === "user" ? "你" : workspace.title}</h2>
                  {entry.text && <p>{entry.text}</p>}
                  {entry.attachment && <div className="message-file"><PaperClipOutlined aria-hidden="true" /> {entry.attachment}</div>}
                  {!!entry.evidence?.length && <ul className="reply-evidence">
                    {entry.evidence.map((item, i) => <li key={i}>
                      <strong>{item.label}</strong>
                      {item.quote && <blockquote>{item.quote}</blockquote>}
                      {item.detail && <p>{item.detail}</p>}
                    </li>)}
                  </ul>}
                  {entry.reviewId && entry.reviewId === review?.id && <Button className="message-report" size="small"
                    icon={<FileSearchOutlined aria-hidden="true" />} onClick={() => setReportOpen(true)}>查看报告</Button>}
                </div>
              </article>)}
              {busy && <div className="message-pending" role="status">
                <Spin size="small" /> 审核助手正在处理…
              </div>}
            </div>}
          </div>

          <Form className="desk-form" form={form} name="review-turn" layout="vertical" disabled={busy}
            onFinish={submit} requiredMark={false}>
            <div className="composer" onDragOver={event => {
              if (!busy && event.dataTransfer.types.includes("Files")) event.preventDefault();
            }} onDrop={event => {
              if (!event.dataTransfer.files.length) return;
              event.preventDefault();
              if (!busy) attach(event.dataTransfer.files[0]);
            }}>
              <Form.Item name="message" noStyle>
                <Input.TextArea ref={inputRef} autoSize={{ minRows: 2, maxRows: 6 }} maxLength={12000}
                  aria-label="审核需求" variant="borderless"
                  placeholder={messages.length ? "追问报告中的依据，或补充材料…" : workspace.placeholder}
                  onKeyDown={event => {
                    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing &&
                      event.nativeEvent.keyCode !== 229) {
                      event.preventDefault();
                      form.submit();
                    }
                  }} />
              </Form.Item>
              {file && <div className="attachment">
                {preview && <img src={preview} alt="" />}
                <span title={file.name}>{file.name}</span>
                <Button type="text" size="small" aria-label="移除附图" icon={<CloseOutlined aria-hidden="true" />} onClick={() => attach(null)} />
              </div>}
              <div className="composer-actions">
                <div className="composer-tools">
                  <Upload accept=".png,.jpg,.jpeg" showUploadList={false} beforeUpload={picked => {
                    attach(picked);
                    return Upload.LIST_IGNORE;
                  }}>
                    <Button type="text" icon={<PaperClipOutlined aria-hidden="true" />}>附图</Button>
                  </Upload>
                  <Button type="text" icon={<AppstoreOutlined aria-hidden="true" />}
                    onClick={() => setMaterialsOpen(true)}>补充材料</Button>
                </div>
                <Button type="primary" htmlType="submit" loading={busy} icon={<ArrowUpOutlined aria-hidden="true" />} iconPlacement="end">发送</Button>
              </div>
            </div>
            {error && <Alert className="form-error" type="error" showIcon title={error} />}
            <Drawer title="补充审核材料" open={materialsOpen} onClose={() => setMaterialsOpen(false)}
              size="min(100vw, 520px)" forceRender
              extra={<Button onClick={() => setMaterialsOpen(false)}>完成填写</Button>}>
              <div className="options-content">

                {role === "reviewer" && file && <DamageRegion key={preview} preview={preview} roi={roi} onChange={setRoi} busy={busy} />}
                <Form.Item name="copy" label="宣传正文"><Input.TextArea rows={3} maxLength={12000} /></Form.Item>
                <div className="option-fields">
                  <Form.Item name="product" label="宣传产品"><Input list="known-products" placeholder="完整产品名" /></Form.Item>
                  <Form.Item name="market" label="产品市场"><Input placeholder="如中国大陆" /></Form.Item>
                </div>
                <datalist id="known-products">{[...new Set(references.map(ref => ref.product))].map(name => <option key={name} value={name} />)}</datalist>
                <Form.Item label="包装参考图" extra="仅用于定位宣传图片中的相似包装，两张图片合计须小于 32 MB。">
                  <Upload accept=".png,.jpg,.jpeg" maxCount={1} fileList={packageReference ? [packageReference] : []}
                    onRemove={() => { setPackageReference(null); return true; }} beforeUpload={picked => {
                      if (!/\.(png|jpe?g)$/i.test(picked.name) || picked.size >= MAX_IMAGE_BYTES) {
                        setError("请选择小于 32 MB 的 JPG 或 PNG 包装参考图。");
                        return Upload.LIST_IGNORE;
                      }
                      setPackageReference(picked);
                      setError("");
                      return false;
                    }}><Button icon={<UploadOutlined />}>选择参考图</Button></Upload>
                </Form.Item>
                {!!references.length && <Form.Item name="referenceIds" label="已核验资料">
                  <Select mode="multiple" options={references.map(ref => ({
                    value: ref.id, label: [ref.source, ref.product, ref.locator].join(" · "),
                  }))} />
                </Form.Item>}
                {role === "reviewer" && <Button type="link" className="add-reference"
                  onClick={() => { setReferenceError(""); setReferenceOpen(true); }}>添加已核验资料</Button>}
                <div className="option-fields">
                  <Form.Item name="referenceSource" label="待核验依据名称"><Input /></Form.Item>
                  <Form.Item name="referenceProduct" label="依据对应产品"><Input /></Form.Item>
                </div>
                <Form.Item name="referenceExcerpt" label="待核验摘录"><Input.TextArea rows={3} /></Form.Item>
                <Form.Item name="fast" valuePropName="checked" noStyle>
                  <Checkbox>快速预览（跳过生成倾向和局部异常模型）</Checkbox>
                </Form.Item>
              </div>
            </Drawer>
            <div className="model-preference">
              {deepseekReady ? <>
                <Checkbox checked={useModel} onChange={event => setUseModel(event.target.checked)}>DeepSeek 辅助</Checkbox>
                <p>{useModel ? "必要的消息、图片文字、资料摘录和检查摘要可发送至 DeepSeek。图片始终在本地处理。" : "当前使用本地规则和检测工具。"}</p>
              </> : <p>当前使用本地规则和检测工具 · 结果供人工复核</p>}
            </div>
          </Form>
        </section>
        {reportOpen && !compact && <aside className="report-region" aria-labelledby="report-heading">
          <div className="report-panel-header">
            <FileSearchOutlined aria-hidden="true" />
            <h2 id="report-heading">审核报告</h2>
            <Button type="text" size="small" aria-label="关闭报告" onClick={() => setReportOpen(false)}
              icon={<CloseOutlined aria-hidden="true" />} />
          </div>
          {reportContent}
        </aside>}
      </div>
    </main>

    {compact && <Drawer title="审核报告" open={reportOpen} onClose={() => setReportOpen(false)}
      size="min(100vw, 560px)" rootClassName="report-drawer">{reportContent}</Drawer>}


    <Modal open={referenceOpen} title="添加已核验资料" footer={null} onCancel={() => setReferenceOpen(false)}>
      <p>请先核对原始文件、产品规格和适用条件，记录来源以便回查。</p>
      <p>自动支持功效宣称需要明确一致的市场和产品版本；资料版本指报告本身的版本。</p>
      <Form form={referenceForm} layout="vertical" onFinish={addReference} requiredMark={false}>
        <Form.Item name="source_type" label="资料类型" rules={[{ required: true }]}>
          <Select options={Object.entries(referenceTypes).map(([value, label]) => ({ value, label }))} />
        </Form.Item>
        {[
          ["source", "资料来源"], ["product", "产品及规格"], ["version", "资料版本"],
          ["locator", "页码或位置"], ["reviewer", "审核人"],
        ].map(([name, label]) => <Form.Item name={name} label={label} key={name} rules={[{ required: true, whitespace: true }]}>
          <Input />
        </Form.Item>)}
        {[["market", "适用市场"], ["product_version", "产品版本（如 2025版、第四代）"], ["published", "发布日期"], ["method", "测量方法"],
          ["sample", "样本量"], ["population", "受试人群"], ["period", "使用周期"],
          ["conditions", "试验条件"]].map(([name, label]) => <Form.Item name={name} label={label} key={name}>
          <Input />
        </Form.Item>)}
        <Form.Item name="excerpt" label="对应原文摘录" rules={[{ required: true, whitespace: true }]}>
          <Input.TextArea rows={3} />
        </Form.Item>
        <Form.Item name="reviewer_confirmed" valuePropName="checked"
          rules={[{ validator: (_, value) => value ? Promise.resolve() : Promise.reject(new Error("请先确认核对")) }]}>
          <Checkbox>我已核对原始资料及适用范围</Checkbox>
        </Form.Item>
        {referenceError && <Alert type="error" showIcon title={referenceError} />}
        <Button type="primary" htmlType="submit" loading={referenceSaving}>保存到本地知识库</Button>
      </Form>
    </Modal>
  </div>;
}

function ChatWorkspace({ role = "reviewer" }) {
  const [session, setSession] = useState(0);
  return <Workbench key={session} role={role} onReset={() => setSession(value => value + 1)} />;
}

function locationRole() {
  const role = window.location.hash.slice(1);
  return role === "consumer" || role === "reviewer" ? role : null;
}

function App() {
  const [role, setRole] = useState(locationRole);
  useEffect(() => {
    const updateRole = () => setRole(locationRole());
    window.addEventListener("hashchange", updateRole);
    return () => window.removeEventListener("hashchange", updateRole);
  }, []);
  useEffect(() => {
    document.title = role ? workspaceCopy[role].title + " · BeautyProof" : "BeautyProof · 美妆证据审核";
  }, [role]);
  return role ? <ChatWorkspace key={role} role={role} /> : <Portal />;
}

createRoot(document.getElementById("workbench")).render(
  <ConfigProvider locale={zhCN} theme={theme}><App /></ConfigProvider>,
);

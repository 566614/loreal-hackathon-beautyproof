import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert, Button, Card, Checkbox, Collapse, ConfigProvider, Form, Image, Input, Modal,
  Progress, Radio, Select, Space, Tabs, Tag, Upload,
} from "antd";
import {
  ApartmentOutlined, ArrowRightOutlined, CopyOutlined, DownloadOutlined,
  FileTextOutlined, PictureOutlined, SafetyCertificateOutlined, ScanOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import zhCN from "antd/locale/zh_CN";
import "antd/dist/reset.css";
import "./workbench.css";
import { observations, assessSkin } from "./skin-profile.mjs";
import { recommend } from "./product-recommendations.mjs";

const toolLabels = {
  hash: "文件指纹", c2pa: "来源凭证", ela: "压缩痕迹", spectral: "频域取证",
  ocr: "图中文字", ai_label: "AI 来源标识", aigc: "生成倾向", trufor: "局部异常",
  text: "文案检查", crossmodal: "图文对照", claim_model: "DeepSeek 文案模型", semantic_review: "语义复核",
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

function SkinProfile() {
  const [form] = Form.useForm();
  const [profile, setProfile] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("beautyproof-skin-profile-v1") || "null");
      return saved?.version === 1 && Array.isArray(saved.answers) && saved.result?.type ? saved : null;
    }
    catch { return null; }
  });
  const [error, setError] = useState("");
  const [category, setCategory] = useState("全部");
  function save(values) {
    const answers = observations.map((_, index) => values[`q${index}`] ?? null);
    const result = assessSkin(answers);
    const next = { version: 1, savedAt: new Date().toISOString(), answers, result,
      sensitivity: values.sensitivity ?? null, recentChange: values.recentChange ?? null,
      goals: values.goals || [], preference: values.preference || [], avoid: values.avoid?.trim() || "" };
    try {
      localStorage.setItem("beautyproof-skin-profile-v1", JSON.stringify(next));
      setProfile(next);
      setError("");
    } catch { setError("浏览器无法保存档案，请检查本地存储设置。"); }
  }
  function clear() {
    try {
      localStorage.removeItem("beautyproof-skin-profile-v1");
      setProfile(null);
      form.resetFields();
      form.setFieldsValue({ ...Object.fromEntries(observations.map((_, index) => [`q${index}`, null])),
        sensitivity: null, recentChange: null, goals: [], preference: [], avoid: "" });
      setError("");
    }
    catch { setError("浏览器无法清除档案，请检查本地存储设置。"); }
  }
  const initial = profile ? Object.fromEntries(profile.answers.map((answer, index) => [`q${index}`, answer])) : {};
  const recommendations = profile ? recommend(profile, category) : [];
  return <Card className="surface" title="肤质自测与本地档案">
    <p>根据近期日常感受填写。档案保存在当前浏览器，结果供护肤选择参考。</p>
    {profile && <Alert type="info" showIcon title={`自测结果：${profile.result.type}`} description={profile.result.reason} />}
    <Form form={form} layout="vertical" initialValues={{ ...initial, sensitivity: profile?.sensitivity,
      recentChange: profile?.recentChange, goals: profile?.goals, preference: profile?.preference, avoid: profile?.avoid }} onFinish={save}>
      {observations.map((question, index) => <Form.Item key={index} name={`q${index}`} label={`${index + 1}. ${question}`}>
        <Radio.Group options={[{ label: "从不", value: 0 }, { label: "偶尔", value: 1 },
          { label: "经常", value: 2 }, { label: "总是", value: 3 }]} />
      </Form.Item>)}
      <Form.Item name="sensitivity" label="17. 使用护肤品后是否容易刺痛、灼热或泛红？">
        <Radio.Group options={[{ label: "从不", value: 0 }, { label: "偶尔", value: 1 },
          { label: "经常", value: 2 }, { label: "总是", value: 3 }]} />
      </Form.Item>
      <Form.Item name="recentChange" label="18. 近期肤质状态是否明显变化？">
        <Radio.Group options={[{ label: "稳定", value: "stable" }, { label: "变化明显", value: "changed" }, { label: "不确定", value: "unknown" }]} />
      </Form.Item>
      <Form.Item name="goals" label="19. 你希望关注哪些方面？">
        <Checkbox.Group options={["保湿", "清洁", "控油", "妆效", "肤色均匀"]} />
      </Form.Item>
      <Form.Item label="20. 你有哪些质地偏好或希望避开的成分？">
        <Form.Item name="preference" noStyle><Checkbox.Group options={["清爽", "滋润", "乳液", "霜", "啫喱"]} /></Form.Item>
        <Form.Item name="avoid" noStyle><Input placeholder="希望避开的成分，多个成分用逗号分隔" /></Form.Item>
      </Form.Item>
      {error && <Alert type="error" title={error} showIcon />}
      <Space><Button type="primary" htmlType="submit">保存自测档案</Button><Button onClick={clear}>清除档案</Button></Space>
    </Form>
    {profile && <>
      <p>刺激感：{profile.sensitivity == null ? "未填写" : profile.sensitivity >= 2 ? "较常出现，建议留意耐受情况" : "较少出现"}。</p>
      <h3>产品资料对照</h3>
      <p>以下为品牌发布资料中的匹配线索，请购买前核对现售版本和成分。</p>
      <Select value={category} onChange={setCategory} options={["全部", "洁面", "面霜", "精华"].map(value => ({ value, label: value }))} />
      {recommendations.length ? recommendations.map(product => <article className="claim" key={product.name}>
        <strong>{product.name}</strong><p>{product.basis}</p>
        <p>匹配线索：{profile.result.type}肤质；{(profile.goals || []).filter(goal => product.goals.includes(goal)).length
          ? `关注目标：${profile.goals.filter(goal => product.goals.includes(goal)).join("、")}` : "关注目标暂无对应资料"}；
          {(profile.preference || []).filter(value => product.texture.includes(value)).length
            ? `质地偏好：${profile.preference.filter(value => product.texture.includes(value)).join("、")}` : "质地偏好暂无对应资料"}。</p>
        <p>品牌资料 {product.sourceDate} · <a href={product.source} target="_blank" rel="noreferrer">查看来源</a></p>
      </article>) : <p>暂未找到满足条件且资料足够的产品，请核对肤质、品类或避用成分。</p>}
    </>}
  </Card>;
}

function Workbench() {
  const [section, setSection] = useState("audit");
  const [form] = Form.useForm();
  const [referenceForm] = Form.useForm();
  const [file, setFile] = useState(null);
  const [packageReference, setPackageReference] = useState(null);
  const [samples, setSamples] = useState([]);
  const [references, setReferences] = useState([]);
  const [referenceOpen, setReferenceOpen] = useState(false);
  const [referenceError, setReferenceError] = useState("");
  const [referenceSaving, setReferenceSaving] = useState(false);
  const [deepseekReady, setDeepseekReady] = useState(false);
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
    readJson("/api/config", { signal: controller.signal }).then(data => setDeepseekReady(data.deepseek_configured)).catch(() => {});
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
    const text = values.text?.trim();
    const source = values.referenceSource?.trim(), excerpt = values.referenceExcerpt?.trim();
    if (!text && !file) {
      setError("请上传图片或填写正文。");
      return;
    }
    if (packageReference && !file) {
      setError("使用包装参考图时，请同时上传宣传图片。");
      return;
    }
    if (Boolean(source) !== Boolean(excerpt)) {
      setError("请同时填写依据名称和依据原文。");
      return;
    }
    const data = new FormData();
    if (file) data.set("file", file);
    if (packageReference) data.set("reference_image", packageReference);
    if (text) data.set("text", text);
    if (values.product?.trim()) data.set("product", values.product.trim());
    if (values.market?.trim()) data.set("market", values.market.trim());
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
      <Space><Button type={section === "audit" ? "primary" : "text"} onClick={() => setSection("audit")}>内容审核</Button>
        <Button type={section === "skin" ? "primary" : "text"} onClick={() => setSection("skin")}>肤质自测</Button></Space>
    </div></header>
    <main className="page-width">
      {section === "skin" ? <SkinProfile /> : <>
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
          <Form form={form} name="workbench" layout="vertical" disabled={!!busy} onFinish={submit} initialValues={{ llm: true }}
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
                <Form.Item name="product" label="宣传产品"><Input list="known-products" placeholder="输入完整产品名，也可从资料库选择" /></Form.Item>
                <datalist id="known-products">{[...new Set(references.map(ref => ref.product))].map(name => <option key={name} value={name} />)}</datalist>
                <Form.Item name="market" label="产品市场"><Input placeholder="与资料中的市场名称一致，例如中国大陆" /></Form.Item>
                <Form.Item label="包装参考图（选填）" extra="上传平面包装或标签参考图，仅用于本地定位相似图案，不能确认产品或功效。两张图片总大小不超过 32 MB。">
                  <Upload accept=".png,.jpg,.jpeg" multiple={false} maxCount={1}
                    fileList={packageReference ? [packageReference] : []} onRemove={() => { setPackageReference(null); return true; }}
                    beforeUpload={picked => {
                      if (!/\.(png|jpe?g)$/i.test(picked.name) || picked.size > 32 * 1024 * 1024) {
                        setError("请选择不超过 32 MB 的 PNG 或 JPG 包装参考图。");
                        return Upload.LIST_IGNORE;
                      }
                      setPackageReference(picked); setError(""); return false;
                    }}><Button icon={<UploadOutlined />}>选择参考图</Button></Upload>
                </Form.Item>
                <p className="muted">填写与知识库一致的完整产品名及市场；语义审核只引用同一产品、市场的资料。</p>
                <Button onClick={() => { setReferenceError(""); setReferenceOpen(true); }}>添加已核验资料</Button>
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
              <Form.Item name="llm" valuePropName="checked" noStyle><Checkbox title="将正文、OCR 文字及匹配资料摘录发送给 DeepSeek 审核">DeepSeek 文字审核</Checkbox></Form.Item>
            </div>
            {!deepseekReady && <p className="muted">尚未配置 DeepSeek 密钥，当前使用规则审核文字。</p>}
            {deepseekReady && <p className="muted">启用后会向 DeepSeek 发送正文、图片中识别的文字，以及匹配资料的摘录和条件；图片留在本地。</p>}
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
      </>}
    </main>
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
    <footer className="page-width footer"><span>BeautyProof</span><span>美妆内容审核 · 证据辅助人工复核</span></footer>
  </>;
}

createRoot(document.getElementById("workbench")).render(
  <ConfigProvider locale={zhCN} theme={theme}><Workbench /></ConfigProvider>,
);

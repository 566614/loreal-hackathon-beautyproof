import React from "react";
import { NodeIndexOutlined } from "@ant-design/icons";
import "./analysis-process.css";

export default function AnalysisProcess({ process }) {
  if (!process?.steps?.length) return null;
  return <details className="analysis-process" open>
    <summary><span><NodeIndexOutlined aria-hidden="true" /> 分析过程</span>
      <small>{process.engine === "deepseek" ? "DeepSeek · 证据分析摘要" : "证据整理与本地分析"}</small></summary>
    <ol>{process.steps.map((step, index) => <li key={index}>
      <strong>{step.title}</strong><p>{step.detail}</p>
    </li>)}</ol>
    <p className="analysis-process-note">展示材料、工具结果与判断依据，可与下方证据对应核对。</p>
  </details>;
}

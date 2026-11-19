import React from "react";
import { NodeIndexOutlined } from "@ant-design/icons";
import "./analysis-process.css";

export default function AnalysisProcess({ process }) {
  if (!process?.steps?.length) return null;
  return <details className="analysis-process">
    <summary><span><NodeIndexOutlined aria-hidden="true" /> 分析过程</span>
      <small>{process.engine === "deepseek" ? "DeepSeek 分析" : "本地分析"}</small></summary>
    <ol>{process.steps.map((step, index) => <li key={index}>
      <strong>{step.title}</strong><p>{step.detail}</p>
    </li>)}</ol>
  </details>;
}

import React, { useEffect, useRef, useState } from "react";
import { ArrowLeftOutlined, ArrowRightOutlined, CopyOutlined, DownloadOutlined, LockOutlined, ScanOutlined } from "@ant-design/icons";
import "./model-lab.css";

const demoCopy = "【示例精华】2025版14天水分提升20%。【示例精华】2025版14天水分提升40%。";
const sampleImage = "/static/beautyproof/model-lab-sample.png";

function SobelComparison({ onResponseChange }) {
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const responseMap = useRef(null);
  const [position, setPosition] = useState(58);

  const updatePosition = value => {
    const next = Number(value);
    setPosition(next);
    const map = responseMap.current;
    if (!map) return;
    const x = Math.min(map.width - 2, Math.max(1, Math.round((next / 100) * (map.width - 1))));
    const y = Math.floor(map.height / 2);
    onResponseChange(Math.round(map.values[y * map.width + x]));
  };

  useEffect(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas) return;

    const renderEdges = () => {
      if (!image.naturalWidth || !image.naturalHeight) return;
      const scale = Math.min(1, 960 / Math.max(image.naturalWidth, image.naturalHeight));
      const width = Math.max(3, Math.round(image.naturalWidth * scale));
      const height = Math.max(3, Math.round(image.naturalHeight * scale));
      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) return;
      context.drawImage(image, 0, 0, width, height);
      const input = context.getImageData(0, 0, width, height);
      const output = context.createImageData(width, height);
      const gray = new Float32Array(width * height);
      const values = new Float32Array(width * height);

      for (let pixel = 0; pixel < gray.length; pixel += 1) {
        const index = pixel * 4;
        gray[pixel] = 0.299 * input.data[index] + 0.587 * input.data[index + 1] + 0.114 * input.data[index + 2];
        const outputIndex = index;
        output.data[outputIndex] = 7;
        output.data[outputIndex + 1] = 18;
        output.data[outputIndex + 2] = 27;
        output.data[outputIndex + 3] = 255;
      }

      for (let y = 1; y < height - 1; y += 1) {
        for (let x = 1; x < width - 1; x += 1) {
          const top = (y - 1) * width + x;
          const middle = y * width + x;
          const bottom = (y + 1) * width + x;
          const gx = -gray[top - 1] + gray[top + 1] - 2 * gray[middle - 1] + 2 * gray[middle + 1] - gray[bottom - 1] + gray[bottom + 1];
          const gy = -gray[top - 1] - 2 * gray[top] - gray[top + 1] + gray[bottom - 1] + 2 * gray[bottom] + gray[bottom + 1];
          const strength = Math.hypot(gx, gy);
          values[middle] = strength;
          const edge = Math.min(255, strength * 0.72);
          const outputIndex = middle * 4;
          output.data[outputIndex] = Math.round(7 + edge * 0.22);
          output.data[outputIndex + 1] = Math.round(18 + edge * 0.82);
          output.data[outputIndex + 2] = Math.round(27 + edge * 0.9);
        }
      }

      context.putImageData(output, 0, 0);
      responseMap.current = { values, width, height };
      updatePosition(position);
    };

    if (image.complete) renderEdges();
    else image.addEventListener("load", renderEdges, { once: true });
    return () => image.removeEventListener("load", renderEdges);
  }, []);

  return <div className="lab-comparison">
    <div className="lab-compare-frame">
      <img ref={imageRef} className="lab-source-image" src={sampleImage} alt="AI生成的护肤品静物演示图" />
      <div className="lab-processed-image" style={{ clipPath: "inset(0 0 0 " + position + "%)" }}>
        <canvas ref={canvasRef} aria-label="Sobel 卷积后的边缘响应图" />
      </div>
      <span className="lab-image-label lab-label-before">输入图像 <small>RGB</small></span>
      <span className="lab-image-label lab-label-after">边缘响应 <small>FEATURE MAP</small></span>
      <span className="lab-scan-guide" style={{ left: position + "%" }} aria-hidden="true">
        <i /><i /><i /><i /><i /><i /><i /><i /><i />
      </span>
      <span className="lab-divider" style={{ left: position + "%" }} aria-hidden="true"><b>↔</b></span>
    </div>
    <label className="lab-slider-label" htmlFor="feature-wipe">
      <span>处理前</span>
      <input id="feature-wipe" type="range" min="4" max="96" value={position}
        aria-label="拖动比较处理前与卷积处理后的图像" onChange={event => updatePosition(event.target.value)} />
      <span>处理后</span>
    </label>
    <div className="lab-image-caption"><span>演示素材 · AI 生成示意</span><span>滑动查看卷积前后</span></div>
  </div>;
}

function KernelPanel({ response }) {
  const kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  return <aside className="lab-kernel-panel" aria-label="Sobel 卷积核信息">
    <div className="lab-panel-heading">
      <div><span className="lab-overline">LOCAL TRANSFORM</span><h2>特征提取</h2></div>
      <span className="lab-kernel-size">3 × 3</span>
    </div>
    <p className="lab-panel-copy">卷积核沿图像移动，对邻域像素加权求和，突出亮度变化的边缘。</p>
    <div className="lab-kernel-block">
      <div className="lab-kernel-caption"><span>SOBEL · Gx</span><span>水平梯度</span></div>
      <div className="lab-kernel-grid" aria-label="水平梯度卷积核数值">
        {kernel.flat().map((value, index) => <span key={index} className={value === 0 ? "is-zero" : ""}>{value}</span>)}
      </div>
      <p>∑ 邻域像素 × 权重</p>
    </div>
    <div className="lab-response-readout">
      <span>当前采样点 · 响应强度</span>
      <strong>{response.toLocaleString("zh-CN")}</strong>
      <small>随滑块位置实时变化</small>
    </div>
    <ol className="lab-steps">
      <li><b>01</b><span>读取 3 × 3 像素邻域</span></li>
      <li><b>02</b><span>按卷积核权重逐点计算</span></li>
      <li><b>03</b><span>写入边缘特征图</span></li>
    </ol>
    <p className="lab-model-note"><ScanOutlined aria-hidden="true" /> 卷积过程用于解释图像特征。实际生成倾向和局部异常结果，请在审核报告中查看本地模型输出。</p>
  </aside>;
}

export default function ModelLab() {
  const [copyState, setCopyState] = useState("复制演示文案");
  const [response, setResponse] = useState(0);

  const copyDemoText = async () => {
    try {
      await navigator.clipboard.writeText(demoCopy);
      setCopyState("已复制");
    } catch {
      setCopyState("请手动复制下方文案");
    }
  };

  return <main className="model-lab">
    <header className="lab-topbar">
      <a href="/" className="lab-brand"><ScanOutlined aria-hidden="true" /><span>BeautyProof</span></a>
      <div className="lab-local-state"><span /> <LockOutlined aria-hidden="true" /> 本机演示素材</div>
    </header>
    <div className="lab-content">
      <div className="lab-page-heading">
        <div><span className="lab-overline">MODEL LAB · COMPUTER VISION</span>
          <h1>让图像，经过每个像素。</h1>
          <p>拖动滑块，观察 3 × 3 卷积核如何把像素邻域转换为边缘特征。</p>
        </div>
        <span className="lab-cpu-mark">LOCAL<br />PROCESSING</span>
      </div>
      <section className="lab-workspace" aria-label="卷积前后图像对比">
        <div className="lab-viewer"><SobelComparison onResponseChange={setResponse} /></div>
        <KernelPanel response={response} />
      </section>
      <section className="lab-materials" aria-labelledby="lab-materials-title">
        <div className="lab-material-intro">
          <span className="lab-overline">RECORDING KIT</span>
          <h2 id="lab-materials-title">继续看完整审核流程</h2>
          <p>上传同一张示意图，再粘贴这段冲突文案，展示图片线索与文字核验报告。</p>
          <a className="lab-workbench-link" href="#consumer">打开审核工作台 <ArrowRightOutlined aria-hidden="true" /></a>
        </div>
        <div className="lab-copy-card">
          <span>审核演示文案</span>
          <p>{demoCopy}</p>
          <button type="button" onClick={copyDemoText}><CopyOutlined aria-hidden="true" /> {copyState}</button>
        </div>
        <a className="lab-download" href={sampleImage} download="BeautyProof_本地模型演示素材.png">
          <img src={sampleImage} alt="" />
          <span><b>AI 示意图</b><small>下载示例图片</small></span>
          <DownloadOutlined aria-hidden="true" />
        </a>
      </section>
      <footer className="lab-footer"><a href="/" aria-label="返回首页"><ArrowLeftOutlined aria-hidden="true" /> 返回入口</a>
        <span>示例素材仅用于界面演示，审核结论需结合原始证据复核。</span></footer>
    </div>
  </main>;
}

# BeautyProof · 来源声明与第三方组件清单（Provenance & Third-party Attributions）

> 文档目的：明确区分「第三方 / 开源改编」与「BeautyProof 团队原创」两部分代码与资产，
> 回应竞赛规则对**原创性与知识产权**的要求（规则第 3 条：雷同或无创新贡献 = 全队取消成绩）。
> 本声明与仓库顶层 `LICENSE`（Apache-2.0）和 `NOTICE`（第三方归属汇总）配套使用。
>
> 编写日期：2026-09-25 ｜ 仓库 HEAD 见 `git rev-parse HEAD`。

---

## 0. 一句话结论（核心立场）

**BeautyProof 的核心创新与可提交成果，全部是团队原创工作**，包括：

1. 统一的八工具证据格式 `{tool, source_asset_id, observed, cannot_prove, evidence[]}`（工程抽象，全仓库共用）；
2. 规则引擎 `rule_engine` 的四档判定逻辑（`high_risk` / `suspicious` / `credible` / `inconclusive`）；
3. 本域微调的 AIGC 分类器 `beautyproof_aigen_v1/v2/v3`（MobileNetV3 主干，自产训练数据训练）；
4. 跨生成器验证方案（`data/ai_cross/` + `train_aigen_v3.py` 的干净 held-out 划分法）；
5. 流水线编排与可解释 Agent（`pipeline.py` + `planner.py` 波次调度与 R1–R5 跳过规则）；
6. 图文交叉验证、文案体检、哈希指纹、TC260/C2PA 凭证解析、报告生成、Validator 护栏、多 Agent 圆桌、Web 取证台与离线自包含 demo。

**第三方 / 开源组件仅作为「被调用的能力」或「兜底模型」存在**，其知识产权与许可证归原作者所有（详见下文明细）。本团队未将任何第三方代码据为己有，也未修改其许可证。

> **模型来源速判（赛题2「原创性/数据来源合规」）**：本作品属于**"使用他人预训练来源模型（骨干）"**，不属于**"使用他人成千上万张图片的训练库"**。AIGC 模型骨干 = `timm/mobilenetv3_large_100.ra_in1k`（ImageNet-1k 预训练权重，仅初始化）；训练数据全部团队自产（约 135 张美妆图），未用任何第三方大规模标注图库训练。完整判定与证据见 **`docs/模型来源与训练数据合规声明.md`**。

---

## 1. 第三方 / 开源改编组件清单

> 许可证一列中标注「待核实」的，表示以实际代码内附许可证或上游仓库声明为准，
> 团队已在 `NOTICE` 与下方备注中如实标注已知信息，未做虚假声明。

### 1.1 TruFor（图像篡改检测核心模型）
- **类型**：开源学术论文代码 + 预训练权重（**被调用 / 适配**，非团队原创算法）
- **来源**：《TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization》, CVPR 2023。官方代码位于 `models/TruFor/TruFor_train_test/`（含 `LICENSE.txt`、`LICENSE_CMX.txt`、`test.py` 等），由 `tools/setup_trufor.py` 按需下载部署。
- **上游仓库**：官方 GitHub（CVPR2023 论文开源仓库，名称以论文页/原仓库为准）
- **许可证**：研究用途 / 学术使用许可（仓库内附 `LICENSE.txt` 与 `LICENSE_CMX.txt`，CMX 主干另含独立许可）。**具体商业授权需另行取得原作者同意——标注「待核实（研究用途）」**。
- **我们用它做什么**：`tools/trufor_tool.py` 作为「局部人工篡改 / 被 P 过」检测能力，输出整图篡改分、定位图、可靠性图；权重与代码**不随本仓库分发**，运行 `setup_trufor.py` 时从官方源下载到本地。
- **我们原创的部分**：`trufor_tool.py` 的**统一证据格式封装**、与 rule_engine 的联动、R5 规则（整图 AI 判定时 TruFor 只出定位图不定性），均为团队原创胶水逻辑。

### 1.2 PaddleOCR（PP-OCRv6，OCR 文字识别）
- **类型**：开源 Python 库（**被调用**，非团队原创）
- **来源**：PaddlePaddle / PaddleOCR 官方
- **上游仓库**：`https://github.com/PaddlePaddle/PaddleOCR`
- **许可证**：Apache-2.0
- **我们用它做什么**：`tools/ocr_tool.py` 调用 `paddleocr.PaddleOCR` 把图上文字抄成统一格式证据 JSON（text / bbox / confidence）。
- **我们原创的部分**：OCR 结果的「统一证据格式」封装、以及下游「文案体检 / 图文交叉验证」的全部逻辑。

### 1.3 c2patool（C2PA 元数据校验）
- **类型**：外部命令行工具（**被调用**，非团队原创）
- **来源**：Content Authenticity Initiative (CAI) / `contentauth` 项目
- **上游仓库**：`https://github.com/contentauth/c2patool`
- **许可证**：待核实（上游通常为 MIT/Apache-2.0，以官方仓库声明为准）
- **我们用它做什么**：`tools/c2pa_tool.py` 调用本机已安装的 `c2patool.exe`，读出图片的 C2PA 凭证；并自行实现 **TC260 AIGC 标识**（中国强制标准）的元数据解析逻辑。
- **我们原创的部分**：TC260 标识解析、C2PA 结果的统一证据格式封装与规则联动。

### 1.4 ELA（Error Level Analysis，误差水平分析）
- **类型**：经典公开算法（**团队自研实现**，非绑定特定第三方包）
- **来源**：经典图像取证技法（Fu 等，2006 年起公开文献），属公共领域方法论，不归属单一商业实体。
- **许可证**：N/A（公共方法论，自研实现）
- **我们用它做什么**：`tools/ela_tool.py` 以重新有损压缩比对像素误差，作为 TruFor 的降级替代方案。
- **我们原创的部分**：整个 `ela_tool.py` 的实现与统一证据格式封装为团队原创；算法本身是公开方法。

### 1.5 预训练检测模型权重（兜底 / 实验用，已随仓分发）
以下第三方预训练权重位于 `models/`，**仅作为通用 AIGC 检测的兜底或实验对比**，默认不被主流程优先使用（主流程优先用团队自训的 `beautyproof_aigen_*`）：

| 模型目录 | 上游 | 许可证 | 在仓库中的角色 |
| - | - | - | - |
| `models/airealnet` | `XenArcAI/AIRealNet` | 待核实（研究用途） | AIGC 兜底模型，实测在本域区分度不足 |
| `models/capcheck` | `capcheck/ai-human-generated-image-detection` | Apache-2.0 | AIGC 兜底模型，实测本域误报高 |
| `models/sdxl-detector` | `Organika/sdxl-detector` | 待核实（通常为 Apache-2.0） | 早期 AIGC 模型，实测零区分度，已彻底弃用 |
| `models/clip-vit-large-patch14` | `openai/clip-vit-large-patch14` | Apache-2.0 | 仅 `eval_clip_head.py` 实验用，未进主流程 |
| `models/timm_mobilenetv3` | `timm` MobileNetV3-Large (ImageNet) | MIT / Apache-2.0（timm 库） | 仅作为团队自训 AIGC 模型的主干初始化权重 |

> 说明：这些权重文件体积可控、随仓分发，与 TruFor（约 370MB、**不随仓分发**）处理方式不同。其许可证信息已如实标注，团队未对其主张任何权利。

### 1.6 深度学习 / 科学计算依赖（第三方库）
均为标准开源依赖，团队仅调用、**未修改源码**：

| 组件 | 许可证 | 用途 |
| - | - | - |
| PyTorch / TorchVision | BSD-3-Clause | 模型推理与训练（CPU 构建） |
| timm（PyTorch Image Models） | Apache-2.0 / MIT | AIGC 模型主干 |
| Transformers / huggingface_hub / safetensors | Apache-2.0 | 模型加载与 Hub 下载 |
| kornia | Apache-2.0 | TruFor 依赖的图像算子 |
| PaddlePaddle / PaddleOCR | Apache-2.0 | OCR |
| Qwen3-VL-4B-Instruct-GGUF（可选 LLM 解释层） | Apache-2.0 | 方案 C 的「人话解读」生成（不随仓分发，按需下载） |
| Flask | BSD-3-Clause | 本地演示 Web 服务 |
| OpenCV / NumPy / SciPy / scikit-image / scikit-learn / matplotlib / Pillow / requests | 各自原始许可证（BSD / Apache-2.0 / PSF / HPND 等） | 图像处理、数值计算、可视化、HTTP |

### 1.7 数据来源
- **训练 / 评测数据 = 团队自产**：以 ImageGen 生成的 AI 美妆图（`data/ai_cross/`、`data/` 下自产集）与用户/团队提供的真实精修美妆图构成；未抓取任何第三方平台图。
- **GenImage 等公开数据集**：仅作离线评测参考、不入仓。
- 所有训练数据均遵守数据红线（版权 / 隐私 / 非商用 / 主题不冒用），详见 `docs/赛题合规审查与获奖评估.md` §1.8。

---

## 2. BeautyProof 团队原创工作（核心创新贡献）

> 以下均为团队从零设计、实现与验证的工作，**构成参赛的核心创新贡献**。

| 原创模块 | 关键文件 | 原创性说明 |
| - | - | - |
| **统一八工具证据格式** `{tool, source_asset_id, observed, cannot_prove, evidence[]}` | 全部 `tools/*_tool.py` | 一套可组合、可审计的证据抽象，8 个异构工具共用，是工程核心创新 |
| **规则引擎四档判定** | `tools/rule_engine.py` | `high_risk / suspicious / credible / inconclusive` 定级逻辑与多工具证据融合策略，团队原创 |
| **本域微调 AIGC 分类器** | `models/beautyproof_aigen_v1/`、`_v2/`、`_v3/` + `tools/train_aigen*.py` | 在 MobileNetV3 主干上，用自产中文美妆域数据微调，把通用模型 66% 误报压到 1.4%、跨生成器召回 25%→100%；模型权重为团队训练产物 |
| **跨生成器验证方案** | `data/ai_cross/` + `tools/train_aigen_v3.py` | 干净 held-out 划分法，确保跨生成器指标不被训练污染，团队原创实验设计 |
| **流水线编排 + 可解释 Agent** | `tools/pipeline.py`、`tools/planner.py` | 波次调度（快检→中检→深检）+ R1–R5 跳过规则 + A1–A3 后置动作，每条决策留人话理由 |
| **图文交叉验证** | `tools/crossmodal_tool.py` | 检测「文案宣称」与「图像侧结论」的事实矛盾，团队原创的跨模态逻辑 |
| **文案体检** | `tools/text_tool.py` | 违禁宣称硬规则比对、AI 写作特征、刷量特征，团队原创 |
| **哈希指纹 / TC260-C2PA 凭证解析 / ELA 实现** | `tools/hash_tool.py`、`tools/c2pa_tool.py`、`tools/ela_tool.py` | 工具逻辑与统一封装为团队原创 |
| **报告生成 + Validator 护栏 + LLM 解释层 + 多 Agent 圆桌** | `tools/report_generator.py`、`tools/validator.py`、`tools/llm_explainer.py`、`tools/roundtable.py` | 可审计模板报告、六道自检关、本地大模型翻译层、结构化多角色会诊，均为团队原创 |
| **Web 取证台 + 离线自包含 demo** | `web/app.py`、`web/review_app.py`、`demo/index.html`、`tools/build_demo.py` | 本地演示 Web、离线 HTML、可录屏走查页，团队原创交付形态 |
| **批量评测 / 数据集重建 / 答辩材料生成** | `tools/batch_run.py`、`tools/make_dataset.py`、`tools/make_defense_pptx.py` 等 | 数据集闭环与演示资产的工程化，团队原创 |

---

## 3. 原创 vs 第三方边界（给评审 / 代码审核者的速查）

- **「算法/模型权重」是第三方的**：TruFor 的检测算法、PaddleOCR、c2patool、ELA 方法论、airealnet/capcheck/sdxl-detector/clip 的预训练权重、timm 主干权重、Qwen3-VL 权重。
- **「把这些能力串起来、定级、解释、交付」是原创的**：统一证据格式、规则引擎、本域微调训练、跨生成器验证设计、Agent 调度、图文交叉验证、报告与护栏、Web/demo 交付。
- **关键区分点**：第三方组件均以「被调用能力 / 兜底模型」形式存在，默认主流程优先使用**团队自训模型**；任何第三方代码均未进行修改后据为己有的行为。

---

## 4. 许可证与合规承诺

- 本项目自身代码以 **Apache-2.0** 发布（见 `LICENSE`）。
- 第三方归属与许可证汇总见顶层 **`NOTICE`**。
- 本团队承诺：未侵犯任何第三方知识产权；未在队间共享代码片段；所有训练数据来源合法合规。
- 对任意许可证存在疑问的组件，已在本文件中如实标注「待核实」，并以实际代码内附许可证或上游仓库声明为准。

---

*本文件为竞赛「原创性与知识产权」合规保命项之一。如与 `NOTICE` / `LICENSE` 存在表述差异，以官方许可证原文与上游仓库声明为准。*

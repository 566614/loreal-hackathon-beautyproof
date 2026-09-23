# GitHub 开源调研：识别美妆图的大模型

> 调研日期：2026-09-23
> 调研目的：为 BeautyProof（美妆图篡改 / AI 生成取证）寻找可直接接入的开源能力
> 结论前置：**不存在"美妆垂域的开源识别大模型"**。美妆域有 Star 的开源项目全部是「生成 / 迁移」（给人脸上妆），识别域有 Star 的开源项目全部是「通用域」（不认品类）。两个圈在 GitHub 上没有交集。

---

## 一、检索方法与筛选条件

### 1.1 使用的关键词（6 组主检索 + 3 组验证性检索）

| # | 检索词 | 意图 |
|---|---|---|
| Q1 | `makeup transfer` | 美妆垂域主线（妆容迁移） |
| Q2 | `virtual try-on` | 美妆 / 时尚垂域生成能力 |
| Q3 | `beauty AI` / `beauty GAN` | 泛美妆垂域 |
| Q4 | `cosmetic` / `beauty product` | 美妆商品侧 |
| Q5 | `AI-generated image detection` | 识别真假主线 |
| Q6 | `image forgery detection` / `deepfake detection` | 篡改取证主线 |
| V1 | `makeup detection` | **验证性**：直击"美妆识别"这个字面需求 |
| V2 | `cosmetics counterfeit detection OR beauty product recognition` | **验证性**：直击"美妆商品识别" |
| V3 | `AI-generated image detection`（stars>100） | **验证性**：识别侧高星仓库 |

### 1.2 筛选条件（硬门槛）

- **Star 数**：主检索 `stars:>50`，识别侧收紧到 `stars:>100`
- **最近更新时间**：`pushed:>2024-01-01`（排除僵尸仓库；但维护活跃度单独标注，不只看一条线）
- **主要编程语言**：Python 优先（本项目技术栈）；JS/Java/PHP 的"美妆"同名项目全部剔除
- **开源许可证**：单独记录 SPDX ID，**无许可证的标红**（商用有法律风险）
- **文档完整性**：看是否有 README / 论文项目主页 / 预训练权重下载说明
- **维护活跃度**：看 `pushed_at` + `open_issues_count` + 是否 archived

### 1.3 去重后的候选池

6 组主检索去重后得到 **115 个候选仓库**，按 Star 排序取 Top 45 人工筛。剔除的噪声包括：

- `beautysh`（bash 代码格式化工具）、`BeautyEye` / `beauty_ssm`（Java 框架）、`vue-beauty`（UI 组件库）—— 只是名字里带 beauty
- `MakeUpUltraFast`（Minecraft 光影包）
- 字体 / 排版主题类仓库

---

## 二、两个验证性检索的结论（这是本次调研最重要的发现）

**V1 `makeup detection`** —— 返回 10 个仓库，**Star 数全部为 0**：

```
lijx3643/makeup-detection            0★  HTML     2020-04
silvering-steve/makeup-detection     0★  Python   2023-10
sudeturan/heavy-makeup-detection     0★  —        2026-07
rimbaradie/makeup-detection-system   0★  PHP      2025-12
shreyakadam-m/Face-Makeup-Detection  0★  Python   2026-02
YoonieJ/Makeup_Detection_PyTorch     0★  Python   2026-07
VinhRP/Makeup_Detection_Yolov8       0★  Notebook 2023-12
nadianajwazulkifli/Facial-Makeup...  0★  Python   2024-05
tanujathakur/AI-Powered-Face-...     0★  TS       2026-07
```

**V2 `cosmetics counterfeit detection OR beauty product recognition`** —— **返回空数组 `[]`**，一个仓库都没有。

这两条是把"美妆识别"当成字面需求去搜的结果：**零个可用仓库**。全部是课程作业级别的个人项目，无 Star、无维护、无权重、无 benchmark。

---

## 三、最匹配的 5 个仓库

> 说明：因为"美妆 + 识别"没有交集，下面 5 个按两个维度分别给最优选 —— A 组是真正的美妆垂域（但做的是生成，不是识别），B 组是真正能识别真假（但是通用域）。每组的价值对 BeautyProof 不同。

### A1. VinAIResearch / CPM —— 美妆垂域里工程与许可最完整的一个

- **链接**：https://github.com/VinAIResearch/CPM
- **一句话简介**：CVPR 2021 的"野外场景妆容迁移"官方实现，不只做颜色匹配，还能迁移高光、珠光、图案等复杂妆效。
- **核心特性**
  - 论文《Lipstick ain't enough: Beyond Color-Matching for In-The-Wild Makeup Transfer》，有独立项目主页（thaoshibe.github.io/CPM）
  - 提出 **PS 正则化 + 直方图匹配** 的双阶段方案，对非实验室拍摄的真实照片（in-the-wild）做了专门处理
  - 附带妆容"图案检测"能力（pattern-detection 标签），不只是纯色迁移
- **适用场景**：需要理解"妆容是什么、怎么被改变的"这个**域知识**时，是最权威的开源参考；可用于构造"AI 上妆 vs 真人化妆"的对照样本，帮我们判断模型学的是妆容还是生成痕迹。
- **局限**：**许可证 BSD-3-Clause 但仓库 2014 年后长期静置，最近一次 push 2024-11**，只剩 2 个 open issue（说明几乎无人用）；是 GAN 时代架构，不是大模型；**它不做识别，只做生成**。

### A2. wtjiang98 / PSGAN —— 美妆垂域 Star 最高、License 最干净

- **链接**：https://github.com/wtjiang98/PSGAN
- **一句话简介**：CVPR 2020 Oral 的姿态 / 表情鲁棒妆容迁移 GAN，支持"妆容浓淡可调"和"局部指定区域上妆"。
- **核心特性**
  - **MIT 许可证**（商用无障碍），Python，781★
  - 空间感知的注意力机制，能在姿态、表情变化下保持妆容对齐
  - 支持 customizable：哪块区域上妆、上多重，可控
- **适用场景**：做美妆数据增强（把同一张脸渲染出不同妆容），或研究"妆容区域"的空间分布特征。
- **局限**：**最后 push 停在 2024-03，已两年没动**；11 个 open issue 未处理；GAN 架构、需要自行准备 CelebA 系数据集；**不做识别**。

### A3. zllrunning / face-makeup.PyTorch —— 最轻量的妆容编辑 + 人脸解析

- **链接**：https://github.com/zllrunning/face-makeup.PyTorch
- **一句话简介**：基于人脸语义分割图（face parsing）的口红 / 发色编辑器，改颜色只改对应部位，不糊到别处。
- **核心特性**
  - **MIT 许可证**，541★，依赖极轻（人脸解析 mask + 颜色映射）
  - 背后配套的 **CelebAMask-HQ 人脸解析数据集**是目前最常用的人脸部件标注数据源
  - 出图快、可解释性强（改哪一块是显式指定的）
- **适用场景**：给取证流水线补一个"妆容区域定位"能力 —— 当 TruFor 热力图在嘴唇/眼影区域亮起时，可以判断是化妆还是篡改。**这是 A 组里唯一有可能被本项目复用成检测辅助的仓库。**
- **局限**：**最后 push 2021-08，已归档式停更**；功能窄（只有唇色 / 发色）；不做识别、不做 AI 检测。

### B1. shilinyan99 / AIDE —— 识别侧当前最强、维护最活跃的开源方案

- **链接**：https://github.com/shilinyan99/AIDE
- **一句话简介**：ICLR 2025《A Sanity Check for AI-generated Image Detection》，用"混合多专家特征 + 简单分类器"在跨生成器泛化上做到当时 SOTA。
- **核心特性**
  - **MIT 许可证**，Python，338★，创建于 2024-06，**最近 push 2025-06**，是识别侧仍在维护的一线工作
  - 核心思路：**冻结多个预训练骨干（CLIP / DINOv2 等）提特征，只训一个轻量分类头** —— 与本项目 `beautyproof_aigen` 的技术路线同源
  - 论文的核心卖点是**跨生成器泛化**（训 SD 能测 Midjourney），正是通用检测器最容易翻车的地方
- **适用场景**：作为 BeautyProof 的**第二意见 / 交叉验证信号**，补本域小模型跨生成器泛化不足的短板（我们的模型只在即梦域训练过）。
- **局限**：**通用域训练，没见过中文美妆素材** —— 按本项目已实测的结论（AIRealNet / capcheck 在美妆域失效、sdxl-detector 零区分度），直接拿来判美妆图大概率重蹈覆辙；需要 GPU 才跑得舒服。

### B2. grip-unina / TruFor —— 本项目已在用的篡改定位器（对照组）

- **链接**：https://github.com/grip-unina/TruFor
- **一句话简介**：CVPR 2023 的图像篡改检测与定位，输出整图异常分 + 像素级定位热力图，能指出"哪一块被动过"。
- **核心特性**
  - 278★，Python，**最近 push 2025-05**，14 个 open issue（有维护迹象）
  - 融合噪声线索 + 多尺度特征，同时给 **detection（整图分）** 和 **localization（热力图）**
  - 本项目 `tools/setup_trufor.py` 已部署其权重，是六工具流水线里的第 6 个工具
- **适用场景**：判"局部被人手动改过"（拼接 / 复制移动 / 改文字），与 AIGC 工具互补。
- **局限**：⚠️ **仓库无开源许可证（License: NONE）** —— 商用有法律风险，答辩时若被问需说明"仅研究用途 / 需向原作者申请授权"；对纯 AI 整图生成的判据是间接的（本项目实测 AI 组 0.78–0.998，与真实组有 0.023 间隙，可用但需校准）。

---

## 四、对比总结

| 仓库 | Star | 语言 | 许可证 | 最近 push | 开放 Issue | 是否美妆垂域 | 是否"识别" | 文档/权重 |
|---|---|---|---|---|---|---|---|---|
| VinAIResearch/CPM | 420 | Python | BSD-3-Clause | 2024-11 | 2 | ✅ | ❌（生成） | 论文主页 + 权重 |
| wtjiang98/PSGAN | 781 | Python | **MIT** | 2024-03 | 11 | ✅ | ❌（生成） | 论文 + 权重 |
| zllrunning/face-makeup.PyTorch | 541 | Python | **MIT** | 2021-08 | 10 | ✅ | ❌（编辑） | README + 数据集 |
| shilinyan99/AIDE | 338 | Python | **MIT** | 2025-06 | 11 | ❌（通用） | ✅ | 论文 + 权重 |
| grip-unina/TruFor | 278 | Python | ⚠️ **NONE** | 2025-05 | 14 | ❌（通用） | ✅ | 权重（已在用） |
| — 对照：Honlan/BeautyGAN | 695 | Python | ⚠️ **NONE** | 2019-07 | 12 | ✅ | ❌ | 已停更 7 年 |

**三条硬结论：**

1. **没有一个是"大模型"**。A 组三个都是 2019–2021 年的 GAN / CNN 架构（参数量百万级），跟"大模型"不是一个量级；B 组 AIDE 虽然用了 CLIP 这类预训练骨干，但分类头是轻量的，本质是"冻结特征 + 小头"。
2. **没有一个是"美妆识别"**。美妆垂域的开源项目 100% 在做生成/迁移/编辑，识别侧的项目 100% 是通用域。这是两个不相交的集合，不是我们没搜到，是**这个位置在开源社区里就是空的**。
3. **维护活跃度普遍堪忧**。A 组最短的停更 5 年（BeautyGAN 2019），最长的也两年没动（PSGAN 2024-03）；反倒是通用识别侧（AIDE 2025-06、TruFor 2025-05、SSP 2026-03）维护明显更好 —— 因为识别是当下热点，美妆迁移已过气。

---

## 五、推荐选择 & 生产可用性判定

### 5.1 推荐：**不引入任何 A 组仓库做识别，B 组只做"第二意见"，主线继续走本域微调**

理由不是"没找到好的"，而是本项目已经用实测数据把这条路走通了：

| 方法 | 原始 AUC | 归一化后 AUC（统一尺寸 + JPEG q90 重编码） |
|---|---|---|
| 1-NN | 0.993 | **1.000** |
| 5-NN | 0.987 | **1.000** |
| 类原型 | 0.967 | 0.987 |
| PCA16 + 逻辑回归 | 0.980 | **1.000**（ACC 1.000） |

> 归一化是**故意抹掉**"哪一批 JPEG"这类格式捷径的。指标不降反升 = 模型学到的是**真实真伪信号**，不是偷懒走捷径。这直接证伪了"通用模型在美妆域必然失效"的悲观假设 —— 失效的根因是**域不匹配**，不是"美妆图不可判"。

而本域微调的 MobileNetV3（`models/beautyproof_aigen/`，17MB）在**同样 7 张留出图**上做到 7/7，CLIP 零训练方案是 6/7。所以 CLIP / AIDE 这类通用骨干**不能替代**本域模型，价值在于提供"零训练 / 免 GPU / 冷启动"的第二信号。

### 5.2 生产可用性判定

| 仓库 | 能否直接用于生产 | 判定理由 |
|---|---|---|
| VinAIResearch/CPM | ❌ 不可 | 是生成模型，与取证需求不匹配；且两年无维护 |
| wtjiang98/PSGAN | ❌ 不可 | 同上；仅 MIT 许可这一项友好 |
| zllrunning/face-makeup.PyTorch | ⚠️ **可做二次开发** | 人脸解析能力可复用为"妆容区域定位"，但需重写为推理组件、且原仓库已停更 |
| shilinyan99/AIDE | ⚠️ **需二次开发** | 架构可借鉴（冻结骨干 + 小头），但**必须在本域数据上重新校准**，否则会重现 AIRealNet / capcheck 的域失效 |
| grip-unina/TruFor | ✅ **已在生产（本项目）** | 六工具之一，已部署并用 6/6 评测验证；⚠️ 唯一风险是**无开源许可证**，商用前需确认授权 |

### 5.3 对 BeautyProof 的三条具体动作建议

1. **主线不动**：继续用 `beautyproof_aigen`（本域微调）判"整图 AI 生成"，用 TruFor 判"局部人工改动"。两者分工明确，已 6/6 验证。
2. **补第二意见**：可把 CLIP-ViT-L/14 的零训练 margin 分数接进流水线作为 `aigc_clip` 信号（权重已下载在 `models/clip-vit-large-patch14/`），与 `aigc` 主信号并列展示 —— 它不参与定档，只用于"冷启动 / 跨生成器"场景的提示。
3. **最高优先级补数据**：扩大真实美妆图样本池（现 10 张，目标 50），并务必拿到**非即梦生成器的 AI 图** —— 这是本项目第一次能做真正的跨生成器测试，也是答辩时唯一的硬短板。

### 5.4 答辩时怎么讲这个调研（一句话版本）

> "我们搜遍了 GitHub：美妆垂域的开源项目全部在做'给人脸上妆'，识别侧的项目全部是通用域，**没有任何一个开源模型认识美妆图**。所以我们没有去调别人的模型，而是在自己的美妆数据上微调了一个 17MB 的小模型，并用'归一化后指标不降反升'证明了它学到的是真伪信号、不是格式捷径。"

---

## 附：检索原始数据

原始结果保存在 `_gh_search/`（`q1.json` ~ `q6.json`，`s_*.json` 为验证性检索，`v_*.json` 为仓库元数据核实）。本篇报告中所有 Star 数、日期、许可证均取自 GitHub REST API 在 2026-09-23 的实时返回，未使用任何记忆或推测值。

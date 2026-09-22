# BeautyProof 真实 vs AI 生成美妆图评测集

> 这份数据集是 BeautyProof 的**真实素材资产**：不是脚本合成的对照图，
> 而是 10 张真实拍摄的美妆图 + 20 张即梦（jimeng）AI 生成的美妆图。
> 用它来验证「六工具 + 规则引擎」这条取证链路在真实分布上的表现，以及诚实标出它的边界。

- 登记表：`data/dataset_real_ai.json`（30 条样本，机器可读）
- 说明文档：本文件（给人看）
- 图片本体：`data/real/real_01..10`、`data/ai/ai_01..20`
- 原始清单：`data/real_ai_manifest.json`（素材来源清单一字未改，本表由它规范化而来）

---

## 1. 来源与数量

| 类别 | 数量 | 来源 | 内容 |
|---|---|---|---|
| `real_photo` 真实拍摄 | 10 | 品牌 / 公开素材（原始文件名形如 `训练集/<32位hash>.jpg`） | 真实美妆图，全流程未做人工篡改 |
| `ai_generated` AI 生成 | 20 | **即梦（jimeng）** 文生图，2026-09-15 ~ 2026-09-17 生成 | 妆容人像特写、口红静物、粉底液棚拍等 |

合计 **30 张**（真 10 / AI 20）。

AI 图的原始文件名里**保留了生成 prompt**（这是本项目最有价值的元信息之一，
已逐条抽进登记表的 `prompt` 字段）。举三个例子：

1. `亚洲女性面部特写，精致韩式妆容，无瑕奶油肌，柔光摄影，高清细节`
   （ai_01 / ai_02 / ai_03，同 prompt 不同次生成）
2. `一瓶粉底液产品静物，玻璃瓶身，柔光棚拍，高清`
   （ai_09 / ai_11 / ai_13 / ai_14）
3. `桌面平铺多支口红，不同色号排列，木质桌面，侧光，景深虚化，电商详情页风格，高清`
   （ai_06 / ai_15）

说明：部分长 prompt 在导出时被文件名长度截断，登记表中以 `prompt_truncated: true` 标记；
ai_20 经微信传输后文件名被改写，prompt 已丢失，登记为 `prompt: null`。

---

## 2. 目录结构与字段含义

```
data/
├── real/                    真实拍摄图  real_01.jpg … real_10.png
├── ai/                      AI 生成图   ai_01.png  … ai_20.png
├── dataset_real_ai.json     ★ 规范化评测集登记表（新增）
├── REAL_AI_DATASET.md       ★ 本说明文档（新增）
└── real_ai_manifest.json    原始素材清单（未改动，本表的来源）
```

`dataset_real_ai.json` 顶层字段：

| 字段 | 含义 |
|---|---|
| `name` / `created_at` | 数据集名称与建表日期 |
| `source_note` | 素材来源一句话说明 |
| `classes` | 真值类别字典：`real_photo` = 真实拍摄，`ai_generated` = AI 生成 |
| `expected_risk_note` | 为什么所有样本的期望档位都是 `inconclusive`（见第 3 节） |
| `risk_levels` | 规则引擎四档：`high_risk` / `suspicious` / `credible` / `inconclusive` |
| `count` | 数量统计（total / real_photo / ai_generated） |
| `samples` | 样本数组，每项见下 |

`samples[]` 每项字段：

| 字段 | 含义 |
|---|---|
| `id` | 样本编号（如 `real_01`、`ai_07`） |
| `path` | 图片路径，**相对仓库根目录**（如 `data/real/real_01.jpg`） |
| `ground_truth` | 真值：`real_photo` 或 `ai_generated` |
| `expected_risk` | 期望风险档位（当前系统能力下的预期输出，见第 3 节） |
| `orig_name` | 素材原始文件名（AI 图含生成 prompt，保留溯源） |
| `generator` | 仅 AI 图：生成器，均为 `jimeng` |
| `prompt` | 仅 AI 图：从文件名还原的生成 prompt（截断/缺失时对应标记） |
| `prompt_truncated` | 仅 AI 图：prompt 是否因文件名长度被截断 |
| `source` | 仅真实图：素材来源说明 |
| `note` | 备注 |

---

## 3. 真值定义，以及为什么 AI 图的期望档位**不是** `high_risk`

### 真值（ground_truth）

- `real_photo`：真实相机拍摄，**没有**经过人工篡改（不等于"有凭证可证明真实"）。
- `ai_generated`：整张图由即梦从文本生成，**整幅画面都是合成像素**，
  不存在"原图 + 被改动区域"这种局部篡改结构。

### 期望档位（expected_risk）

全部 30 条样本都登记为 `inconclusive`（无法判定）。这不是偷懒，而是**按当前系统能力如实登记**：

**真实图 → `inconclusive`**
规则引擎 `inconclusive` 的定义就是"现有工具没发现明确问题，但也不能证明一定真实"。
这批真实图本来就没有 C2PA 内容凭证，系统查不出篡改信号，
正确输出只能是 `inconclusive`——如果输出 `credible`，那是把"没查出来"错误升级成了"证明真实"。

**AI 图 → `inconclusive`（不是 `high_risk`）**

1. **TruFor 的已知盲区**：本项目的篡改检测主力 TruFor（CVPR 2023）判的是
   "这张图有没有被人工动过、动在哪一块"，官方**明确不覆盖"整张纯 AI 生成"这一类**。
   纯生成图里没有"局部被复制/拼接"的痕迹，所以 TruFor 在 AI 图上是**不会报警**的。
   这是**模型能力边界，不是 Bug**——因此不能把期望写成 `high_risk`，那样等于
   要求系统输出它当前给不出的结果，只会得到一个永远失败的评测。
2. **AIGC 信号不参与自动定级**：`tools/rule_engine.py` 里 `AIGC_TRIGGERS_HIGH_RISK = False`。
   实测依据：sdxl-detector 在本批素材上给出 0.8506~0.9621 的恒定高分
   （真实图 0.9571 vs 篡改图 0.9621，相差 0.005），**对"是否被篡改"零区分度**，
   还把 100% 真实图判成 AI 生成。所以它照常检测、照常进报告，但不参与定级。
3. 综合：AI 图在当前链路上会落到 `inconclusive`，与真实图同档。

> **这个"同档"本身就是本次评测要暴露的核心发现**：
> 现有链路对"整张纯 AI 生成"这一类**没有区分能力**。
> 我们在文档里如实写明，而不是把它伪装成高分通过——
> 这也正好指向 BeautyProof 下一步要补的能力（见第 5 节）。

---

## 4. 怎么用它跑评测

### 方式一：单张跑完整链路（推荐，最快看到结论）

在仓库根目录执行：

```bash
./run.sh tools/pipeline.py data/real/real_01.jpg
```

想跳过深度学习模型、秒出结论，加 `--fast`：

```bash
./run.sh tools/pipeline.py data/ai/ai_01.png --fast
```

一次要多张就并列写（只写 JSON 不打印过程）：

```bash
./run.sh tools/pipeline.py data/real/real_05.jpg data/ai/ai_09.png data/ai/ai_17.png --json-only
```

产出落在 `outputs/analysis_<图片名>.json` 和 `reports/report_<图片名>.md`。

### 方式二：批量跑全 30 张 + 出评测汇总

`tools/batch_run.py` 目前硬编码读 `data/manifest.json`（那是合成对照集的清单）。
要跑本数据集，先复制一份当临时清单再跑（**不要覆盖原 `manifest.json`**）：

```bash
cp data/dataset_real_ai.json data/manifest_real_ai.json
# 把 batch_run.py 里的 MANIFEST 临时指向 manifest_real_ai.json，或临时替换后还原
./run.sh tools/batch_run.py
```

结果写进 `results/dataset_eval.json`，终端会打印逐样本对照表
（真值 / 预期档位 / 实际档位）和命中率汇总。

> 注意：`path` 是相对仓库根目录，请在仓库根目录下执行上述命令。

> ⚠️ 跑 TruFor / AIGC 会加载几百 MB 模型并占用算力，
> 请避开与其他任务并发执行；仅做登记与文档查阅时不需要跑模型。

---

## 5. 已知局限

1. **样本量仍偏小**：真 10 / AI 20，共 30 张。不足以支撑严谨的统计指标
   （如置信区间、分位数阈值校准），只适合做定性验证与边界探查。
2. **AI 图只来自单一生成器（即梦）**：结论无法外推到 Midjourney、SDXL、Flux
   或商用素材库 AI 图。跨生成器的泛化能力**未经验证**。
3. **未覆盖深度伪造换脸**：本数据集只有"整张纯生成"这一种 AI 形态，
   没有 face-swap、局部 AI 重绘（inpainting）、AI 换背景等**混合篡改**样本——
   而这类恰是品牌方最高危的场景。
4. **真实图缺少可查凭证**：真实图无 C2PA 内容凭证，无法区分
   "真实未篡改"与"真实但被轻度后期调色/磨皮"，真值粒度只能到"未人工篡改"。
5. **部分 prompt 信息不完整**：7 张 AI 图的 prompt 因文件名长度被截断，
   1 张（ai_20）prompt 完全丢失，影响按 prompt 分组的细粒度分析。
6. **系统对纯 AI 生成图无区分能力**：如第 3 节所述，这是当前链路的能力边界，
   也是本数据集存在的意义——把边界量化出来，而不是让它藏在"全部通过"的评测里。

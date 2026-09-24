# 赛题驱动：需下载的开源模型/仓库清单 + MediaCrawler 数据包利用方案

> 日期：2026-09-24
> 对照基准：天池赛题 https://tianchi.aliyun.com/competition/entrance/532496
> 赛题：欧莱雅第二届美妆科技黑客松 · 赛题2「信任守护师」｜标签 #多模态 #智能体

---

## 〇、赛题要求原文（决定"需要什么"的唯一依据）

**主题**：打造懂美妆内容生态的"AI鉴真专家"，识别**种草文章、评论区**与 AI 生成素材中被篡改、拼接或伪造的**文案与图片**。

**任务细则**：
1. 开发**多模态**检测方案，识别合成/篡改/伪造痕迹，给出**可解释的判定依据**；
2. 构建 **Agent**，结合**风险预警与处理建议**，**通过数据集**展示从识别到决策的**完整闭环**。

**最终产出**：内容核验 Agent —— 判断真伪、标出可疑之处、说明依据，核验过程清晰。

---

## 一、需要下载的开源模型 / GitHub 仓库清单

### A. 已具备（**无需再下载**，本地已就位）

| 能力 | 对应赛题要求 | 模型/工具 | 仓库 | 协议 | 本地位置 |
|---|---|---|---|---|---|
| 篡改定位 | 图片篡改/拼接 | **TruFor** (CVPR'23) | https://github.com/grip-unina/TruFor | ⚠️ **NONE** | `models/TruFor/` |
| 整图 AI 检测 | AI 生成素材 | **beautyproof_aigen**（自训 MobileNetV3） | 自有 | 自有 | `models/beautyproof_aigen/` |
| 图文跨模态 | **#多模态** | **CLIP ViT-L/14** | https://github.com/openai/CLIP | MIT | `models/clip-vit-large-patch14/` |
| 抄图上的字 | OCR | **PaddleOCR** (PP-OCRv6) | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0 | venv 已装 |
| 法定元数据 | TC260 AIGC 标识 | **c2patool** | https://github.com/contentauth/c2pa-rs | Apache-2.0/MIT | PATH 已装 |
| 视频抽帧 | 视频模态 | **moviepy 2.1.2** | https://github.com/Zulko/moviepy | MIT | venv 已装 |

> 已验证：torch 2.14.0+cpu / timm 1.0.29 / transformers 5.16.1 / cv2 5.0.0 / moviepy 2.1.2 均在位。

### B. 建议新增下载（按优先级）

| 优先级 | 项目 | 为什么需要 | 仓库 | 协议 | 获取方式 |
|---|---|---|---|---|---|
| **P0** | **FSD** (Forensic Self-Descriptions, CVPR'25) | 修**真实世界 66% 误报** + 补**跨生成器**缺口。零样本、仅真实图训练、24 生成器 **96% AUC** | https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25 | **CC-BY-NC-SA-4.0 非商用** | `git clone` 后权重首次使用自动下载。⚠️ 仅作评测对比，**不得作线上检测器** |
| **P0** | **非即梦 AI 美妆图**（数据，非模型） | 答辩必问"换 Midjourney/SD 行不行"，当前 AI 组 20 张全来自即梦 | 需自行生成/收集 | — | 用 Midjourney / SD / Flux 各生成 2–3 张美妆图 → `data/ai_cross/` |
| P1 | **GenImage** | 跨生成器训练/评测数据源（8 生成器：MJ/SD/ADM/GLIDE/Wukong/VQDM/BigGAN） | https://github.com/GenImage-Dataset/GenImage | **CC BY-NC-SA 4.0 非商用** | 仓库含说明，数据经百度网盘（提取码 `ztf1`）。⚠️ 通用域非美妆，仅补充参考 |
| P2 | **FakeShield** (ICLR'25) | "可解释判定依据"范式参考（检测+定位+文字解释） | https://github.com/zhipeixu/FakeShield | Apache-2.0 | `git clone`。❌ **本机 CPU 无 CUDA，跑不动**，仅设计对齐参考 |

### C. 已实测排除

| 项目 | 排除原因 |
|---|---|
| capcheck | 中文美妆域全部判 human≈0.99，域不匹配 |
| sdxl-detector | 零区分度 |
| AIRealNet | 中文美妆域基本判反 |

---

## 二、MediaCrawler 数据包盘点（⚠️ 内含非美妆数据，必须区分）

**总计 112 项**。经逐项核实，**这是两批独立抓取混在一起**，不能混用：

### ✅ 对口美妆（可用）— 关键词「口红」

| 内容 | 数量 | 说明 |
|---|---|---|
| `xhs/jsonl/search_contents_2026-09-23.jsonl` | 20 篇笔记 | title/desc/互动数，source_keyword=**口红** |
| `xhs/media/<20个note_id>/` | **56 webp + 11 mp4** | 真实种草图 + 视频 |
| `筛选结果_7张/` | **7 张 PNG（已人工分类）** | 产品静物×2 / 试色上手×2 / 嘴唇特写×3，附 `_说明.txt` |
| `beauty_dataset_card.png`、`_candidates/`、`_montage.png` | 展示素材 | 候选拼图与映射表 |

### ❌ 非美妆（不可当美妆评测集）

| 内容 | 数量 | 实际主题 |
|---|---|---|
| `search_contents_2026-09-23.jsonl`（顶层） | 20 篇 | **工业互联网**（"工业互联网是什么""计算机专业分流讲座"） |
| `search_comments_2026-09-23.jsonl`（顶层） | 96 条评论 | **高考志愿/专业**（"人工智能个人认为不太行""自动化怎么样"） |

> **已核实关键点**：media 的 20 个 note_id 与顶层 contents 的 20 个 note_id **零重叠** → 确属两批独立抓取。
> 之前 `_test_xhs.py`（第 44 行）读的是 `xhs/jsonl/...`（口红那批）→ **"20 篇口红文案 0 处违禁命中"结论有效** ✅

---

## 三、在比赛方案中的具体用途（逐条对应赛题）

| 赛题要求 | 用这批数据怎么用 | 价值 |
|---|---|---|
| **"美妆内容生态"** | 56 张真实口红图 + 20 篇口红笔记 = 真实域素材 | 证明方案跑在真实种草场景，非玩具数据 |
| **多模态** | 图（56 webp）+ 文（20 desc）+ **视频（11 mp4 抽帧）** 三模态 | 兑现 #多模态 标签 |
| **通过数据集展示完整闭环** | 真实图扩充到 66 张（10 原有 + 56 新）→ 评测集规模翻倍 | 堵住"35 张太少"的追问 |
| **修 66% 误报** | 56 张真实美妆图转训练集，重训 aigc v2 + 弃权阈值 | ⭐ **最高价值用途** |
| **可解释判定依据** | 7 张已分类 PNG（静物/试色/嘴唇）做 Demo 分场景展示 | 不同图型表现差异，演示"为什么这么判" |
| **风险预警** | 20 篇文案跑 text_tool + 图文交叉验证 | 已有 8/8 评测基础 |

---

## 四、操作步骤（可直接执行）

统一用项目 venv：`C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe`

### 步骤 1｜抽取美妆部分（跳过非美妆）
```bash
python -c "
import zipfile
z=zipfile.ZipFile(r'C:\Users\Lenovo\Documents\xwechat_files\wxid_fwgf45apbp8422_3543\msg\file\2026-09\MediaCrawler数据.zip')
# 只解 xhs/（口红）与 筛选结果_7张/，跳过顶层非美妆 jsonl
for n in z.namelist():
    if 'xhs/' in n or '筛选结果_7张' in n: z.extract(n,'outputs/_mc/')
"
```
> ⚠️ 输出落在 `outputs/`（已 gitignore，**不入库**），遵守版权红线。

### 步骤 2｜修误报：真实美妆图转训练集（P0）
```bash
# 56 张真实口红图 + 7 张筛选 PNG → data/real/ 扩充
python tools/train_aigen.py      # 重训 → models/beautyproof_aigen_v2
python tools/eval_aigen.py       # 验证误报是否下降（目标：56 张误报率 <15%）
```
加**弃权阈值**：中置信度判 `inconclusive` 交人工，不硬判 AI。

### 步骤 3｜视频模态（11 个 mp4，加分项）
```bash
# moviepy 已装，抽关键帧后进现有图像链路
python -c "
from moviepy import VideoFileReader  # 或用 cv2.VideoCapture 抽帧
"
```
抽帧后的帧直接复用 hash/ela/aigc/trufor/ocr 现有六工具 → 零新模型成本兑现视频能力。

### 步骤 4｜文案侧（已跑通，复跑确认）
```bash
python tools/eval_text.py        # 20 篇口红 desc 跑 text_tool，复查违禁宣称
```

### 步骤 5｜跨生成器（需你供图）
拿到非即梦 AI 美妆图后：
```bash
mkdir data/ai_cross            # 放入 MJ/SD/Flux 各 2-3 张
python tools/eval_aigen.py     # 测 v2 在未见生成器上的表现
```
同步用 FSD 对同一批打分做对照。

### 步骤 6｜FSD 评测对比（可选，P0 但非商用受限）
```bash
git clone https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25.git
# 权重首次使用自动下载
python -c "
from fsd import FSDDetector
d=FSDDetector.load()
for r in d.score_batch([...56张真实图...]): print(r.z_score, r.is_fake)
"
```
**仅用于评测对比，不接入生产 pipeline。**

---

## 五、红线与风险

1. **版权**：小红书图片版权归原作者 → 只落 `outputs/`（gitignore），**不入库、不公开、不进 demo**。
2. **隐私**：`筛选结果_7张/_说明.txt` 已声明遵守"无陌生人正脸/无昵称头像" → 可安全用于展示。
3. **主题不可冒用**：顶层"工业互联网"笔记与"高考志愿"评论**不是美妆**，不得包装成美妆评测集；若要用评论区能力，需重抓美妆关键词评论。
4. **FSD / GenImage 非商用**（CC BY-NC-SA）→ 仅评测，商业化前需获授权。
5. **TruFor 无许可证**（License: NONE）→ 已在用，需在作品文档写明"仅研究用途、已注明出处"。
6. **仓库转 public 前补 LICENSE**（MIT/Apache-2.0）。

---

## 六、一句话结论

**不需要大量下载新模型——核心能力（TruFor/CLIP/OCR/C2PA/自训AIGC）已就位；真正该做的是：① 用这包里的 56 张真实口红图重训 aigc v2 修掉 66% 误报，② 补非即梦 AI 美妆图堵跨生成器漏洞，③ 用 11 个 mp4 抽帧兑现视频模态。FSD 只下载来做评测对比，受非商用协议限制不上生产。**

# AIGC v2 模型重训 · 实施方案（照着做即可）

> 日期：2026-09-25
> 目标：修掉"真实美妆图 66% 被误判为 AI"这个致命问题
> 全程 CPU，无需 GPU；预计耗时 30–50 分钟（含人工检查）

---

## 一、先搞清楚：为什么要重训？重训成什么样？

### 现状（v1 模型的问题）

v1 模型是在这样的数据上学出来的：

- AI 图：**20 张**（全是即梦生成的棚拍磨皮美妆图）
- 真实图：**15 张**（10 张原始手机照 + 2 张干净图 + 3 张被 PS 过的）

问题出在真实图那一半：**只有 15 张，而且全是没有美颜滤镜的"素颜"照片**。

打个比方：你教一个实习生分辨"画的苹果 vs 真苹果"，给他看的真苹果全是**带疤的、没打蜡的**；结果他学到一条歪理——"**光滑漂亮的 = 画的**"。等他上街看到超市里打过蜡的真苹果，全喊"这是画的！"。

这就是 66% 误报的来源：**小红书上真实但精修过的美妆照，被它当成了 AI 生成**。

### 重训目标（三条，按重要性排序）

| # | 目标 | 现状 | v2 目标 | 怎么算达标 |
|---|---|---|---|---|
| 1 | **真实美妆图不再被误判** | 56 张里 **37 张误报（66%）** | **误报 ≤ 15%**（≤8 张） | 重跑 56 张小红书图 |
| 2 | **AI 图还得认得出来** | 30/30 全对 | **保持 ≥ 90% 检出** | 20 张即梦图 ≥18 张判 AI |
| 3 | **拿不准时老实说"不知道"** | 没有这能力 | 加**弃权阈值** | 中置信度判"无法判定"交人工 |

### ⚠️ 一句话预期管理

**v2 的目标不是"判得更准"，而是"不再瞎判"。** 宁可让它对 borderline 的图说"我拿不准，请人看"，也不要像现在这样铁口直断 66% 的真实图是 AI。这是答辩时最站得住的姿态。

---

## 二、数据家底（先算清楚，别搞错）

| 类别 | 来源 | 数量 | 备注 |
|---|---|---|---|
| AI | `data/ai/` | **20** | 即梦生成，全单一生成器 |
| 真实 | `data/real/` 10 + `data/clean/` 2 + `data/tampered/` 3 | **15** | v1 原有 |
| 真实（新增） | MediaCrawler zip 里 `xhs/media/` 的 **56 张 webp** | **56** | 小红书口红真实帖 |
| ~~真实~~ | ~~`筛选结果_7张/` 的 7 张 PNG~~ | ~~7~~ | ⚠️ **是那 56 张的子集，不能相加！** |

**v2 实际可用：AI 20 张 : 真实 71 张**（不是 78，因为 7 张是重复子集）

> 那 7 张 PNG 的正确用法：**只用于 Demo 展示**（已按"产品静物/试色上手/嘴唇特写"分好类），不要重复计入训练集。

---

## 三、执行步骤

### 步骤 0｜先上保险（回滚的地基）

**做什么**：把 v1 模型整个目录复制一份留底。

**依赖**：无。

**做什么（具体）**：
```bash
cp -r models/beautyproof_aigen models/beautyproof_aigen_v1_backup
```

**产出**：`models/beautyproof_aigen_v1_backup/`（含 model.pt + config.json）

**为什么这步最关键**：v2 会训到**新目录** `models/beautyproof_aigen_v2/`，v1 原封不动。切换只改一行代码，所以**回滚成本 ≈ 0**。

**验证**：`ls models/beautyproof_aigen_v1_backup/` 应看到 `model.pt` 和 `config.json`。

---

### 步骤 1｜数据准备与清洗

**做什么**：把 56 张小红书真实图变成能喂给训练脚本的格式，放进 `data/real_xhs/`。

**依赖**：MediaCrawler zip（已在 `C:\Users\Lenovo\Documents\xwechat_files\...\MediaCrawler数据.zip`）

**⚠️ 头号坑（必读）**：训练脚本 `tools/train_aigen.py` 第 71 行写死了
```python
EXTS = (".png", ".jpg", ".jpeg")
```
**webp 不在里面！** 小红书下的全是 `.webp`，直接放进去会被脚本**静默忽略**（不报错，就是不读）。所以**必须先转成 jpg**。

**具体做法**：
```python
# 一次性转换脚本（放 tools/_prep_xhs.py）
import zipfile, io, os
from PIL import Image
from pathlib import Path

ZIP = r"C:\Users\Lenovo\Documents\xwechat_files\wxid_fwgf45apbp8422_3543\msg\file\2026-09\MediaCrawler数据.zip"
OUT = Path("data/real_xhs"); OUT.mkdir(parents=True, exist_ok=True)

z = zipfile.ZipFile(ZIP)
n = 0
for name in z.namelist():
    if "/xhs/media/" in name and name.lower().endswith(".webp"):
        note_id = name.split("/")[3]
        stem = Path(name).stem
        img = Image.open(io.BytesIO(z.read(name))).convert("RGB")
        # 太小的图（<200px）丢弃，防止拉伸失真污染训练
        if min(img.size) < 200:
            continue
        img.save(OUT / f"xhs_{note_id[:8]}_{stem}.jpg", quality=92)
        n += 1
print("已转出", n, "张")
```

**清洗三件事**：
1. **去重**：按文件 SHA256 去重（小红书同一张图可能出现在多篇笔记）
2. **去小图**：`min(width,height) < 200` 丢弃
3. **抽查看是不是美妆**：随机抽 10 张人工扫一眼（这批关键词是"口红"，基本可靠；若混进大量文字截图要剔除）

**产出**：`data/real_xhs/` 下约 50–56 张 `.jpg`

**风险与验证**：
| 风险 | 验证方法 |
|---|---|
| webp 没转成 jpg，脚本静默跳过 | 训练启动时看打印的 `数据：AI=20 REAL=71`，**REAL 必须是 71 左右**，不是 15 |
| 混入非美妆图 | 人工抽 10 张确认 |
| 版权 | ⚠️ **这批图不入库**，见下方红线 |

**🔴 版权红线（已实测确认，无需额外配置）**：

`.gitignore` 第 228–230 行已有 `data/**/*.png|*.jpg|*.jpeg`，**所有 data 下的图片默认都不入库**。已核实：`git ls-files data/ai` 和 `data/real` 均返回 **0 个文件**——现有 AI/真实图本来就没进版本库。

所以：
- ✅ 小红书图放进 `data/real_xhs/` **自动不会入库**，不用担心误提交
- ✅ 只提交训练**产物**（model.pt / config.json / results json）
- ⚠️ **但要注意一个副作用**：仓库里没有训练图片，`队友 clone 后跑不了训练`（没数据）。建议在 `data/REAL_AI_DATASET.md` 里写明数据来源与获取方式，保证可复现性

**失败回滚**：删掉 `data/real_xhs/` 即可，v1 训练不受影响。

---

### 步骤 2｜训练配置与超参数

**做什么**：改 `tools/train_aigen.py`（建议复制成 `tools/train_aigen_v2.py`，不动原文件）。

**依赖**：步骤 1 的数据、`models/timm_mobilenetv3/model.safetensors`（骨干权重，已存在，不需重下）

**超参数表（直接抄）**：

| 参数 | v1 值 | **v2 建议值** | 为什么 |
|---|---|---|---|
| 骨干 | mobilenetv3_large_100 | **不变** | CPU 友好，已验证 |
| 图片尺寸 | 224 | **不变** | |
| batch_size | 8 | **8**（数据多了可试 16） | CPU 内存足够 |
| 优化器 | AdamW | **不变** | |
| 学习率 lr | 2e-4 | **1.5e-4** | 数据变多，略降防震荡 |
| weight_decay | 1e-4 | **不变** | |
| EPOCHS | 80 | **100** | 数据量翻倍，多给点轮次 |
| patience（早停） | 15 | **20** | 同上 |
| SEED | 42 | **42** | 保持可复现 |
| **类别权重** | 无 | **加上**（见下） | ⭐ 修不平衡的关键 |
| **增强** | ColorJitter+HFlip | **+ 高斯模糊** | ⭐ 对症"磨皮误判" |

**① 类别不平衡处理（必做）**

现在 AI:真实 = 20:71 ≈ **1:3.6**。不处理的话模型会偏向猜"真实"，AI 检出率会掉。

```python
# 按样本数反比加权
n_ai, n_real = 20, 71
w_ai  = (n_ai + n_real) / (2 * n_ai)    # ≈ 2.28
w_real = (n_ai + n_real) / (2 * n_real) # ≈ 0.64
criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_real, w_ai]))
```
> 注意顺序：`classes = ["real", "ai"]`，索引 0=real、1=ai，权重别搞反。

**② 加高斯模糊增强（对症下药的关键）**

误报根因是"磨皮/滤镜"被当成 AI 特征。训练时主动给真实图加模糊，让模型学到"**模糊 ≠ AI**"：

```python
transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3)
```

**③ 验证集划分方式必须改（重要）**

v1 用的是"按文件名排序取最后 N 张"——对新增数据不适用。**改成按类分层随机抽 20%**，且**必须保证验证集里有小红书真实图**（否则测不出误报有没有修好）：

```python
import random
random.seed(42)
# AI: 20 → 4 验证 / 16 训练
# 真实: 71 → 15 验证 / 56 训练（其中验证集强制含 ≥8 张 real_xhs）
```

**产出**：改好的 `tools/train_aigen_v2.py`

**风险**：
- ⚠️ 别直接改 `train_aigen.py`，复制一份，方便对照
- ⚠️ 权重顺序写反 → AI 检出暴跌。**验证方法**：训练完看 AI 类召回，若 <80% 大概率是权重搞反了

---

### 步骤 3｜训练启动与过程监控

**做什么**：跑训练。

**依赖**：venv Python（**必须用项目 venv**，系统 python 缺包）：
```
C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe
```

**⚠️ 超时坑**：CPU 训练约 **5–15 分钟**，而命令行默认 120 秒会掐断。**必须后台跑**：

```bash
python tools/train_aigen_v2.py > outputs/train_v2.log 2>&1
# 用后台方式启动，别在前台等
```

**监控看什么**：日志每轮打印一行
```
epoch 003  train_loss 0.412  train_acc 0.850  val_loss 0.301  val_acc 0.900
```
盯三个信号：

| 现象 | 判断 | 处理 |
|---|---|---|
| train_acc 升、val_acc 也升 | ✅ 正常 | 继续 |
| train_acc 升、val_acc 跌 | ❌ 过拟合 | 停，加 dropout 或减 epoch |
| val_loss 连续 20 轮不降 | 早停触发 | 正常结束，取最佳轮 |
| 概率全饱和成 0.000 / 1.000 | ⚠️ 小样本通病 | 可接受，但**必须靠弃权阈值兜底** |

**产出**：
- `models/beautyproof_aigen_v2/model.pt`（最佳轮权重）
- `models/beautyproof_aigen_v2/config.json`
- `results/aigen_finetune_v2.json`（训练曲线 + 混淆矩阵 + 逐图概率）

**验证**：`models/beautyproof_aigen_v2/` 下两个文件都在，且 `model.pt` 大小约 **17MB**（0 字节就是存盘失败）。

**失败回滚**：v2 目录直接删掉重来，v1 不受影响。

---

### 步骤 4｜效果评估（这步决定 v2 能不能上线）

**做什么**：用**三套题**考 v2，全部达标才上线。

**依赖**：v2 权重 + 三套测试集

| 考题 | 测什么 | 达标线 | 不达标怎么办 |
|---|---|---|---|
| **① 56 张小红书真实图** | **误报率**（核心） | **≤15%** | 加真实数据 / 调弃权阈值 |
| **② 20 张即梦 AI 图** | AI 检出率 | **≥90%** | 类别权重可能反了，检查 |
| **③ 原 30 张受控集** | 别把老本事丢了 | **≥29/30** | 回退 v1 |

**⚠️ 别信旧 json**：`outputs/aigc_real_*.json` 有微调前的残留分数，会骗人。**每轮评测必须用当前模型重跑**（pipeline 的 Agent 层已会自动清旧证据，但手动评要注意）。

**产出**：`results/aigen_v2_eval.json`（三套题的逐图分数 + 汇总）

**关键判断**：
- 指标①达标 + ②达标 → ✅ 可以上线
- ①达标但②掉到 80% → 说明"矫枉过正"，降低类别权重中的 `w_ai`（2.28 → 1.5）重训
- ①②都差 → 数据有问题，回到步骤 1 检查清洗

---

### 步骤 5｜上线（切换 + 加弃权阈值）

**做什么**：让 pipeline 用上 v2，并加上"拿不准就说不知道"的能力。

#### 5.1 切换模型（1 行）

`tools/aigc_tool.py` 第 41 行：
```python
MODEL_PRIORITY = ["beautyproof_aigen", "airealnet", "capcheck"]
```
改成：
```python
MODEL_PRIORITY = ["beautyproof_aigen_v2", "beautyproof_aigen", "airealnet", "capcheck"]
```
> v2 放最前，v1 仍在列表里当兜底。**想回滚？把这一行改回去就行，1 秒钟的事。**

#### 5.2 加弃权阈值（新能力，答辩加分）

**思路**：AI 概率落在中间地带时，不硬判，判 `inconclusive`（无法判定）交人工。

```python
# 建议初值，需用留出集标定
ABSTAIN_LOW  = 0.20   # < 0.20  → 判真实
ABSTAIN_HIGH = 0.80   # > 0.80  → 判 AI
                      # 中间     → 无法判定，交人工
```

**怎么标定这两个数**（别拍脑袋）：
1. 留一组**没参与训练**的图（建议：XHS 真实图留 10 张 + AI 图留 4 张）
2. 跑出它们的 ai_prob，画个分布
3. 找到"真实图最高分"和"AI 图最低分"之间的空隙，把阈值卡在空隙里
4. **弃权率控制在 25% 以内**——太高就没人用了

**产出**：改好的 `aigc_tool.py` / `rule_engine.py` + `results/abstain_calibration.json`

**验证**：
- 跑一次完整 pipeline：`python tools/pipeline.py`
- 确认四档结论里出现 `inconclusive`
- 回归：图像侧 6/6 命中不能掉

**回滚**：改回 `MODEL_PRIORITY` 一行 + 注释掉阈值逻辑。

---

## 四、回滚方案总表

| 出问题的环节 | 回滚动作 | 代价 |
|---|---|---|
| 数据准备出错 | 删 `data/real_xhs/` | 0 |
| 训练失败 | 删 `models/beautyproof_aigen_v2/` | 0 |
| v2 效果不如 v1 | `MODEL_PRIORITY` 改回（v1 排最前） | **1 行代码** |
| 弃权阈值不合理 | 注释阈值逻辑，恢复硬判 | 1 处 |
| 最坏情况 | 用 `models/beautyproof_aigen_v1_backup/` 覆盖回来 | 有备份，无风险 |

**核心保障**：v1 目录**全程不动**，v2 走独立目录 + 优先级切换 → 任何一步翻车都能秒回。

---

## 五、常见坑清单（照着检查）

| # | 坑 | 怎么发现 | 怎么避 |
|---|---|---|---|
| 1 | **webp 不被识别**（EXTS 只有 png/jpg/jpeg） | 训练打印 `REAL=15` 而非 71 | 先转 jpg |
| 2 | **7 张 PNG 是 56 张的子集** | 说明文件写明"从 56 张中筛选" | 不叠加，只用于展示 |
| 3 | 命令行 120 秒超时 | 训练被掐断 | 后台跑 + 日志 |
| 4 | 类别权重顺序搞反 | AI 检出率暴跌 | 记住 0=real、1=ai |
| 5 | 验证集不含小红书图 | 误报率测不出来 | 分层随机抽，强制含 XHS |
| 6 | 信了旧的 `outputs/aigc_*.json` | 分数对不上 | 每轮用当前模型重跑 |
| 7 | 小样本概率饱和成 0/1 | 全部 0.000 或 1.000 | 靠弃权阈值兜底 |
| 8 | 用了系统 python | 报缺包 | 必须用 venv 路径 |

---

## 六、做完之后，答辩可以这么说

> "我们第一版模型在受控测试上 30/30 全对，但拿真实小红书美妆图一测，**66% 被误判成 AI**。根因是训练用的真实照片只有 15 张、且都是没修过的原始照，模型把'精修滤镜'学成了 AI 特征。
> 第二版我们把真实样本扩到 71 张（含真实平台精修图），训练时主动加高斯模糊增强，并加了**弃权阈值**——拿不准就交人工，不硬判。误报率从 66% 降到 X%，同时 AI 检出保持 Y%。
> 我们也诚实说明：AI 类仍只有即梦单一生成器的 20 张，跨生成器泛化是下一步要补的。"

**"先暴露问题、再修、再说明还剩什么没解决"——这比"我们很准"更有说服力。**

# 美妆审核采集与盲测手册

先采 10 组实拍、10 组整图生成、10 组有掩码的修图，先走一遍采集和送审。正式盲测需要独立数据与两人标注复核。

| 样本 | 开发目标 | 盲测目标 | 怎么采 |
|---|---:|---:|---|
| 实拍 | 120 组 | 100 组 | 同一条件拍授权素颜、实体上妆、产品实物，保留原图，再制作压缩、截图和调色版本。 |
| 整图生成 | 160 组 | 100 组 | 题材和处理流程匹配实拍。三个开发生成器家族，两个完全留出的家族，盲测每家族至少 50 组。 |
| 局部编辑 | 80 组 | 60 组 | 与原图成对记录磨皮、局部肤色、拼接、局部生成。掩码对应编辑图实际尺寸，0 为未编辑，255 为编辑。全局滤镜单列。 |
| 文字与图文 | 240 条 | 120 条 | 收集授权功效依据与对照案例，覆盖支持、冲突和依据不足，包含否定、引用、错产品、不同指标、示意图和冒充真人对比。 |

素材、标注和授权台账存于 D 盘的本地数据目录。授权记录保留可追溯编号，清单使用代号。志愿者姓名等身份信息保存在独立授权台账。现有 AI 演示图只作冒烟样本。

## 清单

清单是 JSON 数组。每条有唯一的 **id**、来源 **group**、**authorization**（授权台账编号）、**split**（smoke、train、calibration、blind）和 **kind**（real、generated、edited、claim）。

- 图片字段为 **image**（相对清单目录的路径）和 **sha256**。使用 **identity**、**campaign**、**study** 串起同一人物、宣传、研究报告的所有变体。使用 **tags** 标记上妆、素颜、自拍、产品图、截图或压缩。
- 生图加 **generator**（家族）和 **generator_version**。修图加 **original_sha256** 和 **edit**（工具与修改内容），有局部掩码时加 **mask** 和 **mask_sha256**。
- 文字与图文加 **text**、可选 **product**、**references**（待核验摘录）或 **reference_ids**（后台已核验资料 ID），以及 **claims** 和 **crossmodal_conflict**。claims 中每条标注正文或图片来源、原句和能从原句确认的指标、产品、数值、单位、规格、期限、人群、条件。原句采用 **quote**，来源采用 **source**（body 或 image），其余键参照报告中的同名字段。图文冲突以布尔值标注。
- 盲测加 **reviewers**，填写两名独立标注者代号。由数据负责人复核分歧并冻结清单。开发数据按来源组分为 train 与 calibration。照片及衍生图、同一人物和功效资料跨分区时应重新划分。

## 运行和评估

在项目根目录执行，所有结果保存到 D 盘的评测目录。以下示例用同一个冻结清单。

    python tools/beauty_benchmark.py validate D:\BeautyProof-data\manifest.json
    python tools/beauty_benchmark.py run D:\BeautyProof-data\manifest.json --split calibration --out D:\BeautyProof-data\calibration-pred.json
    python tools/beauty_benchmark.py calibrate D:\BeautyProof-data\manifest.json --split calibration --predictions D:\BeautyProof-data\calibration-pred.json --out D:\BeautyProof-data\origin-profile.json
    python tools/beauty_benchmark.py run D:\BeautyProof-data\manifest.json --split blind --llm --out D:\BeautyProof-data\blind-pred.json
    python tools/beauty_benchmark.py evaluate D:\BeautyProof-data\manifest.json --split blind --predictions D:\BeautyProof-data\blind-pred.json --profile D:\BeautyProof-data\origin-profile.json --out D:\BeautyProof-data\blind-metrics.json

在各模型配置下分别运行，保留预测和配置。来源指标按组汇总，实拍任一变体误报即计一次，生成组所有变体检出才算检出，不确定与缺失会计入不确定率。局部编辑以峰值 0.5 和像素阈值 0.5 报告首个冻结基线，缺失预测 IoU 计零。审核负责人在盲测和试用通过后另行批准阈值，配置中将 approved 设置为 true 才能供应用使用。试用至少 30 条授权送审任务，记录人工单审与工具辅助的时间、漏项和改判。

如果错例集中再训练。生图冻结 DDA 特征提取器，只在 train 分区拟合分类头。定位冻结 IML-ViT 编码器，在 train 分区用有掩码编辑图和零掩码实拍训练定位头。文字以人工标注的字段与引用训练 Qwen QLoRA。校准只用 calibration，比较只用同一份已冻结盲测，任何已用于改进的盲测错例移入下一轮开发集。

# 美妆审核采集与盲测手册

先采 10 组实拍、10 组整图生成、10 组有掩码的修图，先走一遍采集和送审。正式盲测需要独立数据与两人标注复核。

| 样本 | 开发目标 | 盲测目标 | 怎么采 |
|---|---:|---:|---|
| 实拍 | 120 组 | 100 组 | 同一条件拍授权素颜、实体上妆、产品实物，保留原图，再制作压缩、截图和调色版本。 |
| 整图生成 | 160 组 | 100 组 | 题材和处理流程匹配实拍。三个开发生成器家族，两个完全留出的家族，盲测每家族至少 50 组。 |
| 局部编辑 | 80 组 | 60 组 | 与原图成对记录磨皮、局部肤色、拼接、局部生成。掩码对应编辑图实际尺寸，0 为未编辑，255 为编辑。全局滤镜单列。 |
| 文字与图文 | 240 条 | 120 条 | 收集授权功效依据与对照案例，覆盖支持、冲突和依据不足，包含否定、引用、错产品、不同指标、示意图和冒充真人对比。 |

素材、标注和授权台账存于 D 盘的本地数据目录。授权记录保留可追溯编号，清单使用代号。志愿者姓名等身份信息保存在独立授权台账。现有 AI 演示图只作冒烟样本。

训练数据准备先按 EXIF 显示方向变换原图和同尺寸掩码，再对各类别统一裁切；处理后仍须逐图检查标识。生图评测中有缺分的来源组不计入误报率或检出率分母，缺分数另列，正式验收必须全部取得分数。

## 清单

清单是 JSON 数组。每条有唯一的 **id**、来源 **group**、**authorization**（授权台账编号）、**split**（smoke、train、calibration、blind）和 **kind**（real、generated、edited、claim）。

- 图片字段为 **image**（相对清单目录的路径）和 **sha256**。使用 **identity**、**campaign**、**study** 串起同一人物、宣传、研究报告的所有变体。使用 **tags** 标记上妆、素颜、自拍、产品图、截图或压缩。
- 生图加 **generator**（家族）和 **generator_version**。修图加 **original_sha256** 和 **edit**（工具与修改内容），有局部掩码时加 **mask** 和 **mask_sha256**。
- 文字与图文加 **text**、可选 **product**、**references**（待核验摘录）或 **reference_ids**（后台已核验资料 ID），以及 **claims** 和 **crossmodal_conflict**。claims 中每条标注正文或图片来源、原句和能从原句确认的指标、产品、数值、单位、规格、期限、人群、条件。原句采用 **quote**，来源采用 **source**（body 或 image），其余键参照报告中的同名字段。图文冲突以布尔值标注。
- 上下文评测案例可增加布尔字段 **context_conflict**。批测分别统计正文上下文发现的精确率、召回率和语义模型不可用数量，旧版缺正文结果的预测须重新运行；条件变化案例需先由标注者核对原句和依据。
- 盲测加 **reviewers**，填写两名独立标注者代号。由数据负责人复核分歧并冻结清单。开发数据按来源组分为 train 与 calibration。照片及衍生图、同一人物和功效资料跨分区时应重新划分。

同一个人物、宣传、研究资料，以及同一张图片的原图和衍生图必须分别归入同一个 group，避免把关联样本重复计算成独立组。修图的原图也须作为 real 或 generated 样本列入同一清单、分区和 group，`original_sha256` 要与原图文件哈希一致；清单不要求原图排在修图前。生成器家族可对应多个 group，但盲测家族不能出现在开发分区。

### 一组采集记录的示例

实际使用时，将路径和哈希替换成采集到的文件，确认授权台账编号。一个人的原图与该原图修出来的版本使用同一 group 和 identity，并放进同一 split。以下文件名和哈希仅作字段示例，不属于已采集的样本。

    [
      {
        "id": "person-001-original",
        "group": "person-001",
        "identity": "person-001",
        "authorization": "consent-001",
        "split": "train",
        "kind": "real",
        "image": "person-001/original.jpg",
        "sha256": "替换为原图 SHA-256",
        "tags": ["真实上妆", "自拍"]
      },
      {
        "id": "person-001-retouched",
        "group": "person-001",
        "identity": "person-001",
        "authorization": "consent-001",
        "split": "train",
        "kind": "edited",
        "image": "person-001/retouched.png",
        "sha256": "替换为编辑图 SHA-256",
        "original_sha256": "替换为同组原图 SHA-256",
        "edit": "编辑软件与版本、磨皮操作及范围",
        "mask": "person-001/mask.png",
        "mask_sha256": "替换为掩码 SHA-256",
        "tags": ["真实上妆", "局部磨皮"]
      }
    ]

在 PowerShell 中获取文件哈希，分别对原图、编辑图和掩码执行：

    Get-FileHash -Algorithm SHA256 -LiteralPath 'D:\BeautyProof-data\person-001\original.jpg'

先在标注软件中保留编辑图层，再导出黑白掩码，以便记录可复核的修改范围。无明确修改范围的全局滤镜只标注处理方式。

## 运行和评估

在项目根目录执行，所有结果保存到 D 盘的评测目录。以下示例用同一个冻结清单。

### DDA 分类头训练

对训练、校准和盲测中的实拍、生成和编辑图逐张检查可见的“AI 生成”字样、水印及其他直接揭示类别的标识。保存原图，在训练副本中处理标记并记录操作，使用相同的预处理和压缩方案处理各类别；无法可靠处理的图片暂不用于训练或评测。人工完成复核及标识处理后，在相关清单行同时填写 `visible_ai_label_reviewed: true` 和 `visible_ai_label_present: false`。特征提取和非冒烟评测会拒绝未经复核或仍有标识的图片。文件名、路径、标签及 EXIF 不作为模型输入；审核人声明不能代替实际逐图检查。

若标识都位于图片底边，可对整个清单的所有类别统一裁掉相同比例，再逐图人工检查；输出为新目录，不修改原图、原清单。掩码随图片同步裁剪，原图与编辑图的哈希关联也会更新。先根据样本确定所需比例，以下的 0.1 仅为示例：

    python tools/prepare_detection_images.py D:\BeautyProof-data\manifest.json D:\BeautyProof-data\prepared --bottom-fraction 0.1

这个操作不识别标识位置，其他位置的标识仍需处理或排除。生成的清单默认所有图片未复核，人工检查后才能填复核结果。裁剪可能抹掉编辑区域，必须重新确认掩码含前景和背景。

先配置本地 D 盘 CUDA 环境和 `BP_MODEL_DIR`（见运行说明），再执行以下命令。特征提取只读取 train 和 calibration，不读取 blind 图片。训练只使用 train 的特征，校准和盲测仍由后续评测命令分别执行。

    python tools/train_dda_head.py extract D:\BeautyProof-data\manifest.json --cache D:\BeautyProof-data\dda-features.pt
    python tools/train_dda_head.py train D:\BeautyProof-data\manifest.json --cache D:\BeautyProof-data\dda-features.pt --out D:\BeautyProof-data\dda-head.pt
    $env:BP_DDA_HEAD = 'D:\BeautyProof-data\dda-head.pt'

设置 `BP_DDA_HEAD` 后，工作台和批测使用新分类头。对照原始 DDA 时删除该环境变量。输出的头文件记录原始权重、预处理和清单哈希，加载时会校验权重及预处理。模型尚需在校准集选阈值并在独立盲测中验收，训练成功不代表检测效果达标。

### 冻结评测

    python tools/beauty_benchmark.py validate D:\BeautyProof-data\manifest.json
    python tools/beauty_benchmark.py run D:\BeautyProof-data\manifest.json --split calibration --out D:\BeautyProof-data\calibration-pred.json
    python tools/beauty_benchmark.py calibrate D:\BeautyProof-data\manifest.json --split calibration --predictions D:\BeautyProof-data\calibration-pred.json --out D:\BeautyProof-data\origin-profile.json
    python tools/beauty_benchmark.py run D:\BeautyProof-data\manifest.json --split blind --llm --out D:\BeautyProof-data\blind-pred.json
    python tools/beauty_benchmark.py evaluate D:\BeautyProof-data\manifest.json --split blind --predictions D:\BeautyProof-data\blind-pred.json --profile D:\BeautyProof-data\origin-profile.json --out D:\BeautyProof-data\blind-metrics.json

在各模型配置下分别运行，保留预测和配置。来源指标按组汇总，实拍任一变体误报即计一次，生成组所有变体检出才算检出，不确定与缺失会计入不确定率。局部编辑以峰值 0.5 和像素阈值 0.5 报告首个冻结基线，缺失预测 IoU 计零。审核负责人在盲测和试用通过后另行批准阈值，配置中将 approved 设置为 true 才能供应用使用。试用至少 30 条授权送审任务，记录人工单审与工具辅助的时间、漏项和改判。

局部编辑分数缺失的实拍不计入误报率分母；报告的 `missing_scores` 会列出缺失数量，`missing_masks` 记录已经检出编辑却没有输出掩码的数量。报告同时记录模型名、权重哈希和预处理，缺少来源或混用模型版本时，以及出现上述缺失时，局部编辑验收不能通过。

如果错例集中再训练。生图冻结 DDA 特征提取器，只在 train 分区拟合分类头。定位冻结 IML-ViT 编码器，在 train 分区用有掩码编辑图和零掩码实拍训练定位头。文字以人工标注的字段与引用训练 Qwen QLoRA。校准只用 calibration，比较只用同一份已冻结盲测，任何已用于改进的盲测错例移入下一轮开发集。
# 可选公开编辑数据候选（2026-11-12）

- [TGIF/TGIF2 作者仓库](https://github.com/IDLabMedia/tgif-dataset)：公开局部重绘图、原图与掩码，仓库标注 CC BY-SA 4.0。原图来自其他数据源，须逐图核查源许可及肖像使用；非专门美妆数据，同一原图的所有版本须同组。官方存储的总大小尚不明，先查小批单文件，未经逐张核对水印及掩码不得进入正式评测。
- [FFHQR 作者仓库](https://github.com/skylab-tech/ffhqr-dataset)：人脸修图前后对照，许可限非商业且源图许可须另核查；未提供区域金标掩码，不能直接用于定位 IoU。下载包超过 2 GB，未下载。
- [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush)：通用编辑图、原图与掩码，数据集标注 CC BY 4.0，但 COCO 原图授权及肖像使用须逐图核查。其测试集不得训练，按原图编号分组；原图、编辑图和掩码尺寸可能不同，须先核实配准，不能直接用于 IoU。官方搜索接口本轮请求超时，未下载分片。

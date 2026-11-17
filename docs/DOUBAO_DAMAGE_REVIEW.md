# 退款破损审核

在 BeautyProof 顶栏选择「退款破损审核」，上传 JPG 或 PNG，点击「开始检查」。可以框选裂痕、碎粉或包装破口。结果包含整图和选区的检测分、复核阈值、像素线索与来源标识。橙色覆盖图显示候选边缘，可能包含印刷线、接缝与反光。

本地默认从 `models/doubao_damage/model.json` 读取 v5 CPU 权重。整图用色度和再压缩响应，框选同时使用局部噪声与裂纹叠片，在统一的 288 像素短边提取特征。v5 的权重是 17,201 字节。v4 权重保存在 `outputs/runs/doubao-damage-accuracy/reproduced/model.json`。

## v5 降低误报

训练时给初次拟合后分数较高的实拍更高的权重，保留实拍总权重、豆包总权重和配对训练项。框选使用较弱的正则化；两种模式都在内层商品留出折校准后略微提高复核阈值。推理仍使用两种模式原有的特征与同一套 CPU 算子。

| 上传条件 | 整图误报 v4 → v5 | 整图豆包检出 v4 → v5 | 框选误报 v4 → v5 | 框选豆包检出 v4 → v5 |
| --- | ---: | ---: | ---: | ---: |
| 原文件 | 17 → 9 | 41 → 36 | 16 → 13 | 37 → 37 |
| JPEG Q60 | 20 → 13 | 48 → 45 | 13 → 11 | 43 → 43 |
| JPEG Q45 缩略图 | 8 → 6 | 24 → 21 | 14 → 10 | 31 → 32 |

每项对照用同一批 55 张豆包破损图和 55 个完好实拍条目，涉及 31 个商品组和 46 张不同的实拍。按原图去重，整图原文件误报从 12 张降到 6 张。两批素材轮流留出时，六个框选对照的误报都减少，六个整图对照中四个减少、两个持平。部分批次的豆包检出也有下降。现有资料尚无单独标注的真实破损实拍验证集。

下文 v4 复跑命令加 `--low-false-positives`，把 `--output` 改为 `outputs/runs/doubao-damage-accuracy/low-fp/release`，即可生成 v5 权重和逐图结果。

## 准确率与误报率验收

准确率按全部送审图计算。误报率以实拍为分母，豆包检出率以豆包破损图为分母。对每种上传形态分别用固定阈值检查准确率 `> 95%`、误报率 `< 5%`。没有给出判定的图进入准确率分母，验收要求每张图都有判定。55 张豆包图加 55 个完好实拍对照时，误报至多 2 个，误报和漏报合计至多 5 个。

```powershell
python tools/evaluate_doubao_target.py outputs/runs/doubao-damage-accuracy/low-fp/release/predictions.csv --method v5 --output outputs/runs/doubao-damage-target/v5-target.json
```

本批数据里所有实拍都来自同款手机的原文件，豆包图经过不同导出流程。来源取证实验同时检查原文件、Q60、Q45，以及两类图统一裁到 288×354 并统一 JPEG Q55 的条件。验收还需要独立采集、单独标注的真实破损实拍和豆包破损图。图像导出、商品、采集会话应按组留出。

### 教师模型与蒸馏实验

本地跑了 Community Forensics、它的低画质微调版、SAFE 和 NPR。原始分类头迁移到破损局部编辑的效果有限。Community Forensics 冻结骨干、每折重训分类头并约束同图多种上传形态的分数差异后，原文件、Q60、Q45、统一编码的豆包检出分别为 39、40、29、36 张，完好实拍误报分别为 1、0、0、2 个。每种形态各有 55 个正例和 55 个反例，阈值从其他商品组校准。 [Community Forensics](https://github.com/JeongsooP/Community-Forensics) · [SAFE](https://github.com/Ouxiang-Li/SAFE) · [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)

让教师只用 Q60 形态训练和校准时，商品留出结果为检出 52/55、误报 2/55、准确率 95.5%。按两个采集批次分别留出时，Q60 准确率分别为 96.6% 和 65.4%，误报率分别为 6.9% 和 26.9%。统一编码的商品留出准确率为 73.6%，说明教师的表现仍受批次与导出条件影响。

训练了 16,297 参数的卷积学生，学习教师分数和中间特征。统一编码留出集上，它的 AUROC 从纯监督时的 0.797 升到 0.816，固定阈值下检出 9 张，误报 1 个，准确率为 57.3%。用 Q60 训练的第二个学生，蒸馏后豆包检出从 16/55 增到 23/55，误报从 5/55 降到 2/55，准确率为 69.1%。还测试了使用少量教师卷积投影的学生和使用像素统计的学生。实验脚本与逐图结果在 `outputs/runs/doubao-damage-target/`。

## v4 准确率对照

v4 优先提高平衡准确率，也就是对豆包破损检出率与完好实拍正确放过率取平均。55 对图按 31 个商品组划分，完整保留过去的五个留出折。阈值只在各训练折的内层开发组校准。原文件、外部 JPEG 质量 60 和缩略图 JPEG 质量 45 均经过与线上一致的推理预处理。

| 同一批留出图 | v3 整图 | v4 整图 | v3 框选 | v4 框选 |
| --- | ---: | ---: | ---: | ---: |
| 原文件 | 53.6% | 71.8% | 56.4% | 69.1% |
| 长边最多 1024、JPEG Q60 后上传 | 50.9% | 75.5% | 57.3% | 77.3% |
| 短边最多 256、JPEG Q45 后上传 | 65.5% | 64.5% | 60.9% | 65.5% |

原文件整图 55 张豆包破损中的检出从 55 降到 41，55 个完好对照中的误报从 51 降到 17。Q60 上传时，v4 整图为 48 个检出和 20 个误报；重用 v3 特征并重新校准的对照为 40 个检出和 24 个误报。Q45 缩略图是整图模式的主要难点。开发时已查看这些商品组的结果，新的采集来源需要独立盲测确认。

两批素材轮流留出时，v4 在原文件和 Q60 的四个整图对照上均提高了平衡准确率。框选原文件在其中一批从 57.7% 降至 55.8%。本机 CPU 对 56 张图实测，整图 P95 为 0.365 秒，框选 P95 为 0.394 秒。

v4 最终权重、商品分组和跨批次评测在 `outputs/runs/doubao-damage-accuracy/reproduced/`。额外样本检查与 CPU 测速在 `outputs/runs/doubao-damage-accuracy/release/`。

```powershell
python tools/train_doubao_damage_accuracy.py --annotations outputs/runs/doubao-damage-v3/annotations.csv --real-dir $realDir --ai-dir $aiDir --v3-report outputs/runs/doubao-damage-v3/dual-scale/evaluation.json --v3-predictions outputs/runs/doubao-damage-v3/dual-scale/predictions.csv --output outputs/runs/doubao-damage-accuracy/reproduced
```

## v3 扩展训练与评测

对 93 张处理图重新查看，70 张可见破损。其中 55 张有高或中置信的实拍参照（32 高、23 中），形成 31 个商品组。15 张能看到破损但缺少可靠原图，单独检查整图模型。与旧图同源及重复的文件归入同组或去重。

主实验保留 v2 的商品留出分组，把新商品各自放进一个留出折。阈值与候选模型均在训练折内选择。两版在相同的 55 张处理图与配对实拍上比较。

| 条件 | v2 整图检出 | v3 整图检出 | v2 框选检出 | v3 框选检出 |
| --- | ---: | ---: | ---: | ---: |
| 统一编码 | 53/55 | 55/55 | 49/55 | 55/55 |
| JPEG 质量 60 | 44/55 | 51/55 | 46/55 | 49/55 |

- 统一编码条件下，v3 整图把 51/55 个完好配对对照标为疑似，框选为 48/55。55 个对照条目对应 46 张不同实拍。v3 的用途是高召回复核筛查，真实破损实拍的表现仍未知。
- 整图按纹理强弱分别比较 RGB 残差和色度，15 张缺少可靠原图的破损图中，v3 在原文件上整图检出 15 张，JPEG 质量 60 重新上传后检出 14 张。v2 在原文件上检出 7 张。这些图来自同一批资料，数据源关系未全部核清。框选继续使用局部重建残差、色度和框内外差异。试验过 2,949 参数的残差 CNN，在这些商品划分上接近随机。
- 有原图时另做了对齐实验。55 张处理中，有 34 张能可靠对齐，框选定位变化检出其中 32 张。模拟正常 JPEG、裁剪、旋转调色与锐化的 217 个可对齐对照中有 11 个触发。这个实验度量相对原图的变化，来源归因还需结合证据复核。
- 这批图已用于模型开发。群组留出与跨批次结果用于对照候选。验证更高召回需要新的商品、编辑批次和转发链路。

记录保存在 `outputs/runs/doubao-damage-v3/`。v3 权重与逐图分数见 `dual-scale/`，复核标注见 `annotations.csv`。其余包含微型 CNN、无可靠原图的 15 张诊断和原图对齐实验。v2 原始评测保留在 `outputs/runs/doubao-damage-v2/`。

## 复跑

将 `$realDir` 和 `$aiDir` 设为配对后的两个目录。使用含 NumPy、OpenCV 与 Pillow 的 Python。

```powershell
python tools/train_doubao_damage_expanded.py --annotations outputs/runs/doubao-damage-v3/annotations.csv --real-dir $realDir --ai-dir $aiDir --output outputs/runs/doubao-damage-v3/dual-scale
python tools/doubao_damage.py "样图.jpg"
python tools/doubao_damage.py "样图.jpg" --roi "[0.25,0.4,0.7,0.85]"
python tools/doubao_damage.py $aiDir
```

训练命令写模型副本到指定输出目录，可用 `--model` 指向该文件单独评分。工作台默认权重路径维持不变。接口 `POST /api/damage-review` 的 multipart 表单接收 `file` 和可选的归一化 EXIF 坐标 `roi`，响应包含 `whole`、`region` 和候选边缘叠图。


## 裂纹算子叠片和数学建模

可用裂纹诊断命令生成六格图，分别显示输入、Hessian 中心线、黑帽与白帽增强、Scharr 对向梯度、成对 Canny 边界和融合中心线。绿线是候选线，橙点是候选分叉。融合用每个位置的各层强度给中心线加权，再用连通弱线连接断续段。选区里的木纹、印刷和包装边也可能被提取。

数学模型分两部分。图上量端点、分叉、闭环、线段迂曲程度和三叉角偏差。像素上沿法线采样，分别拟合背景斜坡、阶跃边缘和窄线剖面 i(n) = a + b·n + c·exp(−n²/(2σ²))，看窄线模型能解释多少误差。宽度和角度都是图像量。光照、透视、材料与外力条件会改变真实裂纹的形态和投影。 [曲线提取](https://ieeexplore.ieee.org/document/659930/) · [干裂分叉实验](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.74.021405)

在相同的 55 对、31 个留出商品组上比较，其中 30 对按标注含可见裂线。JPEG 质量 60 时，只用几何量的整图和框选 AUROC 分别为 0.460 和 0.537。加入原有分类器后有小幅收益，该轮在 v3 基础上测试，叠片和曲线拟合通过独立诊断命令提供。

| JPEG 质量 60 | AI 破损检出 | 完好配对触发 | AUROC |
| --- | ---: | ---: | ---: |
| v3 整图 | 51/55 | 34/55 | 0.754 |
| 整图复查尺度加入叠片 | 53/55 | 36/55 | 0.753 |
| v3 框选 | 49/55 | 40/55 | 0.751 |
| 框选先裁材料，再叠片和拟合剖面 | 50/55 | 38/55 | 0.769 |

对发丝裂纹，先裁材料表面再缩放保留了更多线条。当前完好实拍对照没有真实破损，归因能力需用同材料的真实裂纹与豆包裂纹对照。完整候选及每图评分见 outputs/runs/doubao-damage-geometry/crop-first/。

```powershell
python tools/fracture_geometry.py "样图.jpg" --roi "[0.57,0.34,0.75,0.516]" --crop-first --quality 60 --output "裂纹叠片.png"
python tools/evaluate_damage_geometry.py --annotations outputs/runs/doubao-damage-v3/annotations.csv --real-dir $realDir --ai-dir $aiDir --v3-predictions outputs/runs/doubao-damage-v3/dual-scale/predictions.csv --output outputs/runs/doubao-damage-geometry/crop-first --crop-first
```

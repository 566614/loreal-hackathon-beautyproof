# 退款破损审核

在 BeautyProof 顶栏选择「退款破损审核」，上传 JPG 或 PNG，点击「开始检查」。可选「框选破损」，拖动包住裂痕、碎粉及其邻近纹理。页面会显示整图和选区状态、检测分、主要像素线索及来源标识。橙色覆盖图是选区候选边缘，可能包含印刷、接缝与反光。

v2 优先检出疑似 AI 生成破损图，超过阈值即进入人工复核。该入口独立运行 CPU 模型，推理使用 NumPy、OpenCV 与 Pillow，权重默认从 `models/doubao_damage/model.json` 加载。

## v2 评测

主评测沿用 28 对高置信样本与 17 个独立商品组。15 对中置信样本作补充；额外检查了 19 对高置信同源但未确认破损的处理图。候选模型可以用与训练折商品同组的补充图训练，整个商品组及其 JPEG 变体一起留出。阈值在内层开发折上选择，目标是三种视图都达到至少 90% 召回。表中数字来自外层留出。

| 模式 | AUROC | 豆包破损图检出 | 完好实拍触发复核 |
| --- | ---: | ---: | ---: |
| v2 整图 | 0.847 | 28/28 | 23/28 |
| v2 框选 | 0.726 | 27/28 | 20/28 |
| v1 整图，仅按同一规则降低阈值 | 0.672 | 28/28 | 28/28 |
| v1 框选，仅按同一规则降低阈值 | 0.647 | 28/28 | 26/28 |

- JPEG 质量 60 的对照中，v2 整图检出 25/28，框选检出 23/28。只用另一批数据训练时，较大批次的整图 AUROC 降至 0.525，接近随机排序。这批资料已用于研发，结果属于内部实验。
- 整图选中高频 RGB 通道关系、色度与纹理关系。框选选中这些线索加残差形状、相邻残差排列及框内外对照。候选覆盖成对训练、浅层提升树、FFT、细裂纹形态，以及有无辅助样本的训练。
- 对同一图片的不同压缩视图做平均，整图从 28/28 降至 27/28，完好实拍触发从 23/28 降至 18/28。默认沿用召回更高的单视图。
- 裂纹形态只作为展示与实验对照，最终分类器按留出表现选了残差和色度。当前实拍对照都是完好品，真实破损照片的结果未知。跨两批和压缩条件未找到稳定的非 JPEG 周期峰。来源元数据单独显示。

数据来自本地两份目录，标注在 `outputs/runs/doubao-damage/annotations.csv`。v2 的候选、跨批次结果、逐图评分、测速和模型副本在 `outputs/runs/doubao-damage-v2/`。原版评测在 `outputs/runs/doubao-damage/`。

## 复跑

将 `$realDir` 和 `$aiDir` 设为配对后的图片目录。使用含 NumPy、OpenCV 与 Pillow 的本地 Python。

```powershell
python tools/train_doubao_damage.py --annotations outputs/runs/doubao-damage/annotations.csv --real-dir $realDir --ai-dir $aiDir --output-dir outputs/runs/doubao-damage-v2
python tools/doubao_damage.py "样图.jpg"
python tools/doubao_damage.py "样图.jpg" --roi "[0.25,0.4,0.7,0.85]"
python tools/doubao_damage.py $aiDir
```

接口 `POST /api/damage-review` 接收 multipart 表单，`file` 为原图，`roi` 可传 JSON 数组 `[左, 上, 右, 下]`，坐标为相对 EXIF 纠正后图像的 0 到 1。结果中的 `whole` 与 `region` 分别包含状态、评分、原因与线索。框选时还会返回候选边缘 PNG 和边缘统计。

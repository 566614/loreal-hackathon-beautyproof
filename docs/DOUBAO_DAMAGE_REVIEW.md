# 退款破损审核

在 BeautyProof 顶栏选择「退款破损审核」，上传 JPG 或 PNG，点击「开始检查」。可以框选裂痕、碎粉或包装破口。结果包含整图和选区的检测分、复核阈值、像素线索与来源标识。橙色覆盖图显示候选边缘，可能包含印刷线、接缝与反光。

v3 优先检出疑似 AI 生成破损图。首次检查用 256 像素短边，未命中时额外检查 288 像素短边。入口独立在 CPU 运行，使用 NumPy、OpenCV 和 Pillow。模型 30,140 字节。对 56 张图片测速，整图 P95 为 233 ms，框选 P95 为 275 ms。权重默认从 `models/doubao_damage/model.json` 加载。

## 扩展训练与评测

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

记录保存在 `outputs/runs/doubao-damage-v3/`。当前权重与逐图分数见 `dual-scale/`，复核标注见 `annotations.csv`。其余包含微型 CNN、无可靠原图的 15 张诊断和原图对齐实验。v2 原始评测保留在 `outputs/runs/doubao-damage-v2/`。

## 复跑

将 `$realDir` 和 `$aiDir` 设为配对后的两个目录。使用含 NumPy、OpenCV 与 Pillow 的 Python。

```powershell
python tools/train_doubao_damage_expanded.py --annotations outputs/runs/doubao-damage-v3/annotations.csv --real-dir $realDir --ai-dir $aiDir --output outputs/runs/doubao-damage-v3/dual-scale
python tools/doubao_damage.py "样图.jpg"
python tools/doubao_damage.py "样图.jpg" --roi "[0.25,0.4,0.7,0.85]"
python tools/doubao_damage.py $aiDir
```

训练命令写模型副本到指定输出目录，可用 `--model` 指向该文件单独评分。工作台默认权重路径维持不变。接口 `POST /api/damage-review` 的 multipart 表单接收 `file` 和可选的归一化 EXIF 坐标 `roi`，响应包含 `whole`、`region` 和候选边缘叠图。

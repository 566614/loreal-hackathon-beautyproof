# 退款破损审核

在 BeautyProof 顶栏选择「退款破损审核」，上传一张 JPG 或 PNG，再点击「开始检查」。需要检查局部时，先点击「框选破损」，拖动包住裂痕或碎粉及其邻近纹理。结果会显示整图和选区评分、贡献较大的像素特征、文件来源标识和候选边缘。

这个入口独立运行 CPU 模型，推理使用 NumPy、OpenCV 与 Pillow。权重默认从项目下的 `models/doubao_damage/model.json` 加载。文件内的来源标识是元数据线索，页面按是否声明 AI 或待核验显示。橙色覆盖图是候选边缘，正常印刷、包装接缝与反光也会被增强。

## 本次数据和结果

从 127 张完好实拍与 93 张处理图的配对清单中，逐张检查高、中置信配对的破损变化。28 对高置信破损样本分在 17 个商品组，用于训练和分组评测。15 对中置信破损样本用于补充对照。其余 29 张高、中置信处理图未纳入破损训练，21 张低置信图保留待复核。两张双色粉盘处理图在训练标注中重新指向了相同粉盘的实拍图。

| 高置信商品留出 | AUROC | 豆包破损图召回 | 完好实拍误报 |
| --- | ---: | ---: | ---: |
| 整图 | 0.672 | 8/28（28.6%） | 6/28（21.4%） |
| 框选 | 0.647 | 9/28（32.1%） | 6/28（21.4%） |

评测阈值在各训练折内部按完好实拍误报不高于 10% 设定。门槛要求留出召回至少 60%、完好实拍误报不高于 10%。本次两个模式均未通过，工作台只给实验评分和「证据不足」。

- 文件字节数单独区分两类的 AUROC 为 1.000，说明采集链路差异极大。模型统一缩放并用同一 JPEG 参数重编码，文件大小和元数据没有输入模型。统一编码也无法消除文件此前经历的处理。
- 整图模型最终使用方向高通残差，框选模型使用拉普拉斯残差和框内外差异。评测同时比较 RGB、Gaussian、中值、其他残差组合、FFT、边缘及灰度纹理对照。
- 边缘与灰度纹理在部分验证折也有区分度。当前对照样本都是完好实拍，指标只覆盖这批样本，无法估计真实破损照片的误报。
- 对两批图分别检查高频能量、自相关、二维峰和 JPEG 网格，未找到达到本次预设复核标准并跨压缩条件重复的非 JPEG 频谱峰。现有图的像素统计不足以确认隐藏水印或豆包专有指纹。

完整本地报告在 `outputs/runs/doubao-damage/evaluation.json`，包含各算子的开发对照、批次留出和压缩退化结果。逐图样本与分数在同目录的 `annotations.csv` 和 `predictions.csv`，CPU 测速在 `benchmark.json`。这些文件与权重存放在本地忽略目录。

## 复跑

将 `$realDir` 和 `$aiDir` 设为配对后图片的两个目录。使用含 NumPy、OpenCV 与 Pillow 的本地 Python 即可重新训练。

```powershell
python tools/train_doubao_damage.py --annotations outputs/runs/doubao-damage/annotations.csv --real-dir $realDir --ai-dir $aiDir --output-dir outputs/runs/doubao-damage
python tools/doubao_damage.py "样图.jpg"
python tools/doubao_damage.py "样图.jpg" --roi "[0.25,0.4,0.7,0.85]"
python tools/doubao_damage.py $aiDir
```

接口 `POST /api/damage-review` 接收 multipart 表单，`file` 为原图，`roi` 可传 JSON 数组 `[左, 上, 右, 下]`，坐标为相对 EXIF 纠正后图像的 0 到 1。结果中的 `whole` 与 `region` 分别包含状态、评分、原因与线索；框选时还会返回候选边缘 PNG 和边缘统计。

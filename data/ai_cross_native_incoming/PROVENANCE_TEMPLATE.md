# 跨生成器素材 — 来源登记表（PROVENANCE LEDGER / 使用说明）

本目录（`data/ai_cross_native_incoming/`）用于补齐「非即梦族」的 AI 美妆/人像图
（Midjourney / Stable Diffusion / Flux），目标：把跨生成器零样本召回从 ~58% 提到 ≥80%。

## 红线（务必遵守，collect_crossgen.py 会强制校验）
1. 只用「可证明为 AI 生成、来源清晰、无真实人物肖像权风险」的图。
2. **禁止**把任何真实人物照片、网红/名人肖像当 AI 训练标签
   ——会污染「真实美妆图低误报」这一核心优势（曾因此吃过大亏：66% 误报）。
3. **禁止**无来源标注的图进入训练：每个生成器子目录必须有本文件，逐张登记。

## 目录结构（照此放）
```
data/ai_cross_native_incoming/
├── midjourney/
│   ├── PROVENANCE.md   ← 本文件，逐张填写下表
│   └── mj_01.png ...    ← 美妆/人像风格图
├── stable_diffusion/
│   ├── PROVENANCE.md
│   └── sd_01.png ...
└── flux/
    ├── PROVENANCE.md
    └── flux_01.png ...
```

## 字段模板（每个生成器子目录的 PROVENANCE.md 里，每张图一行）
| 文件名 | 生成器 | 提示词(可简写) | 来源 / 许可 | 是否合成(无真人) | 备注 |
|--------|--------|----------------|-------------|------------------|------|
| xxx.png | flux | "beauty portrait, soft makeup" | SFHQ-T2I (Kaggle, MIT) | 是 | 合成脸，无肖像权风险 |
| yyy.png | midjourney | "close-up beauty, natural light" | 本人订阅生成 | 是 | 本人账号产出 |

## 推荐合规来源（已核实许可）
- **SFHQ-T2I**（SelfishGene，Kaggle，MIT 许可，122K 合成脸，含 Flux1/SDXL/DALL·E 逐张
  model+prompt+seed 元数据）：免费、合成、无隐私/肖像权问题，最适合直接取 Flux/SDXL 子集。
- 本人 MJ/SD/Flux 订阅自行生成：来源最清晰，许可最稳。

## 收编命令
```bash
python tools/collect_crossgen.py \
  --staging data/ai_cross_native_incoming \
  --train-out data/ai_mj_sd_flux \
  --heldout-out data/ai_mj_sd_flux_heldout \
  --heldout-per-generator 4
```
收编后跑：
```bash
python tools/train_aigen_v6.py        # 重训 v6，跨生成器进训练，原 ai_cross_native 仍作零样本
```

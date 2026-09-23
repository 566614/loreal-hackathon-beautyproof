# -*- coding: utf-8 -*-
"""
AIGC 工具 —— 判断一张图是不是「整张由 AI 生成」的，输出统一格式的证据 JSON

用法：
    python tools/aigc_tool.py <图片路径>

大白话：
    用开源深度学习模型扫一遍图，给出「这张图是 AI 生成的」概率（0~1）。
    它补的是 TruFor 查不出的盲区：TruFor 擅长查「局部被人动过」，但查不出
    「整张图都是 AI 画的」。本工具专门管后者。

能证明什么：
    这张图「像不像整张由 AI 生成的」。概率高 → 大概率是 AI 生成的图。

不能证明什么（铁律）：
    模型会误判。分数只是「模型觉得」，不是「一定的事实」。
    真实照片也可能被误判成 AI，AI 生成的也可能蒙混过关；分数不是定罪依据。

模型：
    优先用「本域微调」模型 beautyproof_aigen（timm mobilenetv3_large_100，
    用标注图 10 真 + 20 AI（即梦）微调，见 tools/train_aigen.py）。

    为什么必须自己微调（这是实测结论，不是拍脑袋）：
        通用鉴伪模型在我们的中文美妆域彻底失效——
        · AIRealNet：把真实美妆图判成 artificial(0.96)、把 AI 图判成 real(0.0)，基本反向
        · capcheck ：所有图（连 AI 图在内）都输出 human≈0.99，零区分度
        原因：通用模型训练域是艺术图/新闻图，迁移不到「中文美妆 AI 图」细分域。
        所以正确解 = 用本域标注图微调专属模型。

    AIRealNet / capcheck 保留为兜底（见 MODEL_PRIORITY），但默认不再走它们。
    旧 sdxl-detector 零区分度，已彻底弃用。
    任一模型未就绪都优雅降级（返回 available=False），绝不联网下载。
"""
import json
import sys
from pathlib import Path

# 优先级：本域微调模型 > 通用模型兜底。
# 顺序按实测区分度排（见 results/aigen_finetune.json）—— 微调模型胜。
MODEL_PRIORITY = ["beautyproof_aigen", "airealnet", "capcheck"]

# 权重文件名（微调模型是 model.pt，通用模型是 safetensors / bin）
WEIGHT_FILES = ("model.pt", "model.safetensors", "pytorch_model.bin")

# 标签关键词：哪些是「AI 生成」类，哪些是「真实/人」类
AI_KW = ["artificial", "ai", "fake", "generated", "synthetic"]
REAL_KW = ["real", "human", "authentic", "natural"]


def model_path():
    """返回本地模型目录；都没下好就返回 None（调用方据此优雅降级）。"""
    root = Path(__file__).resolve().parent.parent
    for name in MODEL_PRIORITY:
        d = root / "models" / name
        if (d / "config.json").exists() and any((d / w).exists() for w in WEIGHT_FILES):
            return str(d)
    return None


def is_timm_model(model_dir):
    """本域微调模型是 timm/torch 权重，通用模型是 transformers 权重。

    靠 config.json 里的 "framework": "timm" 区分。
    """
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return cfg.get("framework") == "timm"


def ai_prob_from(results):
    """把模型输出映射成统一的「AI 生成概率」与对应标签。

    返回 (ai_score, ai_label)：
      - 若某标签含 artificial/ai/fake/generated/synthetic → 直接取该标签分数
      - 否则（标签是 real/human 等）→ ai_score = 1 - 该标签分数
    """
    for r in results:
        if any(k in r["label"].lower() for k in AI_KW):
            return float(r["score"]), r["label"]
    for r in results:
        if any(k in r["label"].lower() for k in REAL_KW):
            return round(1.0 - float(r["score"]), 4), r["label"]
    # 兜底：取非 top-1 那一类的补集
    top = max(results, key=lambda x: x["score"])
    return round(1.0 - float(top["score"]), 4), top["label"]


# 微调模型加载一次即可（CPU 上建图约 1~2 秒），缓存起来避免每张图重建
_FT_CACHE = {}


def _load_ft_model(model_dir):
    """加载本域微调的 timm 模型 + 推理用的图像预处理，按 model_dir 缓存。"""
    if model_dir in _FT_CACHE:
        return _FT_CACHE[model_dir]

    import torch
    import timm
    from torchvision import transforms

    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    arch = cfg["arch"]
    size = int(cfg.get("img_size", 224))
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])
    classes = cfg.get("classes", ["real", "ai"])
    ai_idx = classes.index("ai") if "ai" in classes else 1

    model = timm.create_model(arch, pretrained=False, num_classes=int(cfg.get("num_classes", 2)))
    state = torch.load(str(Path(model_dir) / "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    _FT_CACHE[model_dir] = (model, tfm, ai_idx)
    return _FT_CACHE[model_dir]


def run_aigc_timm(image_path, model_dir):
    """用本域微调模型推理（torch 直推，不走 transformers pipeline）

    返回: (ai_score, ai_label, available)
    """
    import torch
    from PIL import Image

    model, tfm, ai_idx = _load_ft_model(model_dir)
    img = Image.open(str(image_path)).convert("RGB")
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0][ai_idx].item()
    label = "ai_generated" if prob >= 0.5 else "real"
    return round(float(prob), 4), label, True


def run_aigc(image_path):
    """判断是不是整图 AI 生成 —— 优先走本域微调模型，否则 transformers 通用模型兜底

    返回: (ai_score, ai_label, available)
        available=False 表示模型没下下来 / 加载失败 —— 此时 ai_score 为 None，
        调用方应当按「无法判断」处理，而不是当成真的 0 分。
    """
    local_model = model_path()
    if local_model is None:
        # 模型没下好：直接降级，绝不联网下载（默认网络会被掐断，重试退避会卡死整条流水线）
        return None, "model_missing: 先跑 train_aigen.py 或 download_aigen_models.py", False

    # 本域微调模型：timm 权重，单独走 torch 推理分支
    if is_timm_model(local_model):
        try:
            return run_aigc_timm(image_path, local_model)
        except Exception as e:  # noqa: BLE001
            return None, f"inference_failed(timm): {type(e).__name__}: {str(e)[:200]}", False

    from transformers import pipeline

    try:
        detector = pipeline("image-classification", model=local_model)
    except Exception as e:  # noqa: BLE001
        return None, f"model_unavailable: {type(e).__name__}: {str(e)[:200]}", False

    try:
        results = detector(str(image_path))
    except Exception as e:  # noqa: BLE001
        return None, f"inference_failed: {type(e).__name__}: {str(e)[:200]}", False

    ai_score, ai_label = ai_prob_from(results)
    return round(ai_score, 4), ai_label, True


def build_evidence(image_path, ai_score, ai_label, available=True):
    if not available:
        return {
            "tool": "aigc",
            "source_asset_id": Path(image_path).name,
            "observed": "AI 生成检测暂不可用（模型未下载成功 / 加载失败）",
            "cannot_prove": "模型未就绪，无法给出 AI 生成概率；本项按「无法判断」处理，不影响其他工具结论",
            "evidence": [
                {"aigc_score": None, "label": ai_label, "available": False},
            ],
        }
    # 措辞保持中性：新模型在本素材上确实能区分真实图与 AI 图，但仍强调「分数不是事实」。
    if ai_score >= 0.9:
        verdict = "模型给出的 AI 生成概率很高（≥0.9），结合其他工具，很可能是整图由 AI 生成的图"
    elif ai_score >= 0.8:
        verdict = "有部分 AI 生成的迹象（≥0.8），建议人工复核"
    elif ai_score >= 0.5:
        verdict = "AI 生成概率中等，无法单独定性，需结合其他工具"
    else:
        verdict = "模型认为看起来像真实拍摄 / 人工制作的图"
    return {
        "tool": "aigc",
        "source_asset_id": Path(image_path).name,
        "observed": f"AI 生成概率 {ai_score}（{verdict}）",
        "cannot_prove": "模型也会误判：分数高不代表一定 AI 生成，分数低也不代表一定真实；"
                        "本项与 TruFor 互补——TruFor 查局部篡改，本项查整图 AI 生成",
        "evidence": [
            {"aigc_score": ai_score, "label": ai_label, "available": True},
        ],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/aigc_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    ai_score, ai_label, available = run_aigc(image_path)
    report = build_evidence(image_path, ai_score, ai_label, available)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"aigc_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Local adapters for the official DDA and IML-ViT checkpoints."""
import hashlib
import os
from pathlib import Path
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root():
    return Path(os.environ["BP_MODEL_DIR"]).resolve()


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def dda(image_path):
    import torch
    from torch import nn
    from PIL import Image
    from torchvision import transforms

    root = _root()
    weights = root / "DDA_ckpt.pth"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    sys.path.insert(0, str(root / "dda" / "Inference"))
    from models.lora import apply_lora_to_linear_layers

    # The official checkpoint has the backbone, LoRA and head. Build the
    # backbone from local source so inference does not download a second copy.
    model = nn.Module()
    model.base_model = nn.Module()
    backbone = torch.hub.load(str(root / "dinov2"), "dinov2_vitl14", source="local", pretrained=False)
    model.base_model.model = apply_lora_to_linear_layers(
        backbone, rank=8, alpha=1, target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        trainable_orig=False,
    )
    model.base_model.fc = nn.Linear(1024, 1)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    device = _device(torch)
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.CenterCrop(336), transforms.ToTensor(),
        transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]),
    ])
    with Image.open(image_path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = model.base_model.model.forward_features(x)["x_norm_clstoken"]
        score = model.base_model.fc(features).sigmoid().item()
    return {"aigc_score": round(score, 6), "available": True, "model": "dda-dinov2-vitl14",
            "weights_sha256": file_hash(weights), "preprocessing": "dda-official-center-crop336-clip",
            "device": device, "review_only": True}


def iml_vit(image_path, output_dir):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    root = _root()
    weights = root / "iml-vit_checkpoint.pth"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    sys.path.insert(0, str(root / "iml-vit"))
    from iml_vit_model import iml_vit_model

    model = iml_vit_model()
    model.load_state_dict(state.get("model", state), strict=True)
    del state
    device = _device(torch)
    model.to(device).eval()
    with Image.open(image_path) as input_image:
        image = input_image.convert("RGB")
    scaled = image.copy()
    scaled.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    w, h = scaled.size
    x = F.normalize(F.to_tensor(scaled), [.485, .456, .406], [.229, .224, .225])
    x = F.pad(x, [0, 0, 1024 - w, 1024 - h], fill=0).unsqueeze(0).to(device)
    empty = x.new_zeros((1, 1, 1024, 1024))
    with torch.inference_mode(), torch.autocast(device_type=device, enabled=device == "cuda", dtype=torch.bfloat16):
        _, output, _ = model(x, empty, empty)
    probability = output[0, 0, :h, :w].float().cpu().numpy()
    heatmap = Image.fromarray(np.rint(probability * 255).astype("uint8")).resize(image.size, Image.Resampling.BILINEAR)
    mask = heatmap.point(lambda v: 255 if v >= 128 else 0)
    alpha = heatmap.point(lambda v: int(v * .55))
    overlay = Image.composite(Image.new("RGB", image.size, "#ff442b"), image, alpha)
    heatmap.save(output_dir / "iml_heatmap.png")
    overlay.save(output_dir / "iml_overlay.jpg", quality=90)
    mask.save(output_dir / "iml_mask.png")
    return {"trufor_score": round(float(probability.max()), 6), "available": True, "model": "iml-vit",
            "weights_sha256": file_hash(weights), "preprocessing": "iml-v1-thumbnail1024-imagenet-pad-br",
            "device": device, "review_only": True,
            "tampered_area_ratio": float(np.count_nonzero(np.asarray(mask)) / (image.width * image.height)),
            "heatmap": "iml_heatmap.png", "overlay": "iml_overlay.jpg", "mask": str(output_dir / "iml_mask.png")}

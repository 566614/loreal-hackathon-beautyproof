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


def _available_system_memory():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                    ("total_phys", ctypes.c_ulonglong), ("available_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.available_phys


def _check_system_memory(required_gib, model):
    available = _available_system_memory()
    if available is not None and available < required_gib * 1024 ** 3:
        raise RuntimeError(f"{model} 需要至少 {required_gib:g} GiB 可用系统内存，请释放内存后重试")


def _check_iml_memory(torch, device):
    _check_system_memory(5.5 if device == "cuda" else 8, "IML-ViT")
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 5 * 1024 ** 3:
            raise RuntimeError("IML-ViT 需要至少 5 GiB 可用显存，请释放显存后重试")


def _check_dda_memory(torch, device):
    _check_system_memory(4, "DDA")
    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 1024 ** 3:
            raise RuntimeError("DDA 需要至少 3 GiB 可用显存，请释放显存后重试")


def load_dda():
    _check_system_memory(4, "DDA")
    import torch
    from torch import nn
    from torchvision import transforms

    root = _root()
    device = _device(torch)
    _check_dda_memory(torch, device)
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
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.CenterCrop(336), transforms.ToTensor(),
        transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]),
    ])
    return model, transform, device


def dda_feature(model, transform, device, image_path):
    import torch
    from PIL import Image

    with Image.open(image_path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model.base_model.model.forward_features(x)["x_norm_clstoken"]


def dda(image_path):
    model, transform, device = load_dda()
    import torch

    weights = _root() / "DDA_ckpt.pth"
    model_name = "dda-dinov2-vitl14"
    weights_hash = file_hash(weights)
    head_path = os.environ.get("BP_DDA_HEAD")
    if head_path:
        head = torch.load(head_path, map_location="cpu", weights_only=True)
        if (head["backbone_sha256"] != weights_hash or
                head["preprocessing"] != "dda-official-center-crop336-clip"):
            raise ValueError("分类头与 DDA 权重或预处理不匹配")
        model.base_model.fc.load_state_dict(head["head"], strict=True)
        model_name = "dda-dinov2-vitl14-finetuned-head"
        weights_hash = file_hash(Path(head_path))
    features = dda_feature(model, transform, device, image_path)
    with torch.inference_mode():
        score = model.base_model.fc(features).sigmoid().item()
    return {"aigc_score": round(score, 6), "available": True, "model": model_name,
            "weights_sha256": weights_hash, "preprocessing": "dda-official-center-crop336-clip",
            "device": device, "review_only": True}


def _iml_preprocess(image):
    """Resize only oversized images, then use the official padding order."""
    from PIL import Image
    from torchvision.transforms import functional as F

    scaled = image.copy()
    scaled.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    w, h = scaled.size
    # The official pipeline pads black pixels before ImageNet normalization.
    x = F.pad(F.to_tensor(scaled), [0, 0, 1024 - w, 1024 - h], fill=0)
    return F.normalize(x, [.485, .456, .406], [.229, .224, .225]).unsqueeze(0), (w, h)


def iml_vit(image_path, output_dir):
    _check_system_memory(5.5, "IML-ViT")
    import numpy as np
    import torch
    from PIL import Image

    root = _root()
    device = _device(torch)
    _check_iml_memory(torch, device)
    weights = root / "iml-vit_checkpoint.pth"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    sys.path.insert(0, str(root / "iml-vit"))
    from iml_vit_model import iml_vit_model

    model = iml_vit_model()
    model.load_state_dict(state.get("model", state), strict=True)
    del state
    model.to(device).eval()
    with Image.open(image_path) as input_image:
        image = input_image.convert("RGB")
    x, (w, h) = _iml_preprocess(image)
    x = x.to(device)
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
            "weights_sha256": file_hash(weights), "preprocessing": "iml-v2-thumbnail1024-pad-black-imagenet",
            "device": device, "review_only": True,
            "tampered_area_ratio": float(np.count_nonzero(np.asarray(mask)) / (image.width * image.height)),
            "heatmap": "iml_heatmap.png", "overlay": "iml_overlay.jpg", "mask": str(output_dir / "iml_mask.png")}

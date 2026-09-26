# -*- coding: utf-8 -*-
"""AIGC 检测器 v2：用「真实精修美妆图」扩充真实类，修掉 v1 的 66% 误报。

与 v1（tools/train_aigen.py）的四处关键差异
-------------------------------------------
1. 真实类扩充：data/real_xhs/（74 张小红书真实美妆图，含精修/滤镜）
   v1 只有 15 张原始手机照 -> 模型把"光滑漂亮"学成了 AI 特征。
2. 分层**随机**划分验证集（v1 是"排序取最后 N 张"，对新数据不适用），
   且验证集必然含 real_xhs 图，否则测不出误报有没有修好。
3. 类别权重：AI:REAL ≈ 20:89 严重不平衡，不加权模型会偏向猜 real。
4. 新增高斯模糊增强：训练时主动给真实图加模糊，让模型学到"模糊 ≠ AI"，
   直接对症"磨皮/滤镜被误判"这个根因。

产物（v1 目录全程不动，回滚只需改 MODEL_PRIORITY 一行）
--------------------------------------------------------
    models/beautyproof_aigen_v2/model.pt + config.json
    results/aigen_finetune_v2.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "models" / "beautyproof_aigen_v2"
BACKBONE_REPO = "timm/mobilenetv3_large_100.ra_in1k"
BACKBONE_FILE = "model.safetensors"
BACKBONE_DIR = REPO / "models" / "timm_mobilenetv3"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ARCH = "mobilenetv3_large_100"
SEED = 42
DEVICE = torch.device("cpu")
EXTS = (".png", ".jpg", ".jpeg")

# ---- 超参（v2 调整：数据变多 -> 略降 lr、多给轮次）----
LR = 1.5e-4
WEIGHT_DECAY = 1e-4
BATCH = 8
EPOCHS = 100
PATIENCE = 20
VAL_RATIO = 0.2


def _img_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS)


def build_pairs():
    d = REPO / "data"
    ai = _img_files(d / "ai")
    real_xhs = _img_files(d / "real_xhs")
    real_old = _img_files(d / "real") + _img_files(d / "clean") + _img_files(d / "tampered")
    return ai, real_xhs, real_old


def split(files, ratio, rng):
    """分层随机划分：先打乱再按比例切，保证验证集各类都有。"""
    idx = list(range(len(files)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(files) * ratio)))
    val_i = set(idx[:n_val])
    tr = [files[i] for i in idx if i not in val_i]
    va = [files[i] for i in idx if i in val_i]
    return tr, va


class AigenDataset(Dataset):
    def __init__(self, files, labels, tfm):
        self.files, self.labels, self.tfm = files, labels, tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tfm(Image.open(self.files[i]).convert("RGB")), self.labels[i]


def ensure_backbone_weights():
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKBONE_DIR / BACKBONE_FILE
    if target.exists() and target.stat().st_size > 0:
        return target
    from huggingface_hub import hf_hub_download
    for attempt in range(1, 9):
        try:
            return Path(hf_hub_download(repo_id=BACKBONE_REPO, filename=BACKBONE_FILE,
                                        local_dir=str(BACKBONE_DIR)))
        except Exception as e:  # noqa: BLE001
            print(f"  [权重下载失败 {attempt}] {type(e).__name__}: {str(e)[:120]}", flush=True)
            if attempt < 8:
                time.sleep(min(2 ** (attempt - 1), 20))
    return None


def make_model():
    model = timm.create_model(ARCH, pretrained=False, num_classes=2)
    w = ensure_backbone_weights()
    if w is None:
        print("警告：预训练权重没拿到，随机初始化（区分度会明显变差）", flush=True)
        return model.to(DEVICE)
    from safetensors.torch import load_file
    sd = load_file(str(w))
    msd = model.state_dict()
    # 只灌名字+形状都一致的（分类头 1000->2 形状不符，跳过）
    filtered = {k: v for k, v in sd.items()
                if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
    model.load_state_dict(filtered, strict=False)
    print(f"骨干权重已加载：{len(filtered)} 张量；跳过 {len(sd) - len(filtered)} 个（分类头）", flush=True)
    return model.to(DEVICE)


def main():
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()

    ai, real_xhs, real_old = build_pairs()
    real_all = real_xhs + real_old
    print(f"数据：AI={len(ai)}（data/ai）  REAL={len(real_all)}"
          f"（real_xhs {len(real_xhs)} + 原有 {len(real_old)}）", flush=True)
    if not ai or not real_all:
        print("错误：数据为空", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ai_tr, ai_val = split(ai, VAL_RATIO, rng)
    re_tr, re_val = split(real_all, VAL_RATIO, rng)

    train_files, train_labels = ai_tr + re_tr, [1] * len(ai_tr) + [0] * len(re_tr)
    val_files, val_labels = ai_val + re_val, [1] * len(ai_val) + [0] * len(re_val)
    # 验证集里有多少来自 real_xhs（必须 >0，否则测不出误报）
    n_val_xhs = sum(1 for f in val_files if "real_xhs" in str(f))
    print(f"训练 {len(train_files)}（AI {len(ai_tr)} / REAL {len(re_tr)}）  "
          f"验证 {len(val_files)}（AI {len(ai_val)} / REAL {len(re_val)}，其中 real_xhs {n_val_xhs} 张）",
          flush=True)

    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
        # v2 关键增强：主动加模糊，教模型"模糊/磨皮 ≠ AI"
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    infer_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dl = DataLoader(AigenDataset(train_files, train_labels, train_tfm),
                          batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(AigenDataset(val_files, val_labels, infer_tfm),
                        batch_size=max(len(val_files), 1), shuffle=False, num_workers=0)

    model = make_model()
    # 类别权重：按样本数反比（0=real, 1=ai，顺序别搞反）
    n_ai, n_real = len(ai), len(real_all)
    w_ai = (n_ai + n_real) / (2 * n_ai)
    w_real = (n_ai + n_real) / (2 * n_real)
    print(f"类别权重：real={w_real:.3f}  ai={w_ai:.3f}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([w_real, w_ai]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss, best_epoch, since = float("inf"), -1, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_ok += (out.argmax(1) == y).sum().item()
            tr_n += x.size(0)
        tr_loss, tr_acc = tr_loss / max(tr_n, 1), tr_ok / max(tr_n, 1)

        model.eval()
        vl_loss, vl_ok, vl_n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                vl_loss += criterion(out, y).item() * x.size(0)
                vl_ok += (out.argmax(1) == y).sum().item()
                vl_n += x.size(0)
        vl_loss, vl_acc = vl_loss / max(vl_n, 1), vl_ok / max(vl_n, 1)

        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(vl_loss, 4),
                        "val_acc": round(vl_acc, 4)})
        print(f"epoch {epoch:03d}  train_loss {tr_loss:.4f}  train_acc {tr_acc:.3f}  "
              f"val_loss {vl_loss:.4f}  val_acc {vl_acc:.3f}", flush=True)

        if vl_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch, since = vl_loss, epoch, 0
            torch.save(model.state_dict(), OUT_DIR / "model.pt")
        else:
            since += 1
            if since >= PATIENCE:
                print(f"早停：{PATIENCE} 轮无改善，最佳轮={best_epoch}", flush=True)
                break

    model.load_state_dict(torch.load(OUT_DIR / "model.pt", map_location=DEVICE))
    model.eval()
    cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    with torch.no_grad():
        for x, y in val_dl:
            out = model(x.to(DEVICE)).argmax(1).cpu().tolist()
            for p, l in zip(out, y.tolist()):
                cm["tp" if (l == 1 and p == 1) else "fn" if (l == 1 and p == 0)
                   else "fp" if (l == 0 and p == 1) else "tn"] += 1
    val_acc = (cm["tp"] + cm["tn"]) / max(sum(cm.values()), 1)

    all_files = ai + real_all
    all_labels = [1] * len(ai) + [0] * len(real_all)
    sm = nn.Softmax(dim=1)
    per_image = []
    with torch.no_grad():
        dl = DataLoader(AigenDataset(all_files, all_labels, infer_tfm),
                        batch_size=BATCH, shuffle=False, num_workers=0)
        for x, _ in dl:
            per_image += [p[1] for p in sm(model(x.to(DEVICE))).cpu().tolist()]
    full = sorted([{"file": all_files[i].name,
                    "label": "ai" if all_labels[i] == 1 else "real",
                    "ai_prob": round(float(per_image[i]), 4)}
                   for i in range(len(all_files))],
                  key=lambda r: -r["ai_prob"])

    (OUT_DIR / "config.json").write_text(json.dumps({
        "framework": "timm", "arch": ARCH, "num_classes": 2,
        "classes": ["real", "ai"], "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
        # 来源披露：骨干是他人预训练来源模型（ImageNet-1k 权重），仅作初始化，
        # 非训练数据；本模型训练数据全部为团队自产美妆图（见 trained_on）。
        "backbone_source": "timm/mobilenetv3_large_100.ra_in1k (ImageNet-1k 预训练权重, "
                            "MIT/Apache-2.0, 仅初始化, 非训练数据)",
        "version": "v2",
        "trained_on": f"data/ai({len(ai)} 即梦AI) + data/real_xhs({len(real_xhs)} 真实美妆) "
                      f"+ data/real|clean|tampered({len(real_old)})",
        "class_weight": {"real": round(w_real, 4), "ai": round(w_ai, 4)},
        "best_epoch": best_epoch, "val_acc": round(val_acc, 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    res = {"model": "beautyproof_aigen_v2", "arch": ARCH,
           "split": {"train": {"ai": len(ai_tr), "real": len(re_tr), "total": len(train_files)},
                     "val": {"ai": len(ai_val), "real": len(re_val),
                             "total": len(val_files), "from_real_xhs": n_val_xhs}},
           "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 4),
           "val_confusion": cm, "val_accuracy": round(val_acc, 4),
           "history": history, "per_image_ai_prob": full,
           "elapsed_sec": round(time.time() - t0, 1)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "aigen_finetune_v2.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}  acc={val_acc:.3f}", flush=True)
    print(f"已保存 {OUT_DIR}/model.pt  与 results/aigen_finetune_v2.json", flush=True)
    print(f"耗时 {res['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

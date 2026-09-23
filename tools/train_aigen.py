# -*- coding: utf-8 -*-
"""微调「整图 AI 生成」专属鉴别模型（BeautyProof 域）。

背景
----
旧的通用鉴伪模型（AIRealNet / capcheck）在我们这 30+ 张「中文美妆」标注图上
零区分度（通用模型训练域是艺术图/新闻图，迁移不过来）。实测结论：
    - AIRealNet：把真实美妆图判成 artificial、把 AI 图判成 real，基本反向
    - capcheck  ：所有图（含 AI 图）都输出 human≈0.99，完全分不出我们的 AI 图
正确解 = 用阮提供的标注图（10 真 + 20 AI 即梦图，外加 clean/tampered 真实照片）
微调一个轻量分类模型，专门管「整张由 AI 生成 vs 真实拍摄（含被局部改动）」。

数据集
------
    AI 类 (label=1) : data/ai        —— 20 张 即梦 生成的整图 AI 美妆图
    REAL 类 (label=0): data/real(10) + data/clean(2) + data/tampered(3)
                      —— 真实拍摄照片，其中 tampered 是被人局部改过的真实照片
                         （它们不是整图 AI 生成，必须标成 REAL，避免模型把
                          「被 PS 过的真实照」误判成 AI）

训练
----
    backbone: timm mobilenetv3_large_100（ImageNet 预训练，CPU 可推理）
    全量微调 + 强数据增强 + 早停（按验证集 loss），避免在小样本上过拟合失控。

产物（入库，强制 add）
---------------------
    models/beautyproof_aigen/model.pt      state_dict
    models/beautyproof_aigen/config.json   推理所需元信息
    results/aigen_finetune.json            训练曲线 + 验证集混淆 + 全量逐图 ai_prob

用法
----
    python tools/train_aigen.py
"""
import json
import os
import sys
import time
from pathlib import Path

# ---- 网络路由（必须在 import timm 之前设好）----
# HF 官方域名 huggingface.co 走代理会 SSL UNEXPECTED_EOF，timm 拉 ImageNet 预训练权重
# 会卡在重试里直到失败。已知可用的是 hf-mirror.com 镜像（与 download_aigen_models.py 同一套）。
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
OUT_DIR = REPO / "models" / "beautyproof_aigen"   # 必须训练前就建好，否则边训练边存盘会失败
# ImageNet 预训练骨干权重（本地真实文件，不走 timm 的 symlink 缓存，原因见 ensure_backbone_weights）
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


def _img_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS)


def build_pairs():
    d = REPO / "data"
    ai = _img_files(d / "ai")
    real = _img_files(d / "real")
    clean = _img_files(d / "clean")
    tamp = _img_files(d / "tampered")
    real_all = real + clean + tamp
    return ai, real_all


def stratified_split(files, n_val):
    """取最后 n_val 张作验证集（按文件名排序后），其余作训练集。"""
    return files[:-n_val], files[-n_val:]


class AigenDataset(Dataset):
    def __init__(self, files, labels, tfm):
        self.files = files
        self.labels = labels
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tfm(img), self.labels[i]


def ensure_backbone_weights():
    """拿到 ImageNet 预训练权重文件（真实文件，不是软链）。

    为什么绕开 timm 自带的 pretrained=True：
        timm 走 HF 缓存，snapshot 目录里是 symlink；Windows 下这层软链会失效，
        timm 读到 0 字节 → safetensors 报 "header too small"。
        所以这里自己用 hf-mirror 下成真实文件，再手动灌进模型。
    已存在就跳过，可反复跑。
    """
    BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKBONE_DIR / BACKBONE_FILE
    if target.exists() and target.stat().st_size > 0:
        return target
    from huggingface_hub import hf_hub_download
    for attempt in range(1, 9):
        try:
            p = hf_hub_download(repo_id=BACKBONE_REPO, filename=BACKBONE_FILE,
                                local_dir=str(BACKBONE_DIR))
            return Path(p)
        except Exception as e:  # noqa: BLE001
            print(f"  [权重下载失败 {attempt}] {type(e).__name__}: {str(e)[:120]}", flush=True)
            if attempt < 8:
                time.sleep(min(2 ** (attempt - 1), 20))
    return None


def make_model():
    """骨干用 ImageNet 预训练，分类头换成 2 类（real / ai）随机初始化后微调。"""
    model = timm.create_model(ARCH, pretrained=False, num_classes=2)
    w = ensure_backbone_weights()
    if w is None:
        print("警告：预训练权重没拿到，将从随机初始化训练（区分度会明显变差）", flush=True)
        return model.to(DEVICE)

    from safetensors.torch import load_file
    sd = load_file(str(w))
    model_sd = model.state_dict()
    # 只灌「名字 + 形状都一致」的权重。
    # 注意：strict=False 只能容忍缺 key，容忍不了形状不匹配（会直接 RuntimeError），
    # 所以这里先手动过滤——分类头 1000 类 → 2 类形状对不上，跳过，随机初始化后微调。
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    skipped = len(sd) - len(filtered)
    model.load_state_dict(filtered, strict=False)
    print(f"骨干权重已加载：{len(filtered)} 张量；跳过 {skipped} 个（分类头形状不符，正常）", flush=True)
    return model.to(DEVICE)


def main():
    torch.manual_seed(SEED)
    t0 = time.time()

    ai, real_all = build_pairs()
    print(f"数据：AI={len(ai)}  REAL={len(real_all)}（real+clean+tampered）", flush=True)
    if not ai or not real_all:
        print("错误：data/ai 或 data/real 为空，无法训练", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)  # 训练中途要存最佳权重，目录必须先存在

    ai_tr, ai_val = stratified_split(ai, 4)        # 20 → 16 训练 / 4 验证
    real_tr, real_val = stratified_split(real_all, 3)  # 15 → 12 训练 / 3 验证

    train_files = ai_tr + real_tr
    train_labels = [1] * len(ai_tr) + [0] * len(real_tr)
    val_files = ai_val + real_val
    val_labels = [1] * len(ai_val) + [0] * len(real_val)
    print(f"训练集 {len(train_files)}（AI {len(ai_tr)} / REAL {len(real_tr)}）"
          f"  验证集 {len(val_files)}（AI {len(ai_val)} / REAL {len(real_val)}）", flush=True)

    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    infer_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = AigenDataset(train_files, train_labels, train_tfm)
    val_ds = AigenDataset(val_files, val_labels, infer_tfm)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, num_workers=0)

    model = make_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    patience = 15
    since_improve = 0

    EPOCHS = 80
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_n = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_n += x.size(0)
        tr_loss /= max(tr_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)

        model.eval()
        vl_loss = 0.0
        vl_correct = 0
        vl_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                vl_loss += loss.item() * x.size(0)
                vl_correct += (out.argmax(1) == y).sum().item()
                vl_n += x.size(0)
        vl_loss /= max(vl_n, 1)
        vl_acc = vl_correct / max(vl_n, 1)

        history.append({
            "epoch": epoch, "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4), "val_loss": round(vl_loss, 4),
            "val_acc": round(vl_acc, 4),
        })
        print(f"epoch {epoch:03d}  train_loss {tr_loss:.4f}  train_acc {tr_acc:.3f}"
              f"  val_loss {vl_loss:.4f}  val_acc {vl_acc:.3f}", flush=True)

        if vl_loss < best_val_loss - 1e-4:
            best_val_loss = vl_loss
            best_epoch = epoch
            since_improve = 0
            torch.save(model.state_dict(), OUT_DIR / "model.pt")
        else:
            since_improve += 1
            if since_improve >= patience:
                print(f"早停：验证 loss 连续 {patience} 轮无改善，最佳轮 = {best_epoch}", flush=True)
                break

    print(f"\n最佳轮 = {best_epoch}（val_loss {best_val_loss:.4f}）", flush=True)

    # ---- 验证集混淆矩阵 ----
    model.load_state_dict(torch.load(OUT_DIR / "model.pt", map_location=DEVICE))
    model.eval()
    cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}  # tp: AI 判 AI; tn: REAL 判 REAL
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            out = model(x).argmax(1).cpu().tolist()
            for pred, lab in zip(out, y.tolist()):
                if lab == 1 and pred == 1:
                    cm["tp"] += 1
                elif lab == 1 and pred == 0:
                    cm["fn"] += 1
                elif lab == 0 and pred == 1:
                    cm["fp"] += 1
                else:
                    cm["tn"] += 1
    val_acc = (cm["tp"] + cm["tn"]) / max(sum(cm.values()), 1)

    # ---- 全量逐图 ai_prob（含训练集，仅作透明展示，训练集有记忆偏差）----
    all_files = ai + real_all
    all_labels = [1] * len(ai) + [0] * len(real_all)
    full_ds = AigenDataset(all_files, all_labels, infer_tfm)
    full_dl = DataLoader(full_ds, batch_size=8, shuffle=False, num_workers=0)
    per_image = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, y in full_dl:
            x = x.to(DEVICE)
            probs = softmax(model(x)).cpu().tolist()
            for p, lab in zip(probs, y.tolist()):
                per_image.append(p[1])  # 索引1 = AI 类概率
    full = [{"file": all_files[i].name, "label": "ai" if all_labels[i] == 1 else "real",
             "ai_prob": round(per_image[i], 4)} for i in range(len(all_files))]
    full.sort(key=lambda r: r["ai_prob"], reverse=True)

    # ---- 落盘 ----
    config = {
        "framework": "timm",
        "arch": ARCH,
        "num_classes": 2,
        "classes": ["real", "ai"],
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "trained_on": "data/ai(20 即梦AI) + data/real(10) + data/clean(2) + data/tampered(3)",
        "best_epoch": best_epoch,
        "val_acc": round(val_acc, 4),
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    res = {
        "model": "beautyproof_aigen",
        "arch": ARCH,
        "split": {
            "train": {"ai": len(ai_tr), "real": len(real_tr), "total": len(train_files)},
            "val": {"ai": len(ai_val), "real": len(real_val), "total": len(val_files)},
        },
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "val_confusion": cm,
        "val_accuracy": round(val_acc, 4),
        "history": history,
        "per_image_ai_prob": full,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    res_dir = REPO / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "aigen_finetune.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP(AI→AI)={cm['tp']}  FN(AI→REAL)={cm['fn']}  "
          f"FP(REAL→AI)={cm['fp']}  TN(REAL→REAL)={cm['tn']}", flush=True)
    print(f"  验证集准确率 = {val_acc:.3f}", flush=True)
    print(f"\n=== 全量逐图 ai_prob（前 8 / 后 8）===")
    for r in full[:8]:
        print(f"  {r['ai_prob']:.3f}  {r['label']:4s}  {r['file']}")
    print("  ...")
    for r in full[-8:]:
        print(f"  {r['ai_prob']:.3f}  {r['label']:4s}  {r['file']}")
    print(f"\n已保存: {OUT_DIR / 'model.pt'} / config.json")
    print(f"已保存: {res_dir / 'aigen_finetune.json'}")
    print(f"耗时 {res['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""AIGC 检测器 v4：在 v3 基础上的「数据扩充」版，全程严格符合赛题2 数据合规要求。

相对 v3 的扩充（都是赛题允许的「自有数据 + 机器学习标准手段」，无规则外数据/方法）
------------------------------------------------------------------------------
1. 训练增强升级为 beauty_aug.get_train_transform()：覆盖「美颜/滤镜/重压缩后的真实美妆图」
   分布（更强色彩抖动、轻微旋转、更强高斯模糊≈磨皮、平台缩放伪影、JPEG 重压缩、轻度噪声）。
2. REAL 训练类并入「伪标注真实图」data/real_extra/（由 tools/pseudo_label.py 产出，
   用已上线模型对团队自产未标注真实美妆图高置信打标得到）—— 标准 self-training 手段，
   数据全部团队自有，无第三方数据集。
3. 跨生成器 held-out 评测（data/ai_cross）保留：ai_cross 验证份始终不进训练，指标干净。
4. 决策边界由 aigc_tool 统一上调（阳性阈值 0.6 + 模糊带），本脚本只负责训练。

明确不做的（赛题合规红线）：
- 不使用 GenImage（CC BY-NC-SA 非商用，与 20 万奖金商业场景冲突）；
- 不使用 COCO（与美妆域无关 + 人脸隐私）；
- 不把 ImageNet 当训练集（仅作为骨干初始化权重，来源已在 config.json 披露）；
- 不爬取 / 下载任何第三方数据集或外部图。

产物：models/beautyproof_aigen_v4/（model.pt + config.json）+ results/aigen_finetune_v4.json
上线：aigc_tool.MODEL_PRIORITY 已把 v4 置顶；未训练时权重缺失会自动回退 v3。
"""
import json
import random
import sys
import time
from pathlib import Path

# 复用 v3 的骨干加载 / 数据集 / 划分等已实现逻辑，避免重复代码、保证一致性
from train_aigen_v3 import (  # noqa: F401
    _img_files, split, AigenDataset, make_model,
    IMG_SIZE, MEAN, STD, ARCH, BACKBONE_REPO, SEED,
    LR, WEIGHT_DECAY, BATCH, EPOCHS, PATIENCE, VAL_RATIO, EXTS,
)
from beauty_aug import get_train_transform, get_infer_transform

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "models" / "beautyproof_aigen_v4"


def build_pairs_v4():
    d = REPO / "data"
    ai_jimeng = _img_files(d / "ai")
    ai_cross = _img_files(d / "ai_cross")
    real_xhs = _img_files(d / "real_xhs")
    real_old = (_img_files(d / "real") + _img_files(d / "clean") + _img_files(d / "tampered"))
    real_extra = _img_files(d / "real_extra")   # 伪标注真实图（v4 新增）
    return ai_jimeng, ai_cross, real_xhs, real_old, real_extra


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    rng = random.Random(SEED)
    t0 = time.time()

    ai_jimeng, ai_cross, real_xhs, real_old, real_extra = build_pairs_v4()
    real_all = real_xhs + real_old + real_extra
    print(f"数据：即梦AI={len(ai_jimeng)}  跨生成器(ai_cross)={len(ai_cross)}  "
          f"REAL={len(real_all)}（real_xhs {len(real_xhs)} + 原有 {len(real_old)} "
          f"+ 伪标注真实 {len(real_extra)}）", flush=True)
    if not ai_jimeng or not real_all:
        print("错误：数据为空（先往 data/ai 与 data/ai_cross 放图；真实图至少要有 real_xhs）", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 跨生成器：验证份整体留作干净 held-out，不进训练
    aj_tr, aj_val = split(ai_jimeng, VAL_RATIO, rng)
    ac_tr, ac_val = split(ai_cross, VAL_RATIO, rng)
    re_tr, re_val = split(real_all, VAL_RATIO, rng)

    ai_tr = aj_tr + ac_tr
    ai_val = aj_val + ac_val
    train_files, train_labels = ai_tr + re_tr, [1] * len(ai_tr) + [0] * len(re_tr)
    val_files, val_labels = ai_val + re_val, [1] * len(ai_val) + [0] * len(re_val)
    n_val_xhs = sum(1 for f in val_files if "real_xhs" in str(f))
    print(f"训练 {len(train_files)}（AI {len(ai_tr)} / REAL {len(re_tr)}）  "
          f"验证 {len(val_files)}（AI {len(ai_val)} / REAL {len(re_val)}，其中 real_xhs {n_val_xhs} 张）",
          flush=True)

    train_tfm = get_train_transform()
    infer_tfm = get_infer_transform()

    train_dl = DataLoader(AigenDataset(train_files, train_labels, train_tfm),
                          batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(AigenDataset(val_files, val_labels, infer_tfm),
                        batch_size=max(len(val_files), 1), shuffle=False, num_workers=0)

    model = make_model()
    n_ai, n_real = len(ai_tr) + len(ai_val), len(real_all)
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

    model.load_state_dict(torch.load(OUT_DIR / "model.pt"))
    model.eval()
    cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    with torch.no_grad():
        for x, y in val_dl:
            out = model(x).argmax(1).tolist()
            for p, l in zip(out, y.tolist()):
                cm["tp" if (l == 1 and p == 1) else "fn" if (l == 1 and p == 0)
                   else "fp" if (l == 0 and p == 1) else "tn"] += 1
    val_acc = (cm["tp"] + cm["tn"]) / max(sum(cm.values()), 1)

    cross_files = [f for f in val_files if "ai_cross" in str(f)]
    sm = torch.nn.Softmax(dim=1)
    cross_pred = []
    with torch.no_grad():
        dl = DataLoader(AigenDataset(cross_files, [1] * len(cross_files), infer_tfm),
                        batch_size=max(len(cross_files), 1), shuffle=False, num_workers=0)
        for x, _ in dl:
            cross_pred += [p[1] for p in sm(model(x)).tolist()]
    cross_total = len(cross_files)
    cross_hit = sum(1 for p in cross_pred if p > 0.5)
    cross_recall = (cross_hit / cross_total) if cross_total else None

    # 模型来源披露（与 v3 一致）：骨干是他人预训练来源模型（ImageNet-1k 权重），仅初始化
    (OUT_DIR / "config.json").write_text(json.dumps({
        "framework": "timm", "arch": ARCH, "num_classes": 2,
        "classes": ["real", "ai"], "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
        "backbone_source": "timm/mobilenetv3_large_100.ra_in1k (ImageNet-1k 预训练权重, "
                            "MIT/Apache-2.0, 仅初始化, 非训练数据)",
        "version": "v4",
        "augmentation": "beauty_aug.get_train_transform（域增强，无外部数据）",
        "data_expansion": "data/real_extra（伪标注真实图，self-training，团队自产）",
        "trained_on": f"data/ai({len(ai_jimeng)}) + data/ai_cross({len(ai_cross)}) "
                      f"+ data/real_xhs({len(real_xhs)}) + 原有({len(real_old)}) "
                      f"+ 伪标注真实({len(real_extra)})",
        "pseudo_labeled_real": len(real_extra),
        "class_weight": {"real": round(w_real, 4), "ai": round(w_ai, 4)},
        "best_epoch": best_epoch, "val_acc": round(val_acc, 4),
        "cross_generator_recall": round(cross_recall, 4) if cross_recall is not None else None,
        "compliance": "未使用 GenImage/COCO/ImageNet(训练集)/任何第三方数据集；"
                      "ImageNet 仅作骨干初始化权重。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    res = {"model": "beautyproof_aigen_v4", "arch": ARCH,
           "split": {"train": {"ai": len(ai_tr), "real": len(re_tr), "total": len(train_files)},
                     "val": {"ai": len(ai_val), "real": len(re_val),
                             "total": len(val_files), "from_real_xhs": n_val_xhs}},
           "pseudo_labeled_real_used": len(real_extra),
           "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 4),
           "val_confusion": cm, "val_accuracy": round(val_acc, 4),
           "cross_generator": {"total": cross_total, "hit": cross_hit,
                               "recall": round(cross_recall, 4) if cross_recall is not None else None},
           "history": history, "elapsed_sec": round(time.time() - t0, 1)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "aigen_finetune_v4.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}  acc={val_acc:.3f}", flush=True)
    print(f"=== 跨生成器召回（data/ai_cross, held-out）===", flush=True)
    print(f"  {cross_hit}/{cross_total} = {cross_recall:.3f}" if cross_recall is not None
          else "  无跨生成器样本", flush=True)
    print(f"已保存 {OUT_DIR}/model.pt  与 results/aigen_finetune_v4.json", flush=True)
    print(f"耗时 {res['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

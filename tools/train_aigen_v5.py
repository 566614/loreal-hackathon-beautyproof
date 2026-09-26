# -*- coding: utf-8 -*-
"""AIGC 检测器 v5：在 v4 基础上并入「人工已标注真实批次」，主攻真实世界误报。

相对 v4 的唯一变化（配方完全一致，都是赛题允许的自有数据 + 标准手段）
------------------------------------------------------------------------------
REAL 训练类并入 data/real_xhs_batch2/（24 张人工确认的真实美妆照，来自团队 2026-09-26
通过微信收集的「真图.zip」34 张；另外 10 张划入 data/realworld_heldout/ 永不训练，
作为真实世界 held-out 测试集，保证「训练前后」对比诚实可复核）。

为什么要这一批：
    v4 在这 34 张上的基线误报率 5.9%（2/34），且两张误判都是「妆面精致的滤镜自拍」
    （prob=1.0 / 0.9792 的自信错判）——与 2026-09-24 发现的「滤镜磨皮美妆照被学成 AI」
    是同一失败模式。本批生活流随手拍 + 精致自拍直接标注 REAL 进训练，针对性补域。

合规边界（与 v4 相同，红线不碰）：
- 不使用 GenImage（CC BY-NC-SA 非商用）/ COCO / ImageNet（作训练集）；
- 不爬取任何第三方数据集；本批为团队自产/队友提供的真实照片，人工标注，来源可溯源
  （results/realworld_split_20260926.json 记录划分与来源）。

产物：models/beautyproof_aigen_v5/（model.pt + config.json）+ results/aigen_finetune_v5.json
上线：aigc_tool.MODEL_PRIORITY 置顶 v5；未训练时权重缺失自动回退 v4。
"""
import json
import random
import sys
import time
from pathlib import Path

# 复用 v3 的骨干加载 / 数据集 / 划分等已实现逻辑，保证与 v4 配方一致、可横向对比
from train_aigen_v3 import (  # noqa: F401
    _img_files, split, AigenDataset, make_model,
    IMG_SIZE, MEAN, STD, ARCH, BACKBONE_REPO, SEED,
    LR, WEIGHT_DECAY, BATCH, EPOCHS, PATIENCE, VAL_RATIO, EXTS,
)
from beauty_aug import get_train_transform, get_infer_transform

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "models" / "beautyproof_aigen_v5"


def build_pairs_v5():
    d = REPO / "data"
    ai_jimeng = _img_files(d / "ai")
    ai_cross = _img_files(d / "ai_cross")
    real_xhs = _img_files(d / "real_xhs")
    real_old = (_img_files(d / "real") + _img_files(d / "clean") + _img_files(d / "tampered"))
    real_extra = _img_files(d / "real_extra")        # 伪标注真实图（若跑过 pseudo_label）
    real_batch2 = _img_files(d / "real_xhs_batch2")  # v5 新增：人工标注真实批次（真图.zip）
    return ai_jimeng, ai_cross, real_xhs, real_old, real_extra, real_batch2


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    rng = random.Random(SEED)
    t0 = time.time()

    ai_jimeng, ai_cross, real_xhs, real_old, real_extra, real_batch2 = build_pairs_v5()
    real_all = real_xhs + real_old + real_extra + real_batch2
    print(f"数据：即梦AI={len(ai_jimeng)}  跨生成器(ai_cross)={len(ai_cross)}  "
          f"REAL={len(real_all)}（real_xhs {len(real_xhs)} + 原有 {len(real_old)} "
          f"+ 伪标注 {len(real_extra)} + 真图批次 {len(real_batch2)}）", flush=True)
    if not ai_jimeng or not real_all:
        print("错误：数据为空", flush=True)
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
    n_val_b2 = sum(1 for f in val_files if "real_xhs_batch2" in str(f))
    print(f"训练 {len(train_files)}（AI {len(ai_tr)} / REAL {len(re_tr)}）  "
          f"验证 {len(val_files)}（AI {len(ai_val)} / REAL {len(re_val)}，"
          f"其中 real_xhs {n_val_xhs} / 真图批次 {n_val_b2}）", flush=True)

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

    # 模型来源披露（与 v3/v4 一致）：骨干是他人预训练来源模型（ImageNet-1k 权重），仅初始化
    (OUT_DIR / "config.json").write_text(json.dumps({
        "framework": "timm", "arch": ARCH, "num_classes": 2,
        "classes": ["real", "ai"], "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
        "backbone_source": "timm/mobilenetv3_large_100.ra_in1k (ImageNet-1k 预训练权重, "
                            "MIT/Apache-2.0, 仅初始化, 非训练数据)",
        "version": "v5",
        "augmentation": "beauty_aug.get_train_transform（域增强，无外部数据）",
        "data_expansion": "data/real_xhs_batch2（人工标注真实批次，来源 真图.zip，团队自产/队友提供）",
        "trained_on": f"data/ai({len(ai_jimeng)}) + data/ai_cross({len(ai_cross)}) "
                      f"+ data/real_xhs({len(real_xhs)}) + 原有({len(real_old)}) "
                      f"+ 伪标注({len(real_extra)}) + 真图批次({len(real_batch2)})",
        "human_labeled_real_batch2": len(real_batch2),
        "pseudo_labeled_real": len(real_extra),
        "class_weight": {"real": round(w_real, 4), "ai": round(w_ai, 4)},
        "best_epoch": best_epoch, "val_acc": round(val_acc, 4),
        "cross_generator_recall": round(cross_recall, 4) if cross_recall is not None else None,
        "compliance": "未使用 GenImage/COCO/ImageNet(训练集)/任何第三方数据集；"
                      "ImageNet 仅作骨干初始化权重。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    res = {"model": "beautyproof_aigen_v5", "arch": ARCH,
           "split": {"train": {"ai": len(ai_tr), "real": len(re_tr), "total": len(train_files)},
                     "val": {"ai": len(ai_val), "real": len(re_val),
                             "total": len(val_files), "from_real_xhs": n_val_xhs,
                             "from_real_batch2": n_val_b2}},
           "human_labeled_real_used": len(real_batch2),
           "pseudo_labeled_real_used": len(real_extra),
           "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 4),
           "val_confusion": cm, "val_accuracy": round(val_acc, 4),
           "cross_generator": {"total": cross_total, "hit": cross_hit,
                               "recall": round(cross_recall, 4) if cross_recall is not None else None},
           "history": history, "elapsed_sec": round(time.time() - t0, 1)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "aigen_finetune_v5.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}  acc={val_acc:.3f}", flush=True)
    print(f"=== 跨生成器召回（data/ai_cross, held-out）===", flush=True)
    print(f"  {cross_hit}/{cross_total} = {cross_recall:.3f}" if cross_recall is not None
          else "  无跨生成器样本", flush=True)
    print(f"已保存 {OUT_DIR}/model.pt  与 results/aigen_finetune_v5.json", flush=True)
    print(f"耗时 {res['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

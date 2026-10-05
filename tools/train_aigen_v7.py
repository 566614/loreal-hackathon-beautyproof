# -*- coding: utf-8 -*-
"""AIGC 检测器 v7：v4 配方 + 「精致滤镜自拍」真实图（data/real_filter_selfie 训练份）。

单变量对照（reviewer P1-3 要求每轮只改一个主变量）
------------------------------------------------------------------------------
相对 v4（生产模型）唯一的变化：REAL 训练类并入 data/real_filter_selfie/SPLIT_MANIFEST.json
里 train 段的 20 张精致滤镜自拍真实图。模型结构 / 增强 / 类别权重公式 / 超参 / 跨生成器
划分全部与 v4 完全一致；**不包含** v6 的跨生成器 Flux 训练改动（v6 已诚实归档未上线）。

动机：realworld_heldout 的唯一 high_risk 误报样本 Image_1790435103363.jpg（aigc=0.9792）
与阈值扫描结论（判别面饱和）都指向「精致滤镜自拍」这一真实风格不在 v4 训练分布内。
阈值实验已证明调阈值无用，故本轮动样本。

诚实纪律
- data/real_filter_selfie/heldout 段 10 张**永不训练/调参**，仅作误报评测（SPLIT_MANIFEST
  在训练前已冻结，seed=20260929）。
- 30 张图与库内所有集合 md5 零重叠（见 PROVENANCE.md 全库查重记录）。
- 来源登记：data/real_filter_selfie/PROVENANCE.md（团队自备素材，非第三方数据集）。

合规红线（与 v4 相同）
- 不使用 GenImage/COCO/ImageNet(训练集)/任何无来源第三方数据集；
- ImageNet 仅作骨干初始化权重；不把真人照片混入 AI 类训练集。

产物：models/beautyproof_aigen_v7/（model.pt + config.json）+ results/aigen_finetune_v7.json
上线：仅在晋升判定通过后由 eval 脚本核验并切换 aigc_tool.MODEL_PRIORITY；不自动上线。
"""
import json
import random
import sys
import time
from pathlib import Path

# 复用 v3/v4 的骨干加载 / 数据集 / 划分 / 超参，保证与 v4 只差数据
from train_aigen_v3 import (  # noqa: F401
    _img_files, split, AigenDataset, make_model,
    IMG_SIZE, MEAN, STD, ARCH, BACKBONE_REPO, SEED,
    LR, WEIGHT_DECAY, BATCH, EPOCHS, PATIENCE, VAL_RATIO, EXTS,
)
from beauty_aug import get_train_transform, get_infer_transform

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "models" / "beautyproof_aigen_v7"
FS_SPLIT = REPO / "data" / "real_filter_selfie" / "SPLIT_MANIFEST.json"


def load_fs_train():
    """读训练前冻结的切分清单，只取 train 段（heldout 段绝不进训练）。"""
    m = json.loads(FS_SPLIT.read_text(encoding="utf-8"))
    assert m["schema"] == "beautyproof/real_filter_selfie_split@1", "切分清单版本不符"
    d = FS_SPLIT.parent
    files = [d / f for f in m["train"]]
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, f"清单文件缺失（可能被移动/改名）: {missing}"
    return files, m["heldout_never_train"], m["seed"]


def build_pairs_v7():
    d = REPO / "data"
    ai_jimeng = _img_files(d / "ai")
    ai_cross = _img_files(d / "ai_cross")
    real_xhs = _img_files(d / "real_xhs")
    real_old = (_img_files(d / "real") + _img_files(d / "clean") + _img_files(d / "tampered"))
    real_extra = _img_files(d / "real_extra")   # 伪标注真实图（v4 引入）
    return ai_jimeng, ai_cross, real_xhs, real_old, real_extra


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    rng = random.Random(SEED)
    t0 = time.time()

    ai_jimeng, ai_cross, real_xhs, real_old, real_extra = build_pairs_v7()
    fs_train, fs_heldout, fs_seed = load_fs_train()
    real_all = real_xhs + real_old + real_extra + fs_train
    print(f"数据：即梦AI={len(ai_jimeng)}  跨生成器(ai_cross)={len(ai_cross)}  "
          f"REAL={len(real_all)}（real_xhs {len(real_xhs)} + 原有 {len(real_old)} "
          f"+ 伪标注 {len(real_extra)} + 滤镜自拍 {len(fs_train)}）", flush=True)
    print(f"滤镜自拍 held-out（永不训练）{len(fs_heldout)} 张（冻结 seed={fs_seed}）", flush=True)
    if not ai_jimeng or not real_all:
        print("错误：数据为空", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 与 v4 完全相同的划分方式（同 SEED、同 VAL_RATIO、同随机消耗顺序）
    aj_tr, aj_val = split(ai_jimeng, VAL_RATIO, rng)
    ac_tr, ac_val = split(ai_cross, VAL_RATIO, rng)
    re_tr, re_val = split(real_all, VAL_RATIO, rng)

    ai_tr = aj_tr + ac_tr
    ai_val = aj_val + ac_val
    train_files, train_labels = ai_tr + re_tr, [1] * len(ai_tr) + [0] * len(re_tr)
    val_files, val_labels = ai_val + re_val, [1] * len(ai_val) + [0] * len(re_val)
    n_val_fs = sum(1 for f in val_files if "real_filter_selfie" in str(f))
    print(f"训练 {len(train_files)}（AI {len(ai_tr)} / REAL {len(re_tr)}）  "
          f"验证 {len(val_files)}（AI {len(ai_val)} / REAL {len(re_val)}，"
          f"其中滤镜自拍 {n_val_fs} 张）", flush=True)

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

    (OUT_DIR / "config.json").write_text(json.dumps({
        "framework": "timm", "arch": ARCH, "num_classes": 2,
        "classes": ["real", "ai"], "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
        "backbone_source": "timm/mobilenetv3_large_100.ra_in1k (ImageNet-1k 预训练权重, "
                            "MIT/Apache-2.0, 仅初始化, 非训练数据)",
        "version": "v7",
        "augmentation": "beauty_aug.get_train_transform（域增强，无外部数据）",
        "single_variable_vs_v4": "仅新增 data/real_filter_selfie train 段 20 张精致滤镜自拍真实图"
                                 "（SPLIT_MANIFEST 冻结 seed=20260929；不含 v6 改动）",
        "trained_on": f"data/ai({len(ai_jimeng)}) + data/ai_cross({len(ai_cross)}) "
                      f"+ data/real_xhs({len(real_xhs)}) + 原有({len(real_old)}) "
                      f"+ 伪标注真实({len(real_extra)}) + 滤镜自拍({len(fs_train)})",
        "filter_selfie": {"train": len(fs_train), "heldout_never_train": len(fs_heldout),
                          "provenance": "data/real_filter_selfie/PROVENANCE.md"},
        "pseudo_labeled_real": len(real_extra),
        "class_weight": {"real": round(w_real, 4), "ai": round(w_ai, 4)},
        "best_epoch": best_epoch, "val_acc": round(val_acc, 4),
        "cross_generator_recall": round(cross_recall, 4) if cross_recall is not None else None,
        "compliance": "未使用 GenImage/COCO/ImageNet(训练集)/任何第三方数据集；"
                      "ImageNet 仅作骨干初始化权重；真人照片仅作真实类，不入 AI 类。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    res = {"model": "beautyproof_aigen_v7", "arch": ARCH,
           "single_variable_vs_v4": "仅+滤镜自拍真实图20张（train份，heldout 10张永不训练）",
           "filter_selfie": {"train": len(fs_train), "heldout_never_train": len(fs_heldout),
                             "split_seed": fs_seed},
           "split": {"train": {"ai": len(ai_tr), "real": len(re_tr), "total": len(train_files)},
                     "val": {"ai": len(ai_val), "real": len(re_val),
                             "total": len(val_files), "from_filter_selfie": n_val_fs}},
           "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 4),
           "val_confusion": cm, "val_accuracy": round(val_acc, 4),
           "cross_generator": {"total": cross_total, "hit": cross_hit,
                               "recall": round(cross_recall, 4) if cross_recall is not None else None},
           "history": history, "elapsed_sec": round(time.time() - t0, 1)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "aigen_finetune_v7.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}  acc={val_acc:.3f}", flush=True)
    print("=== 跨生成器召回（data/ai_cross, held-out）===", flush=True)
    print(f"  {cross_hit}/{cross_total} = {cross_recall:.3f}" if cross_recall is not None
          else "  无跨生成器样本", flush=True)
    print(f"已保存 {OUT_DIR}/model.pt  与 results/aigen_finetune_v7.json", flush=True)
    print(f"耗时 {res['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

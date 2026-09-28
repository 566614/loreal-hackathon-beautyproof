# -*- coding: utf-8 -*-
"""AIGC 检测器 v6：并入「有来源标注的跨生成器」美妆图（Flux/SD/MJ），主攻跨生成器召回。

相对 v5 的变化
------------------------------------------------------------------------------
v5 只在「即梦族」上微调，跨生成器零样本召回仅 ~58%（7/12）。v6 的设计：
(1) 维持 ai_cross（26 张 ImageGen）整体 held-out（reviewer P1，且与 ai_cross_native 零重叠），
    作为独立验证信号；实测表明把 ai_cross 纳入训练反而稀释模型对 ai_cross_native 中
    「生活流/滤镜自拍」难例的判别力，故回退。
(2) 把「可证明为 AI 生成、来源清晰、合成无真人」的 Flux 跨生成器图（来自 data/ai_mj_sd_flux/，
    由 tools/collect_crossgen.py 收编并校验 PROVENANCE，取自 SFHQ-T2I Kaggle MIT）并入训练，
    让模型见过来真实合成的 Flux 人脸风格，提供真正「跨生成器」的鲁棒性。
诚实基线：ai_cross_native 12 张中有 6 张为「生活流/滤镜自拍/插画/杂志封面」等伪真实感难例，
模型对其自信判真；仅凭少量 Flux 样本难以突破，≥80% 召回需更多同类难例训练图（本 MIT 源仅能
逐文件取到 Flux，SD/MJ 不暴露），故目标为「不退化 + 尽量提升」。

合规边界（红线）
- 来源：SFHQ-T2I（Kaggle, MIT）或本人订阅生成的 Flux/SD/MJ；均为合成图，无真实人物肖像权风险。
- 严禁无来源标注的第三方数据集；严禁把真实人物照片当 AI 标签。
- 跨生成器图进训练前必须过 collect_crossgen.py 的来源登记校验。
- ai_cross（26 张 ImageGen）与 ai_cross_native（12 张 ImageGen）**零重叠**；ai_cross 维持
  held-out 作为独立验证信号，ai_cross_native 始终严格零样本（无泄漏、无虚高）。
- 保留两个「永不训练」零样本验证集，避免自我虚高：
    * data/ai_cross_native/（原 12 张，ImageGen，始终零样本）
    * data/ai_mj_sd_flux_heldout/（每生成器切出的少量，验证对具体生成器的真实召回）

产物：models/beautyproof_aigen_v6/（model.pt + config.json）+ results/aigen_finetune_v6.json
上线：aigc_tool.MODEL_PRIORITY 置顶 v6；未训练时权重缺失自动回退 v5。
"""
import json
import random
import sys
import time
from pathlib import Path

from train_aigen_v3 import (  # noqa: F401
    _img_files, split, AigenDataset, make_model,
    IMG_SIZE, MEAN, STD, ARCH, BACKBONE_REPO, SEED,
    LR, WEIGHT_DECAY, BATCH, EPOCHS, PATIENCE, VAL_RATIO, EXTS,
)
from beauty_aug import get_train_transform, get_infer_transform

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "models" / "beautyproof_aigen_v6"


def build_pairs_v6():
    d = REPO / "data"
    ai_jimeng = _img_files(d / "ai")
    ai_cross = _img_files(d / "ai_cross")
    real_xhs = _img_files(d / "real_xhs")
    real_old = (_img_files(d / "real") + _img_files(d / "clean") + _img_files(d / "tampered"))
    real_extra = _img_files(d / "real_extra")
    real_batch2 = _img_files(d / "real_xhs_batch2")
    # v6 新增：带来源登记的跨生成器训练图（Flux/SD/MJ）—— 递归读取子目录
    ai_crossgen = sorted(p for p in (d / "ai_mj_sd_flux").rglob("*") if p.suffix.lower() in EXTS)
    return (ai_jimeng, ai_cross, real_xhs, real_old, real_extra,
            real_batch2, ai_crossgen)


def group_split(files, val_ratio, rng):
    """按「来源组」整体拆分 train/val（reviewer P1：避免同源变体跨 train/val 泄漏）。"""
    groups = {}
    for f in files:
        groups.setdefault(Path(f).parent.name, []).append(f)
    group_names = sorted(groups.keys())
    rng.shuffle(group_names)
    train, val, val_count = [], [], 0
    val_target = val_ratio * len(files)
    for g in group_names:
        gfiles = groups[g]
        if val_count < val_target:
            val.extend(gfiles)
            val_count += len(gfiles)
        else:
            train.extend(gfiles)
    return train, val


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    rng = random.Random(SEED)
    t0 = time.time()

    (ai_jimeng, ai_cross, real_xhs, real_old, real_extra,
     real_batch2, ai_crossgen) = build_pairs_v6()
    real_all = real_xhs + real_old + real_extra + real_batch2
    print(f"数据：即梦AI={len(ai_jimeng)}  跨生成器(ai_cross)={len(ai_cross)}  "
          f"跨生成器新(Flux/SD/MJ)={len(ai_crossgen)}  "
          f"REAL={len(real_all)}", flush=True)
    if not ai_jimeng or not real_all:
        print("错误：数据为空", flush=True)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 即梦：train/val 随机拆（同族，可接受）
    aj_tr, aj_val = split(ai_jimeng, VAL_RATIO, rng)
    # ai_cross（旧跨生成器）：整体 held-out，绝不进训练（reviewer P1：与 ai_cross_native 零重叠，
    # 保留为独立验证信号，避免自我虚高；实测纳入训练反而稀释对 ai_cross_native 难例的判别，已回退）
    ac_tr, ac_val = [], list(ai_cross)
    # v6 新增：Flux 跨生成器图（写实合成脸，来自 SFHQ-T2I MIT）—— 进训练，提供真正跨生成器信号
    cg_tr, cg_val = split(ai_crossgen, VAL_RATIO, rng) if ai_crossgen else ([], [])
    # 真实类：按来源组拆
    re_tr, re_val = group_split(real_all, VAL_RATIO, rng)

    ai_tr = aj_tr + ac_tr + cg_tr
    ai_val = aj_val + ac_val + cg_val
    train_files = ai_tr + re_tr
    train_labels = [1] * len(ai_tr) + [0] * len(re_tr)
    val_files = ai_val + re_val
    val_labels = [1] * len(ai_val) + [0] * len(re_val)
    print(f"训练 {len(train_files)}（AI {len(ai_tr)} / REAL {len(re_tr)}，"
          f"其中跨生成器新 {len(cg_tr)}）  验证 {len(val_files)}"
          f"（AI {len(ai_val)} / REAL {len(re_val)}）", flush=True)

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

    sm = torch.nn.Softmax(dim=1)

    def recall_on(folder, label=1, thr=0.5):
        fs = _img_files(REPO / "data" / folder)
        if not fs:
            return None, 0, 0
        dl = DataLoader(AigenDataset(fs, [label] * len(fs), infer_tfm),
                        batch_size=max(len(fs), 1), shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for x, _ in dl:
                preds += [p[1] for p in sm(model(x)).tolist()]
        hit = sum(1 for p in preds if p > thr)
        return hit / len(fs), hit, len(fs)

    cross_recall, cross_hit, cross_total = recall_on("ai_cross_native")
    newgen_recall, newgen_hit, newgen_total = recall_on("ai_mj_sd_flux_heldout")

    (OUT_DIR / "config.json").write_text(json.dumps({
        "framework": "timm", "arch": ARCH, "num_classes": 2,
        "classes": ["real", "ai"], "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
        "backbone_source": "timm/mobilenetv3_large_100.ra_in1k (ImageNet-1k 预训练权重, "
                            "MIT/Apache-2.0, 仅初始化, 非训练数据)",
        "version": "v6",
        "augmentation": "beauty_aug.get_train_transform（域增强，无外部数据）",
        "crossgen_source": "data/ai_mj_sd_flux（SFHQ-T2I MIT / 本人订阅生成，合成无真人，"
                           "经 collect_crossgen.py 来源登记校验后入训练）",
        "trained_on": f"data/ai({len(ai_jimeng)}) + data/ai_cross({len(ai_cross)},仅held-out) "
                      f"+ 跨生成器新({len(ai_crossgen)}) + data/real_xhs({len(real_xhs)}) "
                      f"+ 原有({len(real_old)}) + 伪标注({len(real_extra)}) "
                      f"+ 真图批次({len(real_batch2)})",
        "zero_shot_test_sets": ["data/ai_cross_native(始终零样本)",
                                "data/ai_mj_sd_flux_heldout(每生成器切出)"],
        "class_weight": {"real": round(w_real, 4), "ai": round(w_ai, 4)},
        "best_epoch": best_epoch, "val_acc": round(val_acc, 4),
        "cross_generator_recall_ai_cross_native": round(cross_recall, 4) if cross_recall else None,
        "cross_generator_recall_newgen_heldout": round(newgen_recall, 4) if newgen_recall else None,
        "compliance": "跨生成器图均来自 MIT 合成数据集(SFHQ-T2I)或本人订阅生成，"
                      "经来源登记校验；未使用 GenImage/COCO/ImageNet(训练集)/"
                      "任何无来源第三方数据集；ImageNet 仅作骨干初始化权重。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    res = {"model": "beautyproof_aigen_v6", "arch": ARCH,
           "split": {"train": {"ai": len(ai_tr), "real": len(re_tr), "total": len(train_files),
                               "crossgen_new": len(cg_tr)},
                     "val": {"ai": len(ai_val), "real": len(re_val), "total": len(val_files)}},
           "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 4),
           "val_confusion": cm, "val_accuracy": round(val_acc, 4),
           "cross_generator": {
               "ai_cross_native": {"total": cross_total, "hit": cross_hit,
                                   "recall": round(cross_recall, 4) if cross_recall else None},
               "ai_mj_sd_flux_heldout": {"total": newgen_total, "hit": newgen_hit,
                                         "recall": round(newgen_recall, 4) if newgen_recall else None}},
           "history": history, "elapsed_sec": round(time.time() - t0, 1)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "aigen_finetune_v6.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}  acc={val_acc:.3f}", flush=True)
    print(f"=== 跨生成器零样本召回 ===", flush=True)
    print(f"  ai_cross_native:      {cross_hit}/{cross_total} = {cross_recall:.3f}"
          if cross_recall is not None else "  ai_cross_native: 无样本", flush=True)
    print(f"  ai_mj_sd_flux_heldout:{newgen_hit}/{newgen_total} = {newgen_recall:.3f}"
          if newgen_recall is not None else "  ai_mj_sd_flux_heldout: 无样本", flush=True)
    print(f"已保存 {OUT_DIR}/model.pt  与 results/aigen_finetune_v6.json", flush=True)
    print(f"耗时 {res['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

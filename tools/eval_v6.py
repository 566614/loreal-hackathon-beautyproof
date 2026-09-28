# -*- coding: utf-8 -*-
"""加载已训练的 v6 model.pt，复现与 train_aigen_v6.py 完全一致的拆分，输出指标与 JSON。
避免重训（9 分钟）即可拿到 ai_cross_native / ai_mj_sd_flux_heldout 零样本召回与验证集混淆。
"""
import json
import random
import sys
import time
from pathlib import Path

from train_aigen_v3 import (  # noqa: F401
    _img_files, split, AigenDataset, make_model,
    IMG_SIZE, MEAN, STD, ARCH, SEED, VAL_RATIO, EXTS,
)
from beauty_aug import get_infer_transform

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "models" / "beautyproof_aigen_v6"


def build_pairs():
    d = REPO / "data"
    ai_jimeng = _img_files(d / "ai")
    ai_cross = _img_files(d / "ai_cross")
    real_xhs = _img_files(d / "real_xhs")
    real_old = (_img_files(d / "real") + _img_files(d / "clean") + _img_files(d / "tampered"))
    real_extra = _img_files(d / "real_extra")
    real_batch2 = _img_files(d / "real_xhs_batch2")
    ai_crossgen = sorted(p for p in (d / "ai_mj_sd_flux").rglob("*") if p.suffix.lower() in EXTS)
    return (ai_jimeng, ai_cross, real_xhs, real_old, real_extra, real_batch2, ai_crossgen)


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    rng = random.Random(SEED)
    t0 = time.time()
    (ai_jimeng, ai_cross, real_xhs, real_old, real_extra,
     real_batch2, ai_crossgen) = build_pairs()
    real_all = real_xhs + real_old + real_extra + real_batch2
    print(f"数据：即梦={len(ai_jimeng)} ai_cross={len(ai_cross)} 跨生成器新={len(ai_crossgen)} REAL={len(real_all)}", flush=True)

    aj_tr, aj_val = split(ai_jimeng, VAL_RATIO, rng)
    ac_tr, ac_val = list(ai_cross), []
    cg_tr, cg_val = split(ai_crossgen, VAL_RATIO, rng) if ai_crossgen else ([], [])
    re_tr, re_val = split(real_all, VAL_RATIO, rng)

    ai_tr = aj_tr + ac_tr + cg_tr
    ai_val = aj_val + ac_val + cg_val
    train_files = ai_tr + re_tr
    train_labels = [1] * len(ai_tr) + [0] * len(re_tr)
    val_files = ai_val + re_val
    val_labels = [1] * len(ai_val) + [0] * len(re_val)
    print(f"训练 {len(train_files)}（AI {len(ai_tr)} / REAL {len(re_tr)}，跨生成器新 {len(cg_tr)}）"
          f"  验证 {len(val_files)}（AI {len(ai_val)} / REAL {len(re_val)}）", flush=True)

    infer_tfm = get_infer_transform()
    val_dl = DataLoader(AigenDataset(val_files, val_labels, infer_tfm),
                        batch_size=max(len(val_files), 1), shuffle=False, num_workers=0)
    model = make_model()
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
        fs = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS) if Path(folder).is_dir() else []
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

    cross_recall, cross_hit, cross_total = recall_on(REPO / "data" / "ai_cross_native")
    newgen_recall, newgen_hit, newgen_total = recall_on(REPO / "data" / "ai_mj_sd_flux_heldout")

    res = {"model": "beautyproof_aigen_v6", "arch": ARCH,
           "split": {"train": {"ai": len(ai_tr), "real": len(re_tr), "total": len(train_files),
                               "crossgen_new": len(cg_tr)},
                     "val": {"ai": len(ai_val), "real": len(re_val), "total": len(val_files)}},
           "val_confusion": cm, "val_accuracy": round(val_acc, 4),
           "cross_generator": {
               "ai_cross_native": {"total": cross_total, "hit": cross_hit,
                                   "recall": round(cross_recall, 4) if cross_recall else None},
               "ai_mj_sd_flux_heldout": {"total": newgen_total, "hit": newgen_hit,
                                         "recall": round(newgen_recall, 4) if newgen_recall else None}},
           "elapsed_sec": round(time.time() - t0, 1)}
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "aigen_finetune_v6.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 验证集混淆 (AI=正类) ===", flush=True)
    print(f"  TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}  acc={val_acc:.3f}", flush=True)
    print(f"=== 跨生成器零样本召回 ===", flush=True)
    print(f"  ai_cross_native:      {cross_hit}/{cross_total} = {cross_recall:.3f}" if cross_recall is not None else "  ai_cross_native: 无样本", flush=True)
    print(f"  ai_mj_sd_flux_heldout:{newgen_hit}/{newgen_total} = {newgen_recall:.3f}" if newgen_recall is not None else "  ai_mj_sd_flux_heldout: 无样本", flush=True)
    print(f"耗时 {res['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

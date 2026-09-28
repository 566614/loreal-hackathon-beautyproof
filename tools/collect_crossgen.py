# -*- coding: utf-8 -*-
"""跨生成器素材收编：把「有来源标注」的 MJ/SD/Flux 美妆图安全并入训练/held-out。

设计红线（项目铁律，不可绕过）：
- 任何缺来源标注（子目录无 PROVENANCE.md）的图一律拒绝，绝不进训练。
- 默认信任登记表里的「是否合成(无真人)」声明；若某图疑似真实人物，人工复核后再放。
- 收编时自动切出一小部分（--heldout-per-generator）作为「本生成器零样本」验证集，
  永不进训练，用于检验 v6 对该生成器的真实召回（不虚高）。

用法：
    python tools/collect_crossgen.py \
        --staging data/ai_cross_native_incoming \
        --train-out data/ai_mj_sd_flux \
        --heldout-out data/ai_mj_sd_flux_heldout \
        --heldout-per-generator 4
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
GENERATORS = ("midjourney", "stable_diffusion", "flux")


def validate_provenance(staging: Path):
    """每个含图的生成器子目录必须有 PROVENANCE.md，否则拒绝。"""
    missing = []
    for g in GENERATORS:
        d = staging / g
        has_imgs = d.exists() and any(p.suffix.lower() in EXTS for p in d.iterdir())
        if has_imgs and not (d / "PROVENANCE.md").exists():
            missing.append(str(d))
    return missing


def main():
    ap = argparse.ArgumentParser(description="跨生成器素材安全收编（带来源登记校验）")
    ap.add_argument("--staging", default="data/ai_cross_native_incoming")
    ap.add_argument("--train-out", default="data/ai_mj_sd_flux")
    ap.add_argument("--heldout-out", default="data/ai_mj_sd_flux_heldout")
    ap.add_argument("--heldout-per-generator", type=int, default=4,
                    help="每生成器切出 N 张作零样本验证（永不训练）")
    args = ap.parse_args()

    staging = REPO / args.staging
    if not staging.exists():
        print(f"❌ staging 不存在：{staging}", flush=True)
        return 1

    miss = validate_provenance(staging)
    if miss:
        print("❌ 以下生成器子目录缺 PROVENANCE.md，拒绝收编（红线：无来源标注不进训练）：")
        for m in miss:
            print("   ", m)
        return 1

    ledger = {"generated_by": "tools/collect_crossgen.py", "note":
              "仅收编带 PROVENANCE.md 且声明为合成(无真人)的图", "generators": {}}
    total_train = total_held = 0
    for g in GENERATORS:
        sd = staging / g
        imgs = sorted(p for p in sd.iterdir() if p.suffix.lower() in EXTS)
        if not imgs:
            continue
        held = imgs[:args.heldout_per_generator]
        train = imgs[args.heldout_per_generator:]
        tg = REPO / args.train_out / g
        hg = REPO / args.heldout_out / g
        tg.mkdir(parents=True, exist_ok=True)
        hg.mkdir(parents=True, exist_ok=True)
        for p in train:
            shutil.copy(p, tg / p.name)
        for p in held:
            shutil.copy(p, hg / p.name)
        if (sd / "PROVENANCE.md").exists():
            shutil.copy(sd / "PROVENANCE.md", tg / "PROVENANCE.md")
            shutil.copy(sd / "PROVENANCE.md", hg / "PROVENANCE.md")
        # 把 PROVENANCE 关键字段抽进机器可读 ledger（去掉模板空行）
        prov = (sd / "PROVENANCE.md").read_text(encoding="utf-8")
        rows = [ln for ln in prov.splitlines()
                if ln.strip().startswith("|") and "文件名" not in ln and "---" not in ln]
        ledger["generators"][g] = {
            "train": len(train), "heldout": len(held),
            "source_rows": rows,
        }
        total_train += len(train)
        total_held += len(held)
        print(f"  {g}: train={len(train)}  heldout={len(held)}", flush=True)

    (REPO / "data" / "ai_mj_sd_flux_PROVENANCE.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已收编：训练 {total_train} 张 / 零样本 {total_held} 张")
    print(f"已写 data/ai_mj_sd_flux_PROVENANCE.json", flush=True)
    print("下一步：python tools/train_aigen_v6.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""准备 v2 真实类训练数据：合并两批小红书美妆图，去重、转 jpg、质检打分。

两批来源：
  A) MediaCrawler数据.zip  -> outputs/_external_xhs/ (56 张 webp，关键词"口红")
  B) workbuddy在小红书上爬取的图片.zip -> 35 张唯一图（含手背试色/粉底液/眼影盘）
注意 B 里「01~07_*.png」是 A 的 56 张中人工筛出的子集，会与 A 重复 -> 靠内容 hash 去重。

输出：data/real_xhs/  （.jpg；.gitignore 已排除 data 下所有图片，不会入库）
"""
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from PIL import Image  # noqa: E402

NEW_ZIP = Path(r"C:\Users\Lenovo\Documents\xwechat_files"
               r"\wxid_fwgf45apbp8422_3543\msg\file\2026-09"
               r"\workbuddy在小红书上爬取的图片.zip")
OLD_DIR = REPO / "outputs" / "_external_xhs"
OUT = REPO / "data" / "real_xhs"


def content_hash(img: Image.Image):
    """缩略图 md5：跨格式也能判重复。"""
    im = img.copy()
    im.thumbnail((64, 64))
    return hashlib.md5(im.tobytes()).hexdigest()


def collect_from_zip(zp: Path):
    z = zipfile.ZipFile(zp)
    out = []
    for n in z.namelist():
        if n.endswith("/"):
            continue
        try:
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        except Exception:
            continue
        out.append((n.split("/")[-1], im))
    return out


def collect_from_dir(d: Path):
    out = []
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in (".webp", ".jpg", ".jpeg", ".png"):
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        out.append((p.name, im))
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    items = []
    src_tag = {}
    for name, im in collect_from_zip(NEW_ZIP):
        items.append((f"new_{name}", im)); src_tag[f"new_{name}"] = "B_新包"
    for name, im in collect_from_dir(OLD_DIR):
        items.append((f"old_{name}", im)); src_tag[f"old_{name}"] = "A_口红包"

    kept, seen, dup = [], {}, 0
    stats = {"too_small": 0, "dup": 0, "kept": 0}
    for name, im in items:
        if min(im.size) < 200:
            stats["too_small"] += 1
            continue
        h = content_hash(im)
        if h in seen:
            dup += 1
            stats["dup"] += 1
            continue
        seen[h] = name
        # 统一命名，保留来源标记，方便追溯
        stem = Path(name).stem.replace(" ", "_")
        out_name = f"real_{len(kept):03d}_{stem[:40]}.jpg"
        im.save(OUT / out_name, quality=92)
        kept.append({"file": out_name, "src": src_tag[name], "orig": name,
                     "w": im.size[0], "h": im.size[1]})
        stats["kept"] += 1

    print(f"候选 {len(items)}  去重掉 {stats['dup']}  太小丢弃 {stats['too_small']}  "
          f"最终保留 {stats['kept']}")
    from collections import Counter
    print("来源分布:", dict(Counter(k["src"] for k in kept)))

    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "real_xhs_manifest.json").write_text(
        json.dumps({"stats": stats, "files": kept}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("已输出 data/real_xhs/，清单 results/real_xhs_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

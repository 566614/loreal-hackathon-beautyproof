"""Copy supplied AI images into a fixed split after source-group review."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil


def validation_groups(rows, fraction=.2, seed=20261118):
    grouped = {}
    for row in rows:
        if row["group"] != "unresolved-new":
            grouped.setdefault(row["group"], []).append(row)
    if len(grouped) < 2:
        raise ValueError("至少需要两个可区分的同源组才能划分训练与验证。")
    batch_totals = Counter(row["batch"] for row in rows)
    generator = random.Random(seed)
    groups = sorted(grouped)
    best = None
    chosen = None
    for _ in range(128):
        generator.shuffle(groups)
        counts = Counter()
        total = 0
        for index, group in enumerate(groups[:-1], 1):
            counts.update(row["batch"] for row in grouped[group])
            total += len(grouped[group])
            loss = abs(total-len(rows)*fraction)
            loss += .5*sum(abs(counts[batch]-count*fraction)
                           for batch, count in batch_totals.items())
            if best is None or loss < best:
                best = loss
                chosen = set(groups[:index])
    return chosen


def prepare_split(reviewed_path, real_groups_path, real_root, output):
    rows = json.loads(reviewed_path.read_text(encoding="utf-8-sig"))
    groups = validation_groups(rows)
    manifest = []
    for index, row in enumerate(rows, 1):
        source = Path(row["path"])
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != row["sha256"]:
            raise ValueError(f"清点后文件内容发生变化，请重新核对 {source}")
        split = "validation" if row["group"] in groups else "train"
        filename = f"ai_{index:04d}_{source.stem[:16]}{source.suffix.lower()}"
        manifest.append({"file": filename, "source_path": str(source), "split": split,
                         "label": 1, "group": row["group"], "batch": row["batch"],
                         "sha256": actual_hash, "pixel_sha256": row["pixel_sha256"],
                         "reference_file": row.get("reference_file") or ""})
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf8")
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise ValueError("输出目录已有不同划分，请使用新的输出目录以保留验证集。")
    for row in manifest:
        target = output/row["split"]/"ai"/row["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"目标文件内容与登记不一致 {target}")
        else:
            shutil.copy2(row["source_path"], target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                raise OSError(f"复制校验失败 {target}")
    manifest_path.write_bytes(encoded)
    with (output/"manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    real_sources = json.loads(real_groups_path.read_text(encoding="utf8"))["files"]
    controls = [{"file": row["file"], "source_path": str(real_root/row["file"]),
                 "label": 0, "group": row["group"],
                 "split": "validation" if row["group"] in groups else "train"}
                for row in real_sources]
    with (output/"real-controls.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(controls[0]))
        writer.writeheader()
        writer.writerows(controls)
    summary = {"seed": 20261118, "requested_validation_fraction": .2,
               "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
               "ai_images": dict(Counter(row["split"] for row in manifest)),
               "real_controls": dict(Counter(row["split"] for row in controls)),
               "validation_groups": sorted(groups),
               "group_overlap": sorted({r["group"] for r in manifest if r["split"]=="train"} & groups),
               "source": "user-confirmed AI images, with camera controls from supplied originals",
               "actual_photographed_damage_controls": 0}
    (output/"split-lock.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_groups", type=Path)
    parser.add_argument("--real-groups", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_split(args.reviewed_groups, args.real_groups, args.real_root, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

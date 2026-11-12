"""Make uniformly cropped image copies for manual label review."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from beauty_benchmark import local_file, sha256, validate


def prepare(manifest, destination, bottom_fraction):
    if not 0 <= bottom_fraction < .5:
        raise ValueError("底部裁剪比例须在 0 到 0.5 之间")
    rows = validate(manifest)
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("输出目录已存在，请选择新目录")
    destination.mkdir(parents=True)
    images, mask_index = {}, 0
    copied = []
    for row in rows:
        result = dict(row)
        if row.get("image"):
            digest = row["sha256"]
            if digest not in images:
                source = local_file(manifest.parent, row["image"])
                path = destination / f"image-{len(images) + 1:05d}.png"
                with Image.open(source) as raw:
                    orientation = raw.getexif().get(274, 1)
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    height = round(image.height * (1 - bottom_fraction))
                    image.crop((0, 0, image.width, height)).save(path)
                images[digest] = (path.name, sha256(path), orientation)
            result["image"], result["sha256"], _ = images[digest]
            result["visible_ai_label_reviewed"] = False
            result.pop("visible_ai_label_present", None)
        if row.get("mask"):
            mask_index += 1
            path = destination / f"mask-{mask_index:05d}.png"
            with Image.open(local_file(manifest.parent, row["mask"])) as raw:
                mask = raw.convert("L")
                orientation = images[row["sha256"]][2]
                method = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                          4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                          6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                          8: Image.Transpose.ROTATE_90}.get(orientation)
                if method is not None:
                    mask = mask.transpose(method)
                height = round(mask.height * (1 - bottom_fraction))
                mask.crop((0, 0, mask.width, height)).save(path)
            result["mask"], result["mask_sha256"] = path.name, sha256(path)
        copied.append(result)
    for row in copied:
        if row.get("original_sha256"):
            row["original_sha256"] = images[row["original_sha256"]][1]
    (destination / "manifest.json").write_text(json.dumps(copied, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "preparation.json").write_text(json.dumps({"source_manifest_sha256": sha256(manifest),
        "bottom_fraction": bottom_fraction, "format": "RGB PNG", "review_required": True}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--bottom-fraction", type=float, required=True)
    args = parser.parse_args()
    prepare(args.manifest.resolve(), args.destination, args.bottom_fraction)


if __name__ == "__main__":
    main()

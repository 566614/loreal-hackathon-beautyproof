"""Extract frozen DDA features and fit a small classification head."""
import argparse
from collections import Counter
from pathlib import Path

from beauty_benchmark import local_file, sha256, validate
from forensic_models import _root, dda_feature, file_hash, load_dda


def extract(manifest, output):
    rows = [row for row in validate(manifest) if row["split"] in ("train", "calibration")
            and row["kind"] in ("real", "generated")]
    if not rows:
        raise ValueError("清单没有训练或校准图片")
    unchecked = [row["id"] for row in rows if row.get("visible_ai_label_reviewed") is not True
                 or row.get("visible_ai_label_present") is not False]
    if unchecked:
        raise ValueError("训练和校准图片须复核并去除可见 AI 标识：" +
                         ", ".join(unchecked[:5]))
    model, transform, device = load_dda()
    import torch
    features = []
    for row in rows:
        features.append(dda_feature(model, transform, device, local_file(manifest.parent, row["image"]))
                        .squeeze(0).cpu())
    torch.save({"features": torch.stack(features), "rows": rows,
                "manifest_sha256": sha256(manifest),
                "backbone_sha256": file_hash(_root() / "DDA_ckpt.pth"),
                "preprocessing": "dda-official-center-crop336-clip"}, output)


def train(cache, manifest, output):
    import torch

    validate(manifest)
    data = torch.load(cache, map_location="cpu", weights_only=True)
    if data["manifest_sha256"] != sha256(manifest):
        raise ValueError("特征缓存与当前清单不一致")
    if data["backbone_sha256"] != file_hash(_root() / "DDA_ckpt.pth"):
        raise ValueError("特征缓存与 DDA 权重不一致")
    rows = data["rows"]
    if any(row.get("visible_ai_label_reviewed") is not True or
           row.get("visible_ai_label_present") is not False for row in rows):
        raise ValueError("特征缓存含未复核可见 AI 标识的图片，请重新准备数据")
    indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    labels = [int(rows[i]["kind"] == "generated") for i in indices]
    if set(labels) != {0, 1}:
        raise ValueError("训练集需要实拍和整图生成两个类别")
    counts = Counter(rows[i]["group"] for i in indices)
    x = data["features"][indices].float()
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    weights = torch.tensor([1 / counts[rows[i]["group"]] for i in indices]).unsqueeze(1)
    torch.manual_seed(20261112)
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(head(x), y, reduction="none") * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
    torch.save({"head": head.state_dict(), "backbone_sha256": data["backbone_sha256"],
                "preprocessing": data["preprocessing"], "manifest_sha256": data["manifest_sha256"],
                "seed": 20261112, "epochs": 100}, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "train"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if args.action == "extract":
        extract(manifest, args.cache)
    else:
        if not args.out:
            parser.error("train requires --out")
        train(args.cache, manifest, args.out)


if __name__ == "__main__":
    main()

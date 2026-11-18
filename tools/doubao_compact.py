"""Run the experimental single-image Doubao detector with a compact CPU model."""
import argparse
import json
from pathlib import Path
import zlib

import cv2
import numpy as np
from PIL import Image

from doubao_damage import decode, extract, prepare

SCHEMA = "beautyproof/doubao-resampling@1"
FAMILIES = ("texture_conditioned", "rgb", "chroma", "codec_response")
DIMENSIONS = 716


def resampling_features(image):
    rgb = np.asarray(image)
    size = max(rgb.shape[:2])
    target = (round(rgb.shape[1]*312/size), round(rgb.shape[0]*312/size))
    interpolation = cv2.INTER_AREA if size >= 312 else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, target, interpolation=interpolation).astype(np.float32)/255
    red, green, blue = rgb.transpose(2, 0, 1)
    luminance = red*.299 + green*.587 + blue*.114
    channels = np.stack((luminance, red-luminance, blue-luminance), axis=-1)
    height, width = channels.shape[:2]
    positions = np.asarray([(y, x) for y in range(0, height-24, 11)
                            for x in range(0, width-24, 11)])
    if len(positions) < 30:
        square = cv2.resize(rgb, (304, 304), interpolation=cv2.INTER_AREA)
        return resampling_features(Image.fromarray(np.round(square*255).astype(np.uint8)))
    coordinates = (positions[:, 0, None, None]+np.arange(24)[None, :, None],
                   positions[:, 1, None, None]+np.arange(24)[None, None, :])
    blurred = cv2.GaussianBlur(channels, (0, 0), 1.3)
    texture = np.log1p(300*blurred[..., 0][coordinates].std((1, 2)))
    bands = np.array_split(np.argsort(texture, kind="stable"), 3)

    def describe(patches):
        spread = patches.std((1, 2))
        amplitude = np.log(np.abs(patches).mean((1, 2))+.001)
        centered = patches-patches.mean((1, 2), keepdims=True)
        correlation = (centered[..., 0, None]*centered[..., 1:]).mean((1, 2))
        correlation /= np.maximum(spread[:, 0, None]*spread[:, 1:], .000004)
        tails = np.mean(np.abs(patches) > 2.8*np.maximum(spread[:, None, None, :], .002),
                        axis=(1, 2))
        return np.c_[amplitude, amplitude[:, 1:]-amplitude[:, :1], correlation, tails]

    summaries = [describe((channels-blurred)[coordinates])]
    differences = []
    for factor in (2, 3, 4):
        small = cv2.resize(channels, (max(10, width//factor), max(10, height//factor)),
                           interpolation=cv2.INTER_AREA)
        enlarged = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        residual = (channels-enlarged)[coordinates]
        summary = describe(residual)
        summaries.append(summary)
        differences.append(summary-summaries[0])
        planar = (enlarged-blurred)[coordinates][..., 0]
        normalizer = np.sqrt((residual[..., 0]**2).mean((1, 2)) *
                             (planar**2).mean((1, 2))+.000001)
        differences.append(((residual[..., 0]*planar).mean((1, 2))/normalizer)[:, None])
    combined = np.concatenate([*summaries, *differences], axis=1)
    return np.concatenate([np.quantile(combined[band], [.25, .5, .85], axis=0).ravel()
                           for band in bands]).astype(np.float32)


def image_features(image):
    canonical = prepare(image, short_side=288)
    if canonical is None:
        raise ValueError("图片短边需至少 256 像素。")
    whole = extract(canonical, extra=FAMILIES)[0]
    return np.concatenate((resampling_features(image),
                           np.concatenate([whole[key] for key in FAMILIES])))


def load_model(path):
    try:
        payload = zlib.decompress(Path(path).read_bytes())
    except zlib.error as exc:
        raise ValueError("小模型文件压缩数据损坏。") from exc
    header_size = int.from_bytes(payload[:4], "little")
    header = json.loads(payload[4:4+header_size])
    if (header.get("schema") != SCHEMA or header.get("bits") != 4 or
            header.get("views") != 4 or header.get("offset") != 7 or
            header.get("codec") not in ("row-bytes", "row-bits", "delta-bytes", "delta-bits")):
        raise ValueError("小模型格式或特征版本不匹配。")
    rows, dimensions = header["shape"]
    if dimensions != DIMENSIONS or rows <= 0 or rows % 4:
        raise ValueError("小模型参数维度不匹配。")
    threshold = header["threshold"]
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("小模型缺少校准阈值。")
    position = 4+header_size
    mean = np.frombuffer(payload, dtype="<f2", count=dimensions, offset=position).astype(np.float32)
    spread = np.frombuffer(payload, dtype="<f2", count=dimensions,
                           offset=position+2*dimensions).astype(np.float32)
    position += 4*dimensions
    coefficients = np.frombuffer(payload, dtype="<f2", count=rows, offset=position).astype(np.float32)
    position += 2*rows
    packed = np.frombuffer(payload, dtype=np.uint8, offset=position)
    if header["codec"].endswith("bits"):
        bits = np.unpackbits(packed)[:rows*dimensions*4].reshape(-1, 4)
        codes = (bits << np.arange(4, dtype=np.uint8)).sum(1).astype(np.uint8)
    else:
        codes = packed
    codes = codes.reshape(-1, 4, dimensions).copy()
    if header["codec"].startswith("delta"):
        codes[:, 1:] ^= codes[:, :1]
    scale = header["scale"]
    if (not np.isfinite(scale) or scale <= 0 or np.any(spread <= 0) or
            not all(np.isfinite(array).all() for array in (mean, spread, coefficients)) or
            codes.max() > 14):
        raise ValueError("小模型参数无效。")
    centers = (codes.reshape(rows, dimensions).astype(np.float32)-7)*scale
    return {"mean": mean, "spread": spread, "centers": centers,
            "coefficients": coefficients, "threshold": threshold}


def score_features(model, features):
    query = ((features-model["mean"])/model["spread"]).clip(-5, 5)
    centers = model["centers"]
    squared = np.maximum((query*query).sum(-1)[..., None] +
                         (centers*centers).sum(-1)-2*query@centers.T, 0)
    return np.exp(-.62*squared/DIMENSIONS)@model["coefficients"]


def analyze_image(data, model):
    image, marker = decode(data)
    score = float(score_features(model, image_features(image)))
    return {"model": SCHEMA, "experimental": True, "device": "cpu",
            "score": score, "threshold": model["threshold"],
            "status": "ai_suspected" if score >= model["threshold"] else "below_threshold",
            "source_marker": marker}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    try:
        model = load_model(args.model)
        result = analyze_image(args.input.read_bytes(), model)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, str(exc)+"\n")
    print(json.dumps({"file": args.input.name, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()

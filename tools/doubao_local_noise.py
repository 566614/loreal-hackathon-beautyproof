"""Score experimental local channel and noise evidence using a small NumPy model."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from doubao_damage import MIN_SHORT_SIDE, decode

SCHEMA = "beautyproof/doubao-local-noise@1"
SIDE = 288
GRID = 16
DIMENSIONS = 78


def image_features(image):
    if min(image.size) < MIN_SHORT_SIDE:
        raise ValueError("图片短边需至少 256 像素。")
    rgb = cv2.resize(np.asarray(image), (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32)/255
    red, green, blue = rgb.transpose(2, 0, 1)
    luminance = red*.299+green*.587+blue*.114
    channels = (luminance, red-luminance, blue-luminance)
    centers = np.arange(SIDE//GRID//2, SIDE, SIDE//GRID)

    def average(values, span=25):
        pooled = cv2.boxFilter(values.astype(np.float32), -1, (span, span))
        return pooled[np.ix_(centers, centers)].ravel()

    def amplitude(values, span=25):
        return np.log(average(np.abs(values), span)+.0007)

    horizontal = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    vertical = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    gradient = np.log(average(np.hypot(horizontal, vertical))+.0008)
    light = average(luminance)
    texture = np.maximum(average(luminance**2)-light**2, 0)**.5
    descriptors = [gradient]
    for channel in channels:
        residuals = [channel-cv2.GaussianBlur(channel, (0, 0), sigma)
                     for sigma in (.7, 1.35, 2.4)]
        for residual in residuals:
            strength = amplitude(residual)
            descriptors.extend((strength, np.log(average(residual*residual)+.000001)-2*strength))
        descriptors.extend((amplitude(residuals[0])-amplitude(residuals[2]),
                            amplitude(cv2.Laplacian(channel, cv2.CV_32F), 33)-amplitude(residuals[1], 33)))
        for factor in (2, 3):
            reduced = cv2.resize(channel, (SIDE//factor, SIDE//factor), interpolation=cv2.INTER_AREA)
            residual = channel-cv2.resize(reduced, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
            descriptors.extend((amplitude(residual)-amplitude(residuals[0]),
                                amplitude(residual, 17)-gradient))
    chroma = [channel-cv2.GaussianBlur(channel, (0, 0), 1.) for channel in channels[1:]]
    luma = luminance-cv2.GaussianBlur(luminance, (0, 0), 1.)
    for channel in chroma:
        descriptors.append(average(channel*luma)/np.sqrt(
            (average(channel*channel)+.000001)*(average(luma*luma)+.000001)))
    raw = np.stack(descriptors, axis=1)
    light_band = np.digitize(light, np.quantile(light, [.33, .67]))
    texture_band = np.digitize(texture, np.quantile(texture, [.33, .67]))
    bands = 3*light_band+texture_band
    relative = raw.copy()
    for band in range(9):
        selected = bands == band
        reference = raw[selected] if selected.sum() >= 6 else raw
        relative[selected] -= np.median(reference, axis=0)
    return np.concatenate((raw, relative), axis=-1).astype(np.float32)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        model = {key: archive[key].copy() for key in archive.files}
    if str(model.get("schema")) != SCHEMA:
        raise ValueError("局部模型特征版本不匹配。")
    shapes = {"mean": (DIMENSIONS,), "spread": (DIMENSIONS,),
              "layers.0.weight": (32, DIMENSIONS), "layers.0.bias": (32,),
              "layers.3.weight": (16, 32), "layers.3.bias": (16,),
              "layers.5.weight": (1, 16), "layers.5.bias": (1,)}
    for key, shape in shapes.items():
        if model[key].shape != shape or not np.isfinite(model[key]).all():
            raise ValueError("局部模型参数无效。")
    threshold = model.get("threshold")
    if threshold is None or threshold.shape != () or not np.isfinite(threshold):
        raise ValueError("局部模型缺少校准阈值。")
    if np.any(model["spread"] <= 0):
        raise ValueError("局部模型归一化参数无效。")
    return model


def score_features(model, features):
    hidden = ((features-model["mean"])/model["spread"]).clip(-5, 5)
    for index in (0, 3):
        hidden = hidden@model[f"layers.{index}.weight"].T+model[f"layers.{index}.bias"]
        hidden = hidden/(1+np.exp(np.clip(-hidden, -80, 80)))
    scores = (hidden@model["layers.5.weight"].T+model["layers.5.bias"])[..., 0]
    largest = np.partition(scores, -12, axis=-1)[..., -12:]
    return .8*largest.mean(-1)+.2*scores.mean(-1)


def analyze_image(data, model):
    image, marker = decode(data)
    score = float(score_features(model, image_features(image)))
    threshold = float(model["threshold"])
    return {"model": SCHEMA, "experimental": True, "device": "cpu",
            "score": score, "threshold": threshold,
            "status": "ai_suspected" if score >= threshold else "below_threshold",
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

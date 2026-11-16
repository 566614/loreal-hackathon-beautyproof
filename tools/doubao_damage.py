"""CPU-only forensic cues for the BeautyProof damage review experiment."""
import argparse
import base64
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "doubao_damage" / "model.json"
SCHEMA = "beautyproof/doubao-damage@1"
MAX_PIXELS = 20_000_000
MIN_SHORT_SIDE = 256
TILE = 24
STRIDE = 12

FAMILY_LABELS = {
    "rgb": "高频色彩通道关系",
    "gaussian": "平坦区域的高斯残差",
    "median": "中值滤波残差",
    "laplacian": "拉普拉斯高频残差",
    "directional": "方向高通残差",
    "spectrum": "频谱与周期性",
}
FREQ = np.fft.fftfreq(TILE)
RADIUS = np.hypot(FREQ[:, None], FREQ)
BANDS = [(RADIUS >= low) & (RADIUS < high) for low, high in
         ((0.08, 0.18), (0.18, 0.30), (0.30, 0.43), (0.43, 0.55))]
WINDOW = np.outer(np.hanning(TILE), np.hanning(TILE)).astype(np.float32)


def validate_roi(roi):
    """Coordinates refer to the displayed, EXIF-oriented image."""
    if roi is None:
        return None
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in roi)
            or not (roi[0] < roi[2] and roi[1] < roi[3])):
        raise ValueError("框选坐标需为 0 到 1 之间的 [左, 上, 右, 下]。")
    return list(roi)


def _source_marker(opened):
    xmp = opened.info.get("xmp") or opened.info.get("XML:com.adobe.xmp") or ""
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", errors="replace")
    match = re.search(r"<(?:TC260:)?AIGC>(.*?)</(?:TC260:)?AIGC>", xmp, re.S)
    if match:
        try:
            payload = json.loads(match[1].replace("&quot;", '"'))
        except ValueError:
            payload = None
        if isinstance(payload, dict) and str(payload.get("Label")) == "1":
            return "declared_ai"
        return "unverified_marker"
    if any(marker == "APP11" and content.startswith(b"JP") and b"jumb" in content
           for marker, content in getattr(opened, "applist", ())):
        return "unverified_marker"
    return "not_found"


def decode(data):
    """Read only the supplied JPEG or PNG. Preserve the source marker before resizing."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in ("JPEG", "PNG") or getattr(opened, "n_frames", 1) != 1:
                raise ValueError("请选择单张 JPG 或 PNG 图片。")
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("图片不能超过 2000 万像素。")
            marker = _source_marker(opened)
            oriented = ImageOps.exif_transpose(opened)
            # Keep alpha handling independent of browser compositing and omit unsupported pixels.
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                oriented = Image.alpha_composite(background, rgba)
            return oriented.convert("RGB"), marker
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取所选图片，请换一张 JPG 或 PNG。") from exc


def prepare(image, *, quality=85, smaller=False):
    """Equalize sample size and encoder for both labels and every input at inference."""
    if min(image.size) < MIN_SHORT_SIDE:
        return None
    scale = min(MIN_SHORT_SIDE / min(image.size), 768 / max(image.size))
    size = tuple(round(dim * scale) for dim in image.size)
    if min(size) < 96:
        return None
    thumbnail = image.resize(size, Image.Resampling.LANCZOS)
    if smaller:
        reduced_size = tuple(round(dim * 0.75) for dim in size)
        thumbnail = thumbnail.resize(reduced_size, Image.Resampling.LANCZOS)
        thumbnail = thumbnail.resize(size, Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as uniform:
        return np.asarray(uniform).copy()


def _channel_corr(left, right):
    x = left - left.mean(axis=(1, 2), keepdims=True)
    y = right - right.mean(axis=(1, 2), keepdims=True)
    denom = np.sqrt(np.mean(x*x, axis=(1, 2)) * np.mean(y*y, axis=(1, 2)))
    return np.mean(x*y, axis=(1, 2)) / np.maximum(denom, 0.001)


def _residual_stats(residual):
    absolute = np.abs(residual)
    mean = absolute.mean(axis=(1, 2))
    rms = np.sqrt(np.mean(residual**2, axis=(1, 2)))
    return np.log1p(mean), np.log1p(rms), rms / np.maximum(mean, 0.1)


def _summarize(tile_values, flat_indices, indices, surroundings=None):
    values = tile_values[indices]
    result = [np.median(tile_values[flat_indices], axis=0), np.quantile(values, 0.85, axis=0)]
    if surroundings is not None:
        outside = np.median(tile_values[surroundings], axis=0)
        result.append(np.median(values, axis=0) - outside)
    return np.concatenate(result)


def _region_indices(roi, width, height, xx, yy):
    left, top, right, bottom = np.array(roi) * [width, height, width, height]
    if right - left < TILE or bottom - top < TILE:
        return None
    within = (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
    inner = np.flatnonzero(within)
    if len(inner) < 2:
        return None
    margin = TILE + 0.25 * max(right - left, bottom - top)
    nearby = (xx >= left - margin) & (xx <= right + margin) & (yy >= top - margin) & (yy <= bottom + margin)
    surrounding = np.flatnonzero(nearby & ~within)
    # A box around nearly the entire image can legitimately lack a comparison area.
    if len(surrounding) < 2:
        return None
    return inner, surrounding


def _edge_features(gray, roi, preview):
    height, width = gray.shape
    if roi is None:
        bounds = [0, 0, width, height]
    else:
        bounds = np.rint(np.array(roi) * [width, height, width, height]).astype(int)
    left, top, right, bottom = bounds
    region = gray[top:bottom, left:right]
    if min(region.shape) < 4:
        return None, None, None
    blur = cv2.GaussianBlur(region.astype(np.uint8), (3, 3), 0.8)
    sx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    edge = cv2.Canny(blur, 35, 90, L2gradient=True)
    closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, components, _ = cv2.connectedComponentsWithStats(closed)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = closed.copy()
    skeleton = np.zeros_like(edge)
    while remaining.any():
        eroded = cv2.erode(remaining, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        opened = cv2.dilate(eroded, cross)
        skeleton |= cv2.subtract(remaining, opened)
        remaining = eroded
    on = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(on, -1, np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT) - on
    active = on.astype(bool)
    endpoints = int(np.count_nonzero(active & (neighbors == 1)))
    junctions = int(np.count_nonzero(active & (neighbors >= 3)))
    angles = np.arctan2(sy, sx)[edge > 0]
    alignment = abs(np.exp(2j * angles).mean()) if len(angles) else 0.0
    density = float(np.count_nonzero(edge) / edge.size)
    longest = float(components[1:, cv2.CC_STAT_AREA].max() / edge.size) if count > 1 else 0.0
    features = np.array([density, longest, endpoints / edge.size, junctions / edge.size,
                         float(alignment), float(np.mean(np.hypot(sx, sy)))])
    encoded = None
    if preview:
        overlay = np.zeros((*edge.shape, 4), np.uint8)
        overlay[closed > 0] = (0, 175, 255, 210)  # BGRA
        ok, png = cv2.imencode(".png", overlay)
        if ok:
            encoded = base64.b64encode(png).decode("ascii")
    summary = {"edge_density": round(density, 4), "skeleton_endpoints": endpoints,
               "skeleton_junction_pixels": junctions}
    return features, encoded, summary


def extract(image, roi=None, *, preview=False, study_periodicity=False):
    """Extract image and optional region vectors; no metadata or dimensions in vectors."""
    height, width = image.shape[:2]
    rgb = image.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ycrcb = cv2.cvtColor(rgb / 255.0, cv2.COLOR_RGB2YCrCb) * 255.0
    xx, yy = np.meshgrid(np.arange(TILE//2, width - TILE//2 + 1, STRIDE),
                         np.arange(TILE//2, height - TILE//2 + 1, STRIDE))
    xx, yy = xx.ravel(), yy.ravel()
    tile_y = yy[:, None, None] - TILE//2 + np.arange(TILE)[None, :, None]
    tile_x = xx[:, None, None] - TILE//2 + np.arange(TILE)[None, None, :]

    def tiles(array):
        return array[tile_y, tile_x]

    gray_tiles = tiles(gray)
    texture = np.log1p(gray_tiles.std(axis=(1, 2)))
    high_rgb = tiles(rgb - cv2.GaussianBlur(rgb, (3, 3), 0.8))
    sigmas = high_rgb.std(axis=(1, 2))
    rgb_stats = np.column_stack((np.log((sigmas[:, 0]+0.25)/(sigmas[:, 1]+0.25)),
                                 np.log((sigmas[:, 2]+0.25)/(sigmas[:, 1]+0.25)),
                                 _channel_corr(high_rgb[..., 0], high_rgb[..., 1]),
                                 _channel_corr(high_rgb[..., 2], high_rgb[..., 1])))
    gaussian = ycrcb - cv2.GaussianBlur(ycrcb, (5, 5), 1.05)
    median = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    kernel = np.array([1., -2., 1.], dtype=np.float32)
    horizontal = cv2.filter2D(gray, -1, kernel[None, :])
    vertical = cv2.filter2D(gray, -1, kernel[:, None])
    residual = tiles(gaussian[..., 0])
    power = np.abs(np.fft.fft2(residual * WINDOW, axes=(1, 2))) ** 2
    energy = np.stack([power[:, band].mean(axis=1) for band in BANDS], axis=1)
    proportions = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-6)
    spectral = [proportions[:, 2], proportions[:, 3]]
    high_freq = power[:, BANDS[2] | BANDS[3]]
    spectral.append(np.log1p(high_freq.max(axis=1) / np.maximum(high_freq.mean(axis=1), 1e-6)))
    for lag in (2, 4, 8, 16):
        spectral.extend((_channel_corr(residual[:, :, lag:], residual[:, :, :-lag]),
                         _channel_corr(residual[:, lag:, :], residual[:, :-lag, :])))
    all_stats = {
        "rgb": rgb_stats,
        "gaussian": np.column_stack([v for c in range(3) for v in _residual_stats(tiles(gaussian[..., c]))]),
        "median": np.column_stack(_residual_stats(tiles(median))),
        "laplacian": np.column_stack(_residual_stats(tiles(laplacian))),
        "directional": np.column_stack((*_residual_stats(tiles(horizontal)),
                                        *_residual_stats(tiles(vertical)))),
        "spectrum": np.column_stack(spectral),
        "texture": texture[:, None],
    }

    def summarize(indices, surroundings=None):
        flat = indices[np.argsort(texture[indices], kind="stable")[:max(2, (len(indices)+2)//3)]]
        return {key: _summarize(values, flat, indices, surroundings)
                for key, values in all_stats.items()}

    whole = summarize(np.arange(len(xx)))
    if study_periodicity:
        whole["edges"], _, _ = _edge_features(gray, None, False)
        flat = np.argsort(texture, kind="stable")[:max(2, len(texture)//3)]
        tile_power = power[flat]
        normalized_power = tile_power / np.maximum(tile_power.sum(axis=(1, 2), keepdims=True), 1e-9)
        whole["spectrum_map"] = np.median(normalized_power, axis=0)
        # Contrast gradients across the current JPEG grid with adjacent interior gradients.
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        whole["jpeg_grid"] = np.array([
            dx[:, 7::8].mean() / max(dx[:, 3::8].mean(), 0.01),
            dy[7::8, :].mean() / max(dy[3::8, :].mean(), 0.01),
        ])
    if roi is None:
        return whole, None, None, None
    indices = _region_indices(roi, width, height, xx, yy)
    edge_values, overlay, edge_summary = _edge_features(gray, roi, preview)
    if indices is None:
        return whole, None, overlay, edge_summary
    region = summarize(*indices)
    region["edges"] = edge_values
    return whole, region, overlay, edge_summary


def predict_vector(head, features):
    keys = head["families"]
    vector = np.concatenate([features[key] for key in keys])
    z = np.clip((vector - head["mean"]) / head["scale"], -8, 8)
    weights = np.asarray(head["weights"])
    logit = float(np.clip(z @ weights + head["intercept"], -25, 25))
    return 1 / (1 + np.exp(-logit))


def _prediction(head, features, default_reason):
    result = {"status": "insufficient", "score": None, "reason": default_reason, "cues": []}
    if features is None or head is None:
        return result
    score = float(predict_vector(head, features))
    threshold = head["threshold"]
    if threshold is None:
        reason = "当前留出评测未达到复核阈值要求。"
    elif score >= threshold:
        result["status"], reason = "similar", "统计特征与本批豆包破损样本相似，建议核对来源。"
    else:
        result["status"], reason = "weak", "本次未检出明显的相似生成线索。"
    vector = np.concatenate([features[key] for key in head["families"]])
    contributions = np.clip((vector - head["mean"]) / head["scale"], -8, 8) * head["weights"]
    offset = 0
    cues = []
    for key in head["families"]:
        stop = offset + len(features[key])
        cues.append((float(contributions[offset:stop].sum()), FAMILY_LABELS[key]))
        offset = stop
    result.update(score=round(score, 4), reason=reason,
                  cues=[{"label": label, "direction": "raises" if effect > 0 else "lowers"}
                        for effect, label in sorted(cues, key=lambda row: -abs(row[0]))[:3]
                        if abs(effect) >= 0.05])
    return result


def load_model(model_path):
    with Path(model_path).open(encoding="utf-8") as stream:
        model = json.load(stream)
    if (not isinstance(model, dict) or model.get("schema") != SCHEMA
            or not isinstance(model.get("heads"), dict) or not isinstance(model.get("id"), str)):
        raise ValueError("破损审核模型格式不匹配。")
    return model


def analyze_image(data, roi=None, *, model_path=MODEL_PATH):
    roi = validate_roi(roi)
    image, marker = decode(data)
    width, height = image.size
    prepared = prepare(image)
    try:
        model = load_model(model_path)
    except (OSError, ValueError, KeyError):
        model = None
    whole, region, preview, edge_summary = (None, None, None, None)
    if prepared is not None and (model is not None or roi is not None):
        whole, region, preview, edge_summary = extract(prepared, roi, preview=roi is not None)
    if prepared is None:
        reason = "有效像素不足，建议用清晰原图复核。"
        region_reason = reason
    elif model is None:
        reason = region_reason = "找不到可用的本地破损审核模型。"
    else:
        reason = "整图训练样本不足。"
        region_reason = ("框内或周边像素不足，请适当扩大或缩小选区。" if region is None
                         else "框选训练样本不足。")
    heads = model["heads"] if model else {}
    return {
        "model": model["id"] if model else None, "device": "cpu", "source_marker": marker,
        "image_size": [width, height], "roi": roi,
        "whole": _prediction(heads.get("whole"), whole, reason),
        "region": _prediction(heads.get("region"), region, region_reason) if roi else None,
        "edge_preview_png": preview, "edge_summary": edge_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="图片文件或目录")
    parser.add_argument("--roi", type=json.loads, help="EXIF 纠正后的归一化 [左,上,右,下]")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    batch = args.input.is_dir()
    paths = ([p for p in sorted(args.input.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
             if batch else [args.input])
    if not paths:
        parser.error("目录没有 JPG 或 PNG 图片。")
    status = 0
    for path in paths:
        try:
            result = analyze_image(path.read_bytes(), args.roi, model_path=args.model)
        except (ValueError, OSError) as exc:
            result = {"error": str(exc)}
            status = 1
        print(json.dumps({"file": path.name, **result}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

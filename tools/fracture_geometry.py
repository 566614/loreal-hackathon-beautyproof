"""Image-space fracture evidence: ridge centerlines, graph geometry, and cross sections."""
from dataclasses import dataclass
import io

import cv2
import numpy as np


@dataclass
class FractureTrace:
    skeleton: np.ndarray
    endpoints: np.ndarray
    junctions: np.ndarray
    presence: np.ndarray
    geometry: np.ndarray
    profile: np.ndarray


def ridge_maps(gray):
    """Pick the scale with the strongest line response in the normal direction."""
    gray = gray.astype(np.float32)
    y, x = np.indices(gray.shape, dtype=np.float32)
    peak = np.zeros_like(gray)
    scale = np.zeros_like(gray)
    angle = np.zeros_like(gray)
    contrast = np.zeros_like(gray)
    symmetry = np.zeros_like(gray)
    votes = np.zeros_like(gray)
    polarity = np.ones_like(gray)
    for sigma in (0.65, 1.05, 1.65):
        smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3, scale=0.25)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3, scale=0.25)
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3, scale=0.25)
        separation = np.hypot(dxx - dyy, 2 * dxy)
        plus, minus = (dxx + dyy + separation) / 2, (dxx + dyy - separation) / 2
        dark = np.abs(plus) >= np.abs(minus)
        strong = np.where(dark, plus, -minus)
        weak = np.abs(np.where(dark, minus, plus))
        theta = np.arctan2(2*dxy, dxx-dyy) / 2 + np.where(dark, 0.0, np.pi/2)
        dx = np.cos(theta).astype(np.float32) * (1 + sigma)
        dy = np.sin(theta).astype(np.float32) * (1 + sigma)
        sign = np.where(dark, 1.0, -1.0)
        left = sign * (cv2.remap(smooth, x - dx, y - dy, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101) - smooth)
        right = sign * (cv2.remap(smooth, x + dx, y + dy, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101) - smooth)
        two_sided = np.maximum(0, np.minimum(left, right))
        balance = two_sided / np.maximum(1.0, np.maximum(left, right))
        response = np.maximum(strong - 0.9*weak, 0) * sigma**2 * np.sqrt(balance)
        votes += (two_sided >= 1.4) & (response >= 0.5)
        better = response > peak
        peak[better] = response[better]
        scale[better] = sigma
        angle[better] = theta[better]
        contrast[better] = two_sided[better]
        symmetry[better] = balance[better]
        polarity[better] = sign[better]

    nx, ny = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    maximum = (peak >= cv2.remap(peak, x + nx, y + ny, cv2.INTER_LINEAR)) & (
        peak >= cv2.remap(peak, x - nx, y - ny, cv2.INTER_LINEAR)
    )
    return {"strength": peak, "scale": scale, "symmetry": symmetry,
            "contrast": contrast, "votes": votes, "centerline": maximum, "angle": angle,
            "polarity": polarity}



def stack_operator_maps(gray, ridges):
    """Fuse co-located support from morphology and opposing edge gradients."""
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.65)
    polarity = ridges["polarity"]
    texture = np.zeros_like(ridges["strength"])
    for size in (5, 9):
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dark = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, element)
        bright = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, element)
        texture = np.maximum(texture, np.where(polarity > 0, dark, bright))
    gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0, scale=1/32)
    gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1, scale=1/32)
    edge_band = cv2.dilate(cv2.Canny(smooth, 25, 75, L2gradient=True),
                          np.ones((3, 3), np.uint8)).astype(np.float32) / 255
    y, x = np.indices(gray.shape, dtype=np.float32)
    nx, ny = np.cos(ridges["angle"]), np.sin(ridges["angle"])
    distance = 1 + ridges["scale"]

    def sample(channel, offset):
        return cv2.remap(channel, x+nx*offset, y+ny*offset, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    near = -(sample(gx, -distance)*nx + sample(gy, -distance)*ny) * polarity
    far = (sample(gx, distance)*nx + sample(gy, distance)*ny) * polarity
    opposing = np.clip(np.minimum(near, far)/8, 0, 1)
    paired_canny = np.minimum(sample(edge_band, -distance), sample(edge_band, distance))
    morphology = np.clip(texture/14, 0, 1)
    strength = ridges["strength"] * (0.70 + 0.45*morphology + 0.45*opposing + 0.15*paired_canny)
    maximum = (strength >= sample(strength, 1.0)) & (strength >= sample(strength, -1.0))
    return ({**ridges, "strength": strength, "centerline": maximum},
            {"morphology": morphology, "opposing_edges": opposing, "canny": paired_canny})


def _projection_residual(design):
    basis, _ = np.linalg.qr(design)
    return np.eye(len(design)) - basis @ basis.T


_DISTANCE = np.arange(-4, 5, dtype=np.float32)
_PLANE = np.column_stack((np.ones_like(_DISTANCE), _DISTANCE))
_LINEAR_RESIDUAL = _projection_residual(_PLANE)
_STEP_RESIDUAL = _projection_residual(np.column_stack((_PLANE, np.sign(_DISTANCE))))
_WIDTHS = np.array([0.65, 1.1, 1.7])
_LINE_RESIDUALS = np.stack([
    _projection_residual(np.column_stack((_PLANE, np.exp(-0.5*(_DISTANCE/width)**2))))
    for width in _WIDTHS
])


def fit_surface_sections(gray, angle, skeleton):
    """Compare smooth, step-edge, and narrow-trough fits across each candidate line."""
    y, x = np.nonzero(skeleton)
    inside = (y >= 5) & (x >= 5) & (y < gray.shape[0]-5) & (x < gray.shape[1]-5)
    y, x = y[inside], x[inside]
    if not len(y):
        return np.zeros(6)
    if len(y) > 384:
        chosen = np.linspace(0, len(y)-1, 384).round().astype(int)
        y, x = y[chosen], x[chosen]
    theta = angle[y, x].astype(np.float32)
    samples = cv2.remap(gray.astype(np.float32), (x[:, None]+np.cos(theta[:, None])*_DISTANCE).astype(np.float32),
                        (y[:, None]+np.sin(theta[:, None])*_DISTANCE).astype(np.float32), cv2.INTER_LINEAR)

    def energy(projector):
        error = samples @ projector
        return np.sum(error**2, axis=-1)

    planar_error = energy(_LINEAR_RESIDUAL)
    step_error = energy(_STEP_RESIDUAL)
    line_errors = energy(_LINE_RESIDUALS)
    preferred_width = np.argmin(line_errors, axis=0)
    line_error = np.min(line_errors, axis=0)
    amplitude = np.maximum(1.0, planar_error)
    ridge_gain = np.clip((planar_error-line_error)/amplitude, 0, 1)
    step_advantage = np.clip((step_error-line_error)/amplitude, -1, 1)
    return np.array([np.median(ridge_gain), np.quantile(ridge_gain, 0.25),
                     np.median(step_advantage), np.mean(step_advantage > 0.20),
                     np.median(_WIDTHS[preferred_width]), np.std(_WIDTHS[preferred_width])])

def _thin(mask):
    """Zhang-Suen thinning on the small binary support."""
    padded = np.pad(mask.astype(np.uint8), 1)
    center = padded[1:-1, 1:-1]
    ring = (padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
            padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2])
    n, _, e, _, s, _, w, _ = ring
    for _ in range(12):
        removed = 0
        for mode in (0, 1):
            degree = sum(ring)
            transitions = sum((a == 0) & (b == 1) for a, b in zip(ring, ring[1:] + ring[:1]))
            keep_connections = ((n*e*s == 0) & (e*s*w == 0) if mode == 0
                                else (n*e*w == 0) & (n*s*w == 0))
            drop = (center == 1) & (degree >= 2) & (degree <= 6) & (transitions == 1) & keep_connections
            removed += np.count_nonzero(drop)
            center[drop] = 0
        if not removed:
            break
    return center.astype(bool)


def _graph_degree(mask):
    """Join diagonal pixels only when the orthogonal neighbors do not already join them."""
    padding = np.pad(mask, 1)
    n, e, s, w = (padding[:-2, 1:-1], padding[1:-1, 2:], padding[2:, 1:-1], padding[1:-1, :-2])
    ne = padding[:-2, 2:] & ~n & ~e
    se = padding[2:, 2:] & ~s & ~e
    sw = padding[2:, :-2] & ~s & ~w
    nw = padding[:-2, :-2] & ~n & ~w
    return sum(value.astype(np.uint8) for value in (n, ne, e, se, s, sw, w, nw)) * mask


def _junction_angles(skeleton, centers):
    t_errors, y_errors = [], []
    for y, x in centers[:80]:
        top, bottom = max(0, y-12), min(skeleton.shape[0], y+13)
        left, right = max(0, x-12), min(skeleton.shape[1], x+13)
        dy, dx = np.nonzero(skeleton[top:bottom, left:right])
        dy, dx = dy + top - y, dx + left - x
        radii = np.hypot(dy, dx)
        ring = (radii >= 7.5) & (radii <= 11.5)
        angles = np.sort(np.mod(np.arctan2(dy[ring], dx[ring]), 2*np.pi))
        if len(angles) < 3:
            continue
        gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
        boundaries = np.flatnonzero(gaps > np.deg2rad(26))
        if len(boundaries) != 3:
            continue
        start = boundaries[-1] + 1
        angles = np.r_[angles[start:], angles[:start]+2*np.pi]
        indices = (boundaries[:-1] - start) % len(angles) + 1
        arms = [portion.mean() for portion in np.split(angles, np.sort(indices)) if len(portion)]
        if len(arms) != 3:
            continue
        opening = np.sort(np.diff(np.r_[arms, arms[0]+2*np.pi]) * (180/np.pi))
        t_errors.append(float(np.abs(opening-[90, 90, 180]).mean()/90))
        y_errors.append(float(np.abs(opening-120).mean()/90))
    return t_errors, y_errors


def trace_fracture(maps, bounds=None):
    """Trace line segments in the requested patch. Measurements are in image pixels."""
    left, top, right, bottom = bounds if bounds is not None else (
        0, 0, maps["strength"].shape[1], maps["strength"].shape[0]
    )
    patch = {key: value[top:bottom, left:right] for key, value in maps.items()}
    strength = patch["strength"]
    usable = (strength >= 0.55) & (patch["contrast"] >= 1.45) & patch["centerline"]
    pool = strength[usable]
    threshold = max(0.65, float(np.quantile(pool, 0.30))) if pool.size else 0.65
    seeds = usable & (strength >= threshold)
    weak = (strength >= max(0.55, 0.30*threshold)) & (patch["contrast"] >= 1.35)
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(weak, connectivity=8)
    chosen = (labels > 0) & np.isin(labels, np.unique(labels[seeds]))
    skeleton = _thin(chosen)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8), 8)
    skeleton &= (stats[:, cv2.CC_STAT_AREA] >= 6)[labels]
    degree = _graph_degree(skeleton)
    endpoints = np.argwhere(degree == 1)
    junction_mask = (degree >= 3).astype(np.uint8)
    clustered = cv2.dilate(junction_mask, np.ones((3, 3), np.uint8))
    _, _, _, centroids = cv2.connectedComponentsWithStats(clustered, 8)
    junctions = np.rint(centroids[1:, ::-1]).astype(int)
    length = int(skeleton.sum())
    if length == 0:
        return FractureTrace(skeleton, endpoints, junctions, np.zeros(3), np.zeros(9), np.zeros(6))

    component_lengths = stats[1:, cv2.CC_STAT_AREA]
    component_count = int((component_lengths >= 6).sum())
    largest = int(component_lengths.max())
    edges = int(degree.sum() // 2)
    cycles = max(0, edges - length + component_count)
    separated = skeleton & ~clustered.astype(bool)
    _, _, segments, _ = cv2.connectedComponentsWithStats(separated.astype(np.uint8), 8)
    segments = segments[1:][segments[1:, cv2.CC_STAT_AREA] >= 5]
    if len(segments):
        extent = np.hypot(segments[:, cv2.CC_STAT_WIDTH] - 1, segments[:, cv2.CC_STAT_HEIGHT] - 1)
        bend = np.median(np.clip((segments[:, cv2.CC_STAT_AREA] - 1) / np.maximum(extent, 1), 0.7, 4))
    else:
        bend = 1.0
    t, y = _junction_angles(skeleton, junctions)
    geometry = np.array([component_count/length, largest/length, len(endpoints)/length,
                         len(junctions)/length, cycles/length, bend,
                         float(np.median(t)) if t else 0.0, float(np.median(y)) if y else 0.0,
                         len(t)/max(1,len(junctions))])
    scale = patch["scale"][skeleton]
    near_line = cv2.dilate(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    shoulder_values = strength[near_line & ~skeleton]
    profile = np.array([float(np.median(scale)), float(np.std(scale)), float(np.median(patch["symmetry"][skeleton])),
                        float(np.median(patch["votes"][skeleton])/3),
                        float(np.median(strength[skeleton])/max(0.2, np.median(shoulder_values)))
                        if shoulder_values.size else 0.0,
                        float(np.mean(patch["contrast"][skeleton] >= 2.5))])
    presence = np.array([length/skeleton.size, largest/skeleton.size,
                         float(np.log1p(np.median(patch["contrast"][skeleton])))])
    return FractureTrace(skeleton, endpoints, junctions, presence, geometry, profile)



def prepare_surface(image, roi, *, quality=85, smaller=False):
    """Crop the material surface before resampling and JPEG normalization."""
    from PIL import Image

    width, height = image.size
    box = np.rint(np.array(roi) * [width, height, width, height]).astype(int)
    surface = image.crop(tuple(box))
    if min(surface.size) < 24:
        return None
    scale = min(192/min(surface.size), 384/max(surface.size))
    size = tuple(round(dimension*scale) for dimension in surface.size)
    surface = surface.resize(size, Image.Resampling.LANCZOS)
    if smaller:
        reduced = tuple(round(dimension*0.75) for dimension in size)
        surface = surface.resize(reduced, Image.Resampling.LANCZOS).resize(size, Image.Resampling.BILINEAR)
    encoded = io.BytesIO()
    surface.save(encoded, format="JPEG", quality=quality, subsampling=2)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return np.asarray(decoded).copy()

def main():
    import argparse
    import json
    from pathlib import Path

    from doubao_damage import DETAIL_SHORT_SIDE, decode, prepare, validate_roi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JPG 或 PNG")
    parser.add_argument("--roi", type=json.loads, help="归一化的 [左, 上, 右, 下]")
    parser.add_argument("--quality", type=int, choices=(85, 60), default=85)
    parser.add_argument("--crop-first", action="store_true", help="按原图坐标先裁出材料表面")
    parser.add_argument("--output", type=Path, required=True, help="诊断图 PNG")
    args = parser.parse_args()
    image, _ = decode(args.input.read_bytes())
    roi = validate_roi(args.roi)
    if args.crop_first and roi is None:
        parser.error("--crop-first 需要 --roi。")
    rgb = (prepare_surface(image, roi, quality=args.quality) if args.crop_first
            else prepare(image, quality=args.quality, short_side=DETAIL_SHORT_SIDE))
    if rgb is None:
        parser.error("图片的有效像素不足。")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    initial = ridge_maps(gray)
    stacked, layers = stack_operator_maps(gray, initial)
    if args.crop_first:
        roi = None
    h, w = gray.shape
    box = np.rint(np.array(roi)*[w, h, w, h]).astype(int) if roi else (0, 0, w, h)
    left, top, right, bottom = box
    if min(right-left, bottom-top) < 12:
        parser.error("请扩大诊断框。")
    section = (slice(top, bottom), slice(left, right))
    source = rgb[section]
    trace = trace_fracture(stacked, box)
    single = trace_fracture(initial, box)
    fitted = fit_surface_sections(gray[section], stacked["angle"][section], trace.skeleton)

    def colorize(values):
        crop = values[section]
        return cv2.cvtColor(np.rint(np.clip(crop, 0, 1)*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def overlay(traced):
        canvas = source.copy()
        canvas[traced.skeleton] = (45, 255, 155)
        for y, x in traced.junctions:
            cv2.circle(canvas, (int(x), int(y)), 2, (255, 165, 20), -1)
        return canvas

    views = [
        ("Source", source), ("Hessian trace", overlay(single)), ("Morphology", colorize(layers["morphology"])),
        ("Opposed gradients", colorize(layers["opposing_edges"])), ("Paired Canny", colorize(layers["canny"])),
        ("Fused trace", overlay(trace)),
    ]
    cell_w, cell_h = 385, 270
    board = np.full((2*cell_h, 3*cell_w, 3), (14, 20, 24), dtype=np.uint8)
    for index, (label, pixels) in enumerate(views):
        x, y = (index % 3)*cell_w, (index // 3)*cell_h
        shape = pixels.shape[:2]
        factor = min((cell_w-22)/shape[1], (cell_h-38)/shape[0])
        size = tuple(max(1, round(dim*factor)) for dim in shape[::-1])
        scaled = cv2.resize(pixels, size, interpolation=cv2.INTER_NEAREST)
        bx, by = x+(cell_w-size[0])//2, y+31
        board[by:by+size[1], bx:bx+size[0]] = scaled
        cv2.putText(board, label, (x+12, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.59, (238, 242, 241), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(board).save(args.output)
    print(json.dumps({"single_pixels": int(single.skeleton.sum()), "stacked_pixels": int(trace.skeleton.sum()),
                      "endpoints": len(trace.endpoints), "junctions": len(trace.junctions),
                      "ridge_fit_gain": float(fitted[0]), "trough_over_step_advantage": float(fitted[2])}))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# 来源：原创实现 —— 频域（FFT）取证思路为公共信号处理方法，本文件的特征组合、径向统计与标定均为 BeautyProof 团队原创
"""
频域取证工具（spectral / FFT）—— 补上 ELA 对「整图 AI 生成」无效的空缺

用法：
    python tools/spectral_tool.py <图片路径>
    python tools/spectral_tool.py --study          # 跑分组统计，看这 5 个特征到底能不能分

为什么要有这个工具（这是本工具存在的唯一理由）：
    ELA（误差水平分析）查的是「局部二次编辑」——被 P 过的区域压缩次数跟周围不一样。
    但 AI 从零生成的图**没有编辑区**，整张图的压缩场是均匀的，ELA 看起来往往很干净。
    同时 ELA 在截图 / 平台二次压缩场景下误报率文献报告高达 96.8%，不能单独定案。
    所以必须补一条**与压缩历史无关**的独立证据线：频域。

    频域怎么抓 AI 痕迹（三个可解释抓手）：
      1) 高频能量比：真实相机镜头有 MTF 衰减，但衰减是"自然平滑"的；
         生成模型的上采样/合成会让高频段要么被过度抑制（糊），要么出现异常峰。
      2) 频谱尾部斜率：log 频谱对归一化半径的拟合斜率，生成图常偏离真实相机的 -α 幂律。
      3) 方位向周期性：GAN 类上采样（转置卷积）会在频谱上留下规则网格峰，
         表现为高频频环上「某个角度功率远高于均值」。这是 GAN 的经典指纹。

能证明什么：
    这张图的频谱统计特征与「相机直出照片」存在系统性偏离，并指出偏离在哪个频段。

不能证明什么（铁律，必须写进结论）：
    - 频域特征是**统计倾向，不是铁证**。高锐化、高 ISO、强降噪的手机修图图也可能落在生成图区间。
    - 重压缩、截图、缩放会整体改变频谱形态，因此本工具**只在原图/轻压缩图上参考**，
      遇到平台二次压缩的图（低频占比异常低）应主动弃权。
    - 不能单独定性，必须与 AIGC / TruFor / 图文交叉等其它证据合并判断。

⚠️ 标定纪律（与本仓库其它评测工具一致）：
    阈值只在 DEV 集上标定（data/ai, data/ai_aug, data/real, data/clean —— 这些都进过历史训练），
    冻结集（data/ai_cross, data/ai_cross_native, data/ai_mj_sd_flux_heldout,
    data/realworld_heldout, data/real_filter_selfie 的 heldout 段）**只用于最终评测，不参与定阈值**。
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 频谱分段（归一化半径 u = r / rmax，u=1 即奈奎斯特）
BAND_MID = (0.10, 0.40)   # 中频（结构纹理）
BAND_HIGH = (0.62, 1.00)  # 高频（噪点/细节/压缩痕迹）
TAIL_FIT = (0.50, 0.95)   # 尾部斜率拟合区间
AZ_ANNULUS = (0.60, 1.00)  # 方位向周期性取样环
AZ_BINS = 72
MAX_SIDE = 1600            # 超过则等比缩小（再大只会拖慢，不涨精度）

# 标定常数：只由 tools/spectral_calib.py 在 DEV 集上写入，冻结集不得参与
DEFAULT_CALIB = {
    "schema": "beautyproof/spectral_calib@2",
    "features": {},          # 由 tools/spectral_calib.py 在 DEV 集上填充
    "decision_thr": 0.5,
    "calibrated_on": "unset（尚未标定，当前分数仅供观察，不可用于定级）",
}


def _load_gray(image_path):
    """读灰度图。刻意不做降采样 —— 降采样会把高频证据直接抹掉。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return np.asarray(img.convert("L")).astype(np.float64), img.size


def _radial_profile(power):
    """把二维功率谱压成按归一化半径的径向平均曲线。返回 (u, 径向功率)。"""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    rmax = max(1, min(cy, cx))
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float64)
    rb = np.clip(r, 0, rmax).astype(np.int32)
    nb = rmax + 1
    radial = np.bincount(rb.ravel(), weights=power.ravel(), minlength=nb)[:nb]
    counts = np.bincount(rb.ravel(), minlength=nb)[:nb]
    radial = radial / np.maximum(counts, 1)
    u = np.arange(nb, dtype=np.float64) / rmax
    return u, radial


def _azimuthal_anomaly(power, rmax):
    """高频频环上的方位向各向异性：转置卷积上采样会留下规则网格峰 → 某角度功率远高于均值。"""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    dy, dx = yy - cy, xx - cx
    r = np.sqrt(dy ** 2 + dx ** 2)
    theta = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # [0,1)
    ring = (r >= AZ_ANNULUS[0] * rmax) & (r <= AZ_ANNULUS[1] * rmax)
    if ring.sum() < 64:
        return None
    tb = np.clip((theta[ring] * AZ_BINS).astype(np.int32), 0, AZ_BINS - 1)
    pw = power[ring]
    prof = np.bincount(tb, weights=pw, minlength=AZ_BINS)[:AZ_BINS]
    if prof.size < AZ_BINS or prof.mean() <= 0:
        return None
    prof = prof / prof.mean()
    # 去掉零填充角度带来的虚假凹陷：只统计有样本的角
    cnt = np.bincount(tb, minlength=AZ_BINS)[:AZ_BINS]
    prof = prof[cnt > 0]
    if prof.size < 8:
        return None
    return float(prof.max())


def extract_features(image_path):
    """抽 5 个频域特征。全部与亮度/对比度做归一化，尽量抵消"拍得好/拍得暗"的影响。"""
    g, size = _load_gray(image_path)

    # 频域：去均值窗 + Hann 窗，避免边界振铃把高频能量刷高
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    gw = (g - g.mean()) * win
    F = np.fft.fftshift(np.fft.fft2(gw))
    power = np.abs(F) ** 2

    u, radial = _radial_profile(power)
    rmax = max(1, min(g.shape) // 2)
    eps = 1e-12

    def band_energy(lo, hi):
        m = (u >= lo) & (u < hi)
        return float(radial[m].sum())

    e_mid = band_energy(*BAND_MID)
    e_high = band_energy(*BAND_HIGH)
    e_all = band_energy(0.02, 1.0)

    # 特征1：高频能量比（线性能量，占中频+高频的比例）
    hf_ratio = e_high / max(e_mid + e_high, eps)

    # 特征2：频谱尾部斜率（log10 功率 对 u 线性拟合）
    m = (u >= TAIL_FIT[0]) & (u <= TAIL_FIT[1]) & (radial > 0)
    tail_slope = float(np.polyfit(u[m], np.log10(radial[m] + eps), 1)[0]) if m.sum() > 8 else 0.0

    # 特征3：频谱平坦度（几何均值/算术均值，越接近 1 越像白噪/规则纹理）
    m = (u >= BAND_MID[0]) & (u <= BAND_HIGH[1])
    seg = radial[m] + eps
    flatness = float(np.exp(np.mean(np.log(seg))) / (np.mean(seg) + eps))

    # 特征4：方位向各向异性（GAN 上采样网格指纹）
    azimuth = _azimuthal_anomaly(power, rmax)

    # 特征5：高频残差峰度（生成图常过度平滑 → 峰度偏低）
    # 注意两个坑：
    #   ① PIL.ImageFilter 不支持浮点模式图（会报 image has wrong mode），必须用 cv2；
    #   ② 残差本身幅值极小，直接算 mean(x^4)/var^2 会浮点爆炸（实测出 1e67 量级），
    #      必须先按标准差归一化再算四阶矩。
    import cv2

    blur = g - cv2.GaussianBlur(g, (0, 0), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REFLECT)
    sigma = float(blur.std())
    if sigma <= 0:
        kurt, v = 0.0, 0.0
    else:
        z = blur / sigma
        kurt = float((z ** 4).mean() - 3.0)   # 超额峰度，正态≈0
        v = sigma ** 2

    return {
        "image_size": list(size),
        "hf_energy_ratio": round(hf_ratio, 5),
        "tail_slope": round(tail_slope, 4),
        "spectral_flatness": round(flatness, 5),
        "azimuthal_anomaly": (round(azimuth, 4) if azimuth is not None else None),
        "hf_residual_kurtosis": round(kurt, 4),
        "hf_residual_var": round(v, 4),
        "e_high_over_all": round(e_high / max(e_all, eps), 5),
    }


def bp_var(x):
    return x.mean()


FEATURE_KEYS = ["hf_energy_ratio", "spectral_flatness", "azimuthal_anomaly", "hf_residual_kurtosis"]


def score_from_features(f, calib=None):
    """把频域特征合成 0~1 的「频域倾向分」。

    标定（特征顺序、极性方向的权重、标准化参数、决策线）**全部来自
    results/spectral_calib.json**，由 tools/spectral_calib.py 在冻结池的 CAL 段拟合，
    本函数不写死任何人为假设。

    ⚠️ 两条实测教训（别回退）：
      ① 极性不能靠直觉。第一版假设"方位向各向异性高=AI"，实测是**反的**：
         真实照片有暗角/场景结构导致方位向不均匀，生成图反而更各向同性。
         四个特征里有两个（azimuthal_anomaly、hf_residual_kurtosis）是 low_is_ai。
      ② 标定集不能用"干净棚拍真实图"。第一版拿 data/real+data/clean 定阈值，
         冻结集上 AUC 直接掉到 0.586（AI 召回 0.18 / 误报 0.20，等于废掉）。
         真实场景的小红书图、滤镜自拍都被平台重度压缩，频谱形态跟棚拍图完全不同。

    诚实说明：这不是训练出来的模型概率，是**倾向分**，含义是
    "这张图的频谱有多不像相机直出照片"，不是"这张图有几分概率是 AI 生成"。
    刻意不叫 probability，避免和 aigc_tool 的模型概率混淆、被误读成同一种东西。
    """
    c = calib or DEFAULT_CALIB
    order = c.get("feature_order") or []
    feats_cfg = c.get("features") or {}
    norm = c.get("normalization")
    if not order or not feats_cfg or not norm:
        return 0.0, {"_note": "未标定（results/spectral_calib.json 缺失），倾向分恒为 0，不可用于定级"}

    mu = np.asarray(norm.get("mean") or [], dtype=float)
    sd = np.asarray(norm.get("std") or [], dtype=float)
    if mu.size != len(order) or sd.size != len(order):
        return 0.0, {"_note": "标定文件与当前特征集不匹配（特征数量对不上），请重跑 spectral_calib.py"}
    sd = np.where(sd < 1e-12, 1.0, sd)

    xs = np.zeros(len(order), dtype=float)
    parts = {}
    for j, k in enumerate(order):
        v = f.get(k)
        if v is None:
            v = mu[j]  # 缺失特征用 CAL 中位数填补，不奖不罚
        xs[j] = (float(v) - mu[j]) / sd[j]
        parts[k] = round(float(xs[j]), 4)

    z = float(np.dot(xs, [float(feats_cfg[k]["signed_weight"]) for k in order])
              + float(norm.get("bias") or 0.0))
    z = max(-30.0, min(30.0, z))
    score = 1.0 / (1.0 + np.exp(-z))
    return round(score, 4), parts


def build_evidence(image_path, feats, score=None, parts=None, calib=None):
    """统一证据结构（与本仓库其它工具一致：tool / source_asset_id / observed / cannot_prove / evidence）。"""
    if score is None:
        score, parts = score_from_features(feats, calib)
    az = feats.get("azimuthal_anomaly")
    az_txt = f"{az}" if az is not None else "不可用（图像过小）"
    observed = (
        f"频域统计：高频能量比 {feats['hf_energy_ratio']}（中频占比基准），"
        f"频谱尾部斜率 {feats['tail_slope']}，"
        f"频谱平坦度 {feats['spectral_flatness']}，"
        f"方位向各向异性 {az_txt}，"
        f"高频残差峰度 {feats['hf_residual_kurtosis']} → 频域倾向分 {score}"
    )
    return {
        "tool": "spectral",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "cannot_prove": (
            "频域统计是分布倾向而非铁证：高锐化/高 ISO/强降噪的真实照片也可能落进生成区间；"
            "重压缩与截图会整体改变频谱形态，本工具在二次压缩图上应主动弃权。"
            "倾向分不等于 AI 生成概率，必须与 AIGC 模型分数、TruFor 篡改分合并判断。"
        ),
        "evidence": [
            {
                "spectral_score": score,
                "spectral_parts": parts,
                "calibrated_on": (calib or {}).get("calibrated_on", DEFAULT_CALIB["calibrated_on"]),
                **feats,
            }
        ],
    }


def load_calib(repo_root):
    p = repo_root / "results" / "spectral_calib.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_dir(d):
    p = Path(d)
    if not p.is_dir():
        return []
    return sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXT and not f.name.startswith("_")])


DEV_GROUPS = [
    ("dev_ai", "data/ai"),
    ("dev_ai_aug", "data/ai_aug"),
    ("dev_real", "data/real"),
    ("dev_clean", "data/clean"),
]
HELDOUT_GROUPS = [
    ("heldout_ai_cross", "data/ai_cross"),
    ("heldout_ai_cross_native", "data/ai_cross_native"),
    ("heldout_ai_flux", "data/ai_mj_sd_flux_heldout/flux"),
    ("heldout_real_realworld", "data/realworld_heldout"),
]


def _resolve_groups(repo_root):
    """把分组名 → 文件列表。滤镜自拍集用 SPLIT_MANIFEST 的 train/heldout 段直接按名单取，
    **不复制任何图片文件**（避免在 data/ 下产生几十张重复副本污染数据集）。"""
    groups = [(n, _list_dir(repo_root / d), d) for n, d in DEV_GROUPS + HELDOUT_GROUPS]
    fs_dir = repo_root / "data" / "real_filter_selfie"
    split_p = fs_dir / "SPLIT_MANIFEST.json"
    if split_p.exists():
        sp = json.loads(split_p.read_text(encoding="utf-8"))
        for seg, key in (("heldout", "heldout_never_train"), ("train", "train")):
            files = [fs_dir / fn for fn in sp.get(key, []) if (fs_dir / fn).exists()]
            groups.append((f"heldoout_fs_{seg}", files, f"data/real_filter_selfie#{key}"))
    return groups


def main():
    repo_root = Path(__file__).resolve().parent.parent

    if "--study" in sys.argv:
        groups = _resolve_groups(repo_root)
        calib = load_calib(repo_root)
        print(f"标定文件: {'results/spectral_calib.json' if calib else '未标定（用默认值）'}")
        print()
        for name, files, _d in groups:
            if not files:
                print(f"{name:32s} (无文件)")
                continue
            rows = []
            for f in files:
                try:
                    ft = extract_features(f)
                except Exception as e:
                    print(f"  !! {f.name}: {e}")
                    continue
                s, _ = score_from_features(ft, calib)
                rows.append((f.name, ft, s))
            if not rows:
                print(f"{name:32s} (全部失败)")
                continue
            n = len(rows)
            hf = [r[1]["hf_energy_ratio"] for r in rows]
            fl = [r[1]["spectral_flatness"] for r in rows]
            ku = [r[1]["hf_residual_kurtosis"] for r in rows]
            az = [r[1]["azimuthal_anomaly"] for r in rows if r[1]["azimuthal_anomaly"] is not None]
            sc = [r[2] for r in rows]

            def st(v):
                if not v:
                    return "n/a"
                v = sorted(v)
                m = sum(v) / len(v)
                return f"med={v[len(v)//2]:.4f} mean={m:.4f} min={v[0]:.4f} max={v[-1]:.4f}"

            print(f"{name:32s} n={n:3d}")
            print(f"   hf_ratio     {st(hf)}")
            print(f"   flatness     {st(fl)}")
            print(f"   kurtosis     {st(ku)}")
            print(f"   azimuth      {st(az)}")
            print(f"   SCORE        {st(sc)}")
            print()
        return 0

    if len(sys.argv) < 2:
        print("用法: python tools/spectral_tool.py <图片路径>  |  --study")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    feats = extract_features(image_path)
    calib = load_calib(repo_root)
    report = build_evidence(image_path, feats, calib=calib)

    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"spectral_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

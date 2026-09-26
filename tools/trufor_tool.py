# -*- coding: utf-8 -*-
# 来源：第三方改编 —— 封装 TruFor (CVPR2023 开源) 检测能力；统一证据格式与规则联动为 BeautyProof 团队原创
"""
TruFor 工具 —— 用开源深度学习模型判断「这张图有没有被人工篡改 / P 过」

用法：
    python tools/trufor_tool.py <图片路径>              # 单张（有缓存秒回）
    python tools/trufor_tool.py --batch <图1> <图2> ...  # 一次跑多张（推荐，模型只加载一次）

大白话：
    TruFor 是 CVPR 2023 的图像取证模型（官方开源）。它不像 AIGC 检测那样只判
    "是不是 AI 画的"，而是真的去查"这张图有没有被人工动过手脚"。
    它给三样东西：①整图篡改分(0~1) ②篡改区域定位图 ③可靠性图。

能证明什么：
    这张图「有没有被篡改的痕迹」，以及「大概哪块区域可疑」。

不能证明什么（铁律）：
    模型也会误判。分数高 ≠ 一定有罪（压缩、滤镜也会留痕）；
    分数低 ≠ 绝对干净（高明的篡改可能不留痕）。它只是"痕迹检测"，不是法官。

依赖：torch / timm / kornia / opencv-python / matplotlib（已装好）
还需要：TruFor 官方代码 + 官方权重 —— 跑 python tools/setup_trufor.py 一键部署
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rule_engine import THRESHOLDS  # 复用规则引擎篡改阈值，避免与定级阈值失同步

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PY = "C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
CACHE_FILE = REPO / "outputs" / "trufor_cache.json"


# ---------------------------------------------------------------- 定位代码与权重
def find_trufor_dir():
    """找到 TruFor 官方代码里的 TruFor_train_test 目录（里面有 test.py）"""
    # 1) 环境变量优先（方便你把代码放任意位置）
    env = os.environ.get("TRUFOR_DIR")
    if env and (Path(env) / "test.py").exists():
        return Path(env)
    # 2) 项目内标准位置 + 我下载时的临时位置
    for cand in [
        REPO / "models" / "TruFor" / "TruFor_train_test",
        REPO.parent / "_trufor_src" / "TruFor_train_test",
        REPO.parent / "_trufor_prep" / "TruFor_train_test",
    ]:
        if (cand / "test.py").exists():
            return cand
    return None


def find_weight(trufor_dir):
    """找到官方训练权重 trufor.pth.tar"""
    if trufor_dir is None:
        return None
    pm = trufor_dir / "pretrained_models"
    for name in ["trufor.pth.tar", "TruFor_weights/trufor.pth.tar"]:
        p = pm / name
        if p.exists():
            return p
    # 兜底：在 pretrained_models 下递归找
    hits = sorted(pm.rglob("trufor*.pth.tar")) if pm.exists() else []
    return hits[0] if hits else None


def model_ready():
    """返回 (代码目录, 权重路径)；任一缺失则该项为 None —— 调用方据此优雅降级"""
    d = find_trufor_dir()
    return d, find_weight(d)


# ---------------------------------------------------------------- 缓存
def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 推理
def estimate_tampered_ratio(m):
    """估算可疑区域占比。

    官方 map 的具体值域不保证是 0~1，所以这里用「相对阈值」
    （均值+2倍标准差），与值域无关，避免写出死的阈值。
    """
    m = np.asarray(m, dtype=np.float64)
    if m.size == 0:
        return 0.0
    thr = m.mean() + 2 * m.std()
    if not np.isfinite(thr):
        return 0.0
    return float((m > thr).mean())


def estimate_tampered_position(m):
    """估算可疑区域集中在图的哪个方位，供「人话结论」直接说人话用。

    与 estimate_tampered_ratio 用**同一套相对阈值**（均值+2σ），保证「占比」和「方位」
    说的是同一批像素，不会互相打架。

    返回如「画面右下角」「画面中部」「画面左侧」；可疑像素太少时返回 None
    （宁可不说，也不瞎指一个位置）。
    """
    m = np.asarray(m, dtype=np.float64)
    if m.size == 0:
        return None
    thr = m.mean() + 2 * m.std()
    if not np.isfinite(thr):
        return None
    ys, xs = np.where(m > thr)
    if ys.size == 0:
        return None
    h, w = m.shape[:2]
    # 用中位数而非均值：抗离群点，零星噪点不会把方位带偏
    cy = float(np.median(ys)) / max(h - 1, 1)
    cx = float(np.median(xs)) / max(w - 1, 1)

    def band(v):
        if v < 0.34:
            return -1
        if v > 0.66:
            return 1
        return 0

    vy, vx = band(cy), band(cx)
    if vy == 0 and vx == 0:
        return "画面中部"
    # 中文习惯「左右在前、上下在后」：右下角 / 左上角，不是「下右角」
    parts = []
    if vx == -1:
        parts.append("左")
    elif vx == 1:
        parts.append("右")
    if vy == -1:
        parts.append("上")
    elif vy == 1:
        parts.append("下")
    if len(parts) == 2:
        return f"画面{parts[0]}{parts[1]}角"
    return f"画面{parts[0]}{'部' if parts[0] in ('上', '下') else '侧'}"


def _imread_unicode(path):
    """读图（兼容中文路径：cv2.imread 对中文路径会静默返回 None）"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path, img):
    """写图（兼容中文路径：cv2.imwrite 对中文路径会静默失败）"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    Path(path).write_bytes(buf.tobytes())
    return True


def save_localization_map(npz_path, image_path):
    """把 TruFor 的定位图（npz 里的 map）画成两张能直接看的图

    产出（放在 outputs/ 下）：
        <stem>_trufor_heat.png     纯热力图：越红越可疑
        <stem>_trufor_overlay.png  原图 + 半透明热力叠加：一眼看出可疑在哪块

    返回 (heat 文件名, overlay 文件名)；失败返回 (None, None)。
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "map" not in data:
            return None, None
        m = np.asarray(data["map"], dtype=np.float32)
        img = _imread_unicode(image_path)
        if img is None:
            return None, None

        h, w = img.shape[:2]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

        # 归一化到 0~1（map 的值域官方不保证，所以按本图自己的最小最大值拉平）
        mn, mx = float(m.min()), float(m.max())
        norm = (m - mn) / (mx - mn + 1e-8)

        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # 只在"真的可疑"的地方上色：低于均值的地方淡出，避免满屏红色吓人
        mask = np.clip((norm - float(norm.mean())) / 0.35, 0, 1)[:, :, None]
        overlay = (img * (1 - 0.55 * mask) + heat * (0.55 * mask)).astype(np.uint8)

        stem = Path(image_path).stem
        heat_name = f"{stem}_trufor_heat.png"
        ovl_name = f"{stem}_trufor_overlay.png"
        _imwrite_unicode(CACHE_FILE.parent / heat_name, heat)
        _imwrite_unicode(CACHE_FILE.parent / ovl_name, overlay)
        return heat_name, ovl_name
    except Exception:  # noqa: BLE001
        return None, None


def run_trufor_batch(image_paths, force=False):
    """一次跑多张图（官方 test.py 支持传目录，这样模型只加载一次，快很多）

    返回: (cache_dict, error_str_or_None)
        cache 以图片绝对路径为 key，value 含 available / trufor_score / tampered_area_ratio
    """
    d, w = model_ready()
    cache = load_cache()

    if d is None or w is None:
        return cache, "trufor_unavailable: 官方代码或权重未部署（先跑 tools/setup_trufor.py）"

    # 统一成绝对路径再查缓存：否则你用相对路径跑、缓存里存的是绝对路径，会命中不了白白重跑模型
    image_paths = [Path(p).resolve() for p in image_paths]
    todo = [p for p in image_paths if force or str(p) not in cache]
    if not todo:
        return cache, None

    tmp_in = REPO / "outputs" / "_trufor_in"
    tmp_out = REPO / "outputs" / "_trufor_out"
    for t in (tmp_in, tmp_out):
        if t.exists():
            shutil.rmtree(t, ignore_errors=True)
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    for p in todo:
        shutil.copy2(p, tmp_in / Path(p).name)

    # 官方推理命令：test.py -in 目录 -out 输出 -exp trufor_ph3 TEST.MODEL_FILE 权重 -g -1(用CPU)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(d)  # test.py 需要 import 同目录的 lib/
    # 不直接跑官方 test.py，而是垫一层 _trufor_launch.py：
    # 新版 torch 默认 weights_only=True 会拒绝 TruFor 2023 年的老权重，
    # 启动器负责放宽这个限制（不动官方代码）。
    # 注意 -in 末尾必须带分隔符：官方 test.py 用 path.replace(root,'') 算相对路径，
    # 不带尾分隔符时 Windows 会留下前导 '\' 被当成绝对路径，结果写到 C 盘根。
    launcher = Path(__file__).resolve().parent / "_trufor_launch.py"
    cmd = [
        PY, str(launcher), str(d / "test.py"),
        "-g", "-1",
        "-in", str(tmp_in) + os.sep,
        "-out", str(tmp_out),
        "-exp", "trufor_ph3",
        "TEST.MODEL_FILE", str(w),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(d), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="ignore", timeout=3600,
        )
    except Exception as e:  # noqa: BLE001
        return cache, f"trufor_failed: {type(e).__name__}: {str(e)[:200]}"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        return cache, f"trufor_failed: rc={proc.returncode} {tail}"

    # 按文件名(stem)把 npz 对回原图（兼容 clean_01.npz 与 clean_01.png.npz 两种命名）
    npz_map = {}
    for f in tmp_out.rglob("*.npz"):
        npz_map[f.stem] = f                      # clean_01.npz      -> clean_01
        npz_map[Path(f.stem).stem] = f           # clean_01.png.npz  -> clean_01
    n_done = 0
    for p in todo:
        stem = Path(p).stem
        f = npz_map.get(stem)
        if f is None:
            cache[str(p)] = {"available": False, "reason": "npz_not_found"}
            continue
        try:
            data = np.load(f, allow_pickle=True)
            score = float(np.asarray(data["score"]).reshape(-1)[0])
            _map = np.asarray(data["map"])
            ratio = estimate_tampered_ratio(_map)
            pos = estimate_tampered_position(_map)
            heat, ovl = save_localization_map(f, p)
            cache[str(p)] = {
                "available": True,
                "trufor_score": round(score, 4),
                "tampered_area_ratio": round(ratio, 4),
                "tampered_position": pos,
                "heatmap": heat,      # outputs/ 下的定位热力图文件名（画不出就是 None）
                "overlay": ovl,       # 原图 + 热力叠加
            }
            n_done += 1
        except Exception as e:  # noqa: BLE001
            cache[str(p)] = {"available": False, "reason": f"npz_parse_failed:{type(e).__name__}"}

    # 推理成功但一张都没存下来 —— 把官方输出的尾部日志带上，方便排查
    if n_done == 0:
        tail = (proc.stderr or proc.stdout or "")[-600:]
        for p in todo:
            cache.setdefault(str(p), {"available": False, "reason": "npz_not_found"})
        print(f"[TruFor 诊断] 推理已跑完但没拿到结果，官方输出尾部：\n{tail}")

    save_cache(cache)
    return cache, None


def run_trufor(image_path, force=False):
    """单张图：优先读缓存，没有才真跑"""
    abs_path = Path(image_path).resolve()   # 与批量里的缓存 key 口径保持一致
    cache, err = run_trufor_batch([abs_path], force=force)
    hit = cache.get(str(abs_path))
    if hit is not None:
        return hit
    return {"available": False, "reason": err or "no_result"}


# ---------------------------------------------------------------- 统一证据格式
def build_evidence(image_path, res):
    """输出和 hash/c2pa/ela/ocr/aigc 完全一致的证据结构"""
    if not res.get("available"):
        return {
            "tool": "trufor",
            "source_asset_id": Path(image_path).name,
            "observed": "TruFor 检测暂不可用（官方代码或权重未部署）",
            "cannot_prove": "模型未就绪，无法判断篡改痕迹；本项按「无法判断」处理，不影响其他工具结论",
            "evidence": [{"trufor_score": None, "available": False, "reason": res.get("reason", "")}],
        }

    score = res["trufor_score"]
    ratio = res.get("tampered_area_ratio")

    if score >= THRESHOLDS["trufor_high"]:
        verdict = "篡改痕迹非常明显"
    elif score >= THRESHOLDS["trufor_suspicious"]:
        verdict = "有明显篡改痕迹"
    elif score >= THRESHOLDS["trufor_mild"]:
        verdict = "有轻微篡改痕迹"
    else:
        verdict = "未发现明显篡改痕迹"

    observed = f"TruFor 篡改分数 {score}（{verdict}）"
    if ratio is not None:
        observed += f"；可疑区域约占全图 {ratio:.1%}"
    if res.get("tampered_position"):
        observed += f"，主要集中在{res['tampered_position']}"

    return {
        "tool": "trufor",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "cannot_prove": (
            "模型也会误判：分数高不代表一定有罪（压缩/滤镜也会留痕），分数低也不代表绝对干净；"
            "且 TruFor 只擅长查「局部被人动过手脚」，官方明确说明它查不出「整张完全由 AI 生成的图」"
        ),
        "evidence": [{
            "trufor_score": score,
            "tampered_area_ratio": ratio,
            "tampered_position": res.get("tampered_position"),
            "available": True,
            "heatmap": res.get("heatmap"),    # 定位热力图（outputs/ 下）
            "overlay": res.get("overlay"),    # 原图加热力叠加
        }],
    }


# ---------------------------------------------------------------- CLI
def main():
    args = sys.argv[1:]
    if not args:
        print("用法: python tools/trufor_tool.py <图片路径>")
        print("      python tools/trufor_tool.py --batch <图1> <图2> ...")
        return 1

    force = "--force" in args
    args = [a for a in args if a != "--force"]

    out_dir = REPO / "outputs"
    out_dir.mkdir(exist_ok=True)

    if args[0] == "--batch":
        # resolve 成绝对路径：与 run_trufor_batch 内部的缓存 key 口径保持一致，
        # 否则你传相对路径时，推理跑完了却对不上缓存 key，看起来像"没结果"
        paths = [Path(a).resolve() for a in args[1:]]
        cache, err = run_trufor_batch(paths, force=force)
        if err:
            print(f"[警告] {err}")
        for p in paths:
            res = cache.get(str(p), {"available": False, "reason": err or "no_result"})
            ev = build_evidence(p, res)
            (out_dir / f"trufor_{Path(p).stem}.json").write_text(
                json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(ev, ensure_ascii=False, indent=2))
        return 0

    image_path = Path(args[0])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    res = run_trufor(image_path, force=force)
    ev = build_evidence(image_path, res)
    (out_dir / f"trufor_{image_path.stem}.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ev, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

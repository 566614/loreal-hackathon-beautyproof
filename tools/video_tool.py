# -*- coding: utf-8 -*-
"""
视频模态接入 —— 把视频接进 BeautyProof 现有的「图片取证流水线」

思路（不动现有图像主路径）：
    视频 = 一串图。我们把视频按采样率抽成若干帧，对每一帧复用现有的
    `pipeline.analyze()` 跑全套 8 工具图像取证，最后把逐帧判定聚合成
    一个「视频级」风险结论。

为什么这样最稳：
    ① 完全复用已经打磨过的图像流水线（指纹 / C2PA / ELA / OCR / AIGC /
       TruFor / 文案 / 图文交叉），不重写任何一条检测逻辑；
    ② 每一帧的证据继续写进 `outputs/<tool>_<帧名>.json`，沿用统一的
       `{tool, source_asset_id, observed, cannot_prove, evidence[]}` 格式，
       其中 source_asset_id 就是这一帧的文件名 —— 帧天然复用了统一证据格式；
    ③ 聚合规则简单可审：任一帧 high_risk → 视频 high_risk；否则取所有帧里
       风险最高的那一档，并标出是哪些帧触发了判定。

依赖：
    opencv-python-headless（已装在项目 venv 里，cv2 5.x）。
    若 venv 未装，脚本会在抽帧前优雅报错退出，并给出安装指引。

用法：
    python tools/video_tool.py --input demo/BeautyProof_演示视频.mp4
    python tools/video_tool.py --input xxx.mp4 --out outputs/analysis_video_xxx.json
    python tools/video_tool.py --input xxx.mp4 --fast          # 跳过深度学习模型，秒级
    python tools/video_tool.py --input xxx.mp4 --max-frames 20 --fps-sample 2.0

产出：
    outputs/analysis_video_<视频名>.json   视频级聚合结果（frames 里含逐帧判定）
    outputs/video_frames/<视频名>/frame_*.jpg   抽出来的帧（供人工抽查）
    outputs/<tool>_frame_*.json             每帧各自的逐工具证据（沿用统一格式）
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

# 风险级别排序（从高到低），复用于聚合时的「取最高档」
try:
    from rule_engine import RISK_LEVELS
except Exception:  # noqa: BLE001
    RISK_LEVELS = ["high_risk", "suspicious", "credible", "inconclusive"]


# ---------------------------------------------------------------- 抽帧
def extract_frames(video_path, fps_sample=1.0, max_frames=30, frames_dir=None):
    """用 cv2 抽帧：按采样率取帧，最多 max_frames 张。

    参数：
        video_path   视频路径
        fps_sample   抽帧采样率（每秒抽几帧）。默认 1.0 = 每秒 1 帧。
        max_frames   最多抽多少帧（默认 30）。视频再长也只取这么多，控成本。
        frames_dir   帧保存目录；默认 outputs/video_frames/<视频名>/

    返回 dict：
        {
          "video": {name, stem, path, fps, total_frames, duration_seconds, interval},
          "frames_dir": str,
          "frames": [ {frame_id, path, timestamp, original_idx}, ... ],
        }

    异常：缺 opencv / 文件不存在 / 打不开视频 都会以 RuntimeError 抛出，
          由上层 CLI 优雅捕获并退出。
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "未检测到 opencv（cv2）。请先用项目 venv 的 pip 安装：\n"
            "  C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/pip.exe "
            "install opencv-python-headless\n"
            "（绝不要用系统 pip 或全局安装）"
        )

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"找不到这个视频：{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开这个视频（可能不是合法 mp4 / 编码不支持）：{video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (total_frames / fps) if fps > 0 else None
        # 采样间隔：每 (fps / fps_sample) 个原始帧取 1 张
        interval = max(1, round(fps / fps_sample)) if fps > 0 else 1

        stem = video_path.stem
        if frames_dir is None:
            frames_dir = REPO / "outputs" / "video_frames" / stem
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                ts = (idx / fps) if fps > 0 else float(idx)
                fid = f"frame_{saved:04d}"
                out_path = frames_dir / f"{fid}.jpg"
                # 坑：cv2.imwrite 在中文路径下会「静默失败」（返回 False 且不报错），
                # 导致帧一张都存不下来、最后误判成「空视频」。本项目素材/视频名常含中文，
                # 因此改用 imencode + write_bytes 落盘，对 Unicode 路径安全。
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    out_path.write_bytes(buf.tobytes())
                    frames.append({
                        "frame_id": fid,
                        "path": str(out_path),
                        "timestamp": round(ts, 3),
                        "original_idx": idx,
                    })
                    saved += 1
                if saved >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(
            f"这个视频一帧都没抽出来（时长 {duration}s / 总帧数 {total_frames} / fps {fps}），"
            f"可能是空文件或编码异常。"
        )

    return {
        "video": {
            "name": video_path.name,
            "stem": stem,
            "path": str(video_path),
            "fps": round(fps, 3) if fps else None,
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2) if duration else None,
            "interval": interval,
        },
        "frames_dir": str(frames_dir),
        "frames": frames,
    }


# ---------------------------------------------------------------- 逐帧 + 聚合
def analyze_video(video_path, fast=False, max_frames=30, fps_sample=1.0,
                  verbose=True, timeout=900, out_path=None):
    """对视频抽帧 → 逐帧跑现有图像流水线 → 聚合视频级判定。

    参数：
        fast       True 时跳过 aigc / trufor 两个深度学习模型（秒级出结果，
                   代价是漏掉「整图 AI 生成」和深度学习篡改定位，仅做冒烟/预览用）。
        out_path   结果 JSON 写哪；默认 outputs/analysis_video_<视频名>.json

    返回 dict（视频级分析结果，schema = beautyproof/video-analysis@1）：
        verdict.risk_level 聚合风险（high_risk/suspicious/credible/inconclusive）
        verdict.flagged_frames / high_risk_frames 触发判定的帧
        frames[] 逐帧判定（frame_id / timestamp / risk_level / reasons）
        risk_breakdown 各档帧数统计
    """
    import pipeline  # 延迟导入，确保 sys.path 已就绪

    meta = extract_frames(video_path, fps_sample=fps_sample, max_frames=max_frames)
    video = meta["video"]
    if verbose:
        print(f"\n=== 视频：{video['name']} ===")
        print(f"  时长 {video['duration_seconds']}s / 总帧 {video['total_frames']} / "
              f"fps {video['fps']} → 抽 {len(meta['frames'])} 帧"
              f"（每 {video['interval']} 帧取 1，上限 {max_frames}）")

    t0 = time.time()
    frame_results = []
    order = {lvl: i for i, lvl in enumerate(RISK_LEVELS)}

    for i, f in enumerate(meta["frames"], 1):
        if verbose:
            print(f"\n--- 帧 {i}/{len(meta['frames'])}  {f['frame_id']} "
                  f"@ {f['timestamp']}s ---")
        try:
            res = pipeline.analyze(
                f["path"], fast=fast, verbose=verbose, text=None, timeout=timeout)
            v = res["verdict"]
            frame_results.append({
                "frame_id": f["frame_id"],
                "timestamp": f["timestamp"],
                "risk_level": v["risk_level"],
                "reasons": v.get("reasons", []),
                "tools_used": v.get("tools_used", []),
                "output_file": res.get("_output_file"),
            })
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  帧 {f['frame_id']} 分析失败：{type(e).__name__}：{e}")
            frame_results.append({
                "frame_id": f["frame_id"],
                "timestamp": f["timestamp"],
                "risk_level": "error",
                "reasons": [f"逐帧分析异常：{type(e).__name__}：{e}"],
                "tools_used": [],
                "output_file": None,
            })

    # ---------------------------------------------------------- 聚合
    valid = [r for r in frame_results if r["risk_level"] in order]
    errored = [r for r in frame_results if r["risk_level"] == "error"]

    if valid:
        # 「取最高档」天然满足「任一帧 high_risk → 视频 high_risk」
        agg_level = max(valid, key=lambda r: order[r["risk_level"]])["risk_level"]
    else:
        agg_level = "inconclusive"

    high_risk_frames = [r for r in valid if r["risk_level"] == "high_risk"]
    # 触发判定的帧：suspicious 及以上（含 high_risk）
    flagged = [r for r in valid if order[r["risk_level"]] <= order["suspicious"]]

    # 聚合理由：从触发帧里把判定依据拉出来，并标注来自哪一帧哪一秒
    reasons_agg = []
    for r in high_risk_frames + [x for x in flagged if x not in high_risk_frames]:
        head = f"[{r['frame_id']} @ {r['timestamp']}s · {r['risk_level']}]"
        if r["reasons"]:
            reasons_agg.append(head + " " + "；".join(r["reasons"]))
        else:
            reasons_agg.append(head + " 该帧被判为可疑/高风险（无详细理由）")

    if not valid:
        reasons_agg = ["所有采样帧都未能成功分析，无法给出视频级结论（见 error 帧）。"]
    elif not flagged:
        reasons_agg = ["所有采样帧均未触发可疑/高风险判定；但抽帧未覆盖到的画面"
                       "（尤其是快速闪过的编辑）仍可能含篡改，结论不可作为「视频干净」的保证。"]

    risk_breakdown = {lvl: 0 for lvl in RISK_LEVELS}
    for r in valid:
        risk_breakdown[r["risk_level"]] += 1

    result = {
        "schema": "beautyproof/video-analysis@1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "mode": "fast" if fast else "full",
        "video": video,
        "sampling": {"fps_sample": fps_sample, "max_frames": max_frames,
                     "interval": video["interval"], "sampled": len(meta["frames"])},
        "verdict": {
            "risk_level": agg_level,
            "reasons": reasons_agg,
            "flagged_frames": [r["frame_id"] for r in flagged],
            "high_risk_frames": [r["frame_id"] for r in high_risk_frames],
            "error_frames": [r["frame_id"] for r in errored],
            "tools_used": sorted({t for r in valid for t in r["tools_used"]}),
        },
        "risk_breakdown": risk_breakdown,
        "frames": frame_results,
        "tool_meta": getattr(pipeline, "TOOL_META", {}),
    }

    out = Path(out_path) if out_path else (
        REPO / "outputs" / f"analysis_video_{video['stem']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["_output_file"] = str(out)

    if verbose:
        print(f"\n=== 视频级结论：{agg_level} ===")
        print(f"  采样 {len(meta['frames'])} 帧："
              f"高风险 {risk_breakdown['high_risk']} / "
              f"可疑 {risk_breakdown['suspicious']} / "
              f"可信 {risk_breakdown['credible']} / "
              f"无法判定 {risk_breakdown['inconclusive']} / "
              f"错误 {len(errored)}")
        if flagged:
            print(f"  触发帧：{', '.join(r['frame_id'] for r in flagged)}")
        print(f"  用时 {result['elapsed_seconds']}s")
        print(f"  结果：{out}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="BeautyProof 视频模态：抽帧 → 逐帧跑图像取证 → 聚合视频级判定")
    ap.add_argument("--input", required=True, help="输入视频路径（mp4）")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径（默认 outputs/analysis_video_<名>.json）")
    ap.add_argument("--fast", action="store_true",
                    help="跳过 aigc/trufor 深度学习模型，秒级出结果（仅预览用）")
    ap.add_argument("--max-frames", type=int, default=30, help="最多抽帧数（默认 30）")
    ap.add_argument("--fps-sample", type=float, default=1.0, help="抽帧采样率，每秒几帧（默认 1.0）")
    args = ap.parse_args(argv)

    try:
        res = analyze_video(
            args.input, fast=args.fast, max_frames=args.max_frames,
            fps_sample=args.fps_sample, verbose=True, out_path=args.out)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[video_tool] 错误：{e}")
        return 1
    except KeyboardInterrupt:
        print("\n[video_tool] 已被用户中断")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
把 demo/video_render.html（自动播放 + 旁白字幕的渲染版走查页）录成 1080p 视频，
再转成 H.264 的 mp4（用 moviepy 自带的 ffmpeg 编码器，无需系统 ffmpeg）。

用法（venv python）：
    python tools/_record_video.py

产物：demo/BeautyProof_演示视频.mp4
"""
import sys
import time
from pathlib import Path

REPO = Path("C:/Users/Lenovo/WorkBuddy/黑客松/loreal-hackathon-beautyproof")
HTML = (REPO / "demo" / "video_render.html").as_uri()
VDIR = REPO / "outputs" / "_video"
VDIR.mkdir(parents=True, exist_ok=True)

RECORD_SEC = 165  # 7 个场景合计约 157s，多留几秒收尾


def record():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(VDIR),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = ctx.new_page()
        page.goto(HTML)
        print(f"录制中（{RECORD_SEC}s）…", flush=True)
        time.sleep(RECORD_SEC)
        ctx.close()
        browser.close()
    videos = sorted(VDIR.glob("*.webm"))
    if not videos:
        raise RuntimeError("没录到 webm 视频，请检查 Playwright / chromium 是否可用")
    return videos[-1]


def to_mp4(webm):
    from moviepy import VideoFileClip
    out = REPO / "demo" / "BeautyProof_演示视频.mp4"
    clip = VideoFileClip(str(webm))
    clip.write_videofile(str(out), codec="libx264", audio=False, fps=30,
                         preset="medium", logger=None)
    clip.close()
    return out


def main():
    webm = record()
    print(f"录到: {webm}", flush=True)
    out = to_mp4(webm)
    print(f"mp4 成品: {out}  ({out.stat().st_size/1024/1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

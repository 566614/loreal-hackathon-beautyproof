# -*- coding: utf-8 -*-
"""
把 demo 下的走查页录成 1080p mp4（H.264，用 moviepy 自带 ffmpeg，无需系统 ffmpeg）。

两个页面两种模式：
  walkthrough —— demo/video_walkthrough.html（11 屏，2:20 定速，**当前交付用的就是它**）
                 页面加载后不会自动播，需要点一下「▶ 自动播放」；每屏停留按页面内
                 DUR 数组（与 docs/演示视频脚本_v2.md 的口播时间轴一致）走完 11 屏后循环，
                 录制到 total 秒收尾，此时画面停在第 11 屏（结尾屏），观感完整。
  render     —— demo/video_render.html（旧版 7 屏渲染走查页，保留备用）

用法（venv python）：
    python tools/_record_video.py                     # 默认 walkthrough，145s
    python tools/_record_video.py --page render --sec 165
    python tools/_record_video.py --out demo/xxx.mp4 --sec 60

产物：demo/BeautyProof_演示视频_v2.mp4（walkthrough 模式默认）
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAGES = {
    "walkthrough": (REPO / "demo" / "video_walkthrough.html", True, 145),
    "render": (REPO / "demo" / "video_render.html", False, 165),
}
VDIR = REPO / "outputs" / "_video"


def record(html: Path, need_click: bool, seconds: int):
    from playwright.sync_api import sync_playwright
    VDIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(VDIR),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = ctx.new_page()
        page.goto(html.as_uri())
        page.wait_for_timeout(1500)
        if need_click:
            # 页面内触发，避免 Playwright 把鼠标动作录进画面
            page.evaluate("document.getElementById('play').click()")
        print(f"录制中（{seconds}s）…", flush=True)
        time.sleep(seconds)
        ctx.close()
        browser.close()
    videos = sorted(VDIR.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if not videos:
        raise RuntimeError("没录到 webm，请检查 Playwright / chromium 是否可用")
    return videos[-1]


def to_mp4(webm: Path, out: Path):
    from moviepy import VideoFileClip
    clip = VideoFileClip(str(webm))
    clip.write_videofile(str(out), codec="libx264", audio=False, fps=30,
                         preset="medium", logger=None)
    dur = clip.duration
    clip.close()
    return out, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", choices=list(PAGES), default="walkthrough")
    ap.add_argument("--sec", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    html, need_click, default_sec = PAGES[a.page]
    sec = a.sec or default_sec
    out = Path(a.out) if a.out else REPO / "demo" / f"BeautyProof_演示视频{'' if a.page == 'render' else '_v2'}.mp4"

    webm = record(html, need_click, sec)
    print(f"录到: {webm}", flush=True)
    path, dur = to_mp4(webm, out)
    print(f"mp4 成品: {path}  {dur:.1f}s  {path.stat().st_size/1024/1024:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

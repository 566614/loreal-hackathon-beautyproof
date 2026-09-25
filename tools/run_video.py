# -*- coding: utf-8 -*-
"""
run_video.py —— 视频模态接入「现有证据 / 报告格式」的连接器

它不做任何检测，只负责：
    1) 调用 video_tool.analyze_video（抽帧 + 逐帧复用图像流水线 + 聚合）；
    2) 生成一份与 report_generator 风格一致的视频级 Markdown 报告
       reports/report_video_<视频名>.md；
    3) 顺带把聚合结果 analysis_video_<名>.json 落盘（analyze_video 已写，
       这里保证存在即可）。

设计红线（与任务约束一致）：
    · 不改写 pipeline.py 的图像主路径 / analyze()；
    · 检测逻辑全部复用现有图像流水线，本报告同样「每句话都能追溯到 outputs/ 证据」；
    · 帧证据沿用统一格式（source_asset_id = 帧文件名），不另起一套。

用法：
    python tools/run_video.py --input demo/BeautyProof_演示视频.mp4
    python tools/run_video.py --input xxx.mp4 --fast
    python tools/run_video.py --input xxx.mp4 --max-frames 20 --fps-sample 2.0
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

REPORT_DIR = REPO / "reports"

RISK_TEXT = {
    "high_risk": ("高风险", "视频中检出很可能有问题的帧，**不建议直接采信**，必须人工复核。"),
    "suspicious": ("可疑", "视频中发现可疑帧，**建议人工重点查看触发帧**。"),
    "credible": ("可信", "采样帧未触发可疑判定，但仍受抽帧覆盖率限制（见下方局限）。"),
    "inconclusive": ("无法判定", "采样帧未给出明确结论，或逐帧分析失败较多，无法定级。"),
}

ADVICE = {
    "high_risk": "1) 不要直接采信该视频画面；2) 定位触发帧（见下方逐帧明细）人工核验；"
                 "3) 向发布方索取带 C2PA 凭证的原始素材比对。",
    "suspicious": "1) 人工放大查看触发帧的标注区域；2) 尽量拿到原始未编辑版本比对；"
                  "3) 结合视频来源综合判断。",
    "credible": "1) 当前采样未发现可疑帧；2) 因抽帧有覆盖率上限，建议对关键片段提高采样率复检。",
    "inconclusive": "1) 不能据此认定视频造假，也不能据此认定真实；"
                   "2) 提高采样率或改用 full 模式（含深度学习模型）复检。",
}


def build_video_report(result):
    """把视频级分析结果拼成中文报告（模板拼装，结论可追溯到逐帧证据）。"""
    v = result["video"]
    verdict = result["verdict"]
    risk = verdict["risk_level"]
    zh_level, zh_desc = RISK_TEXT.get(risk, ("无法判定", ""))
    bd = result.get("risk_breakdown", {})

    lines = []
    lines.append(f"# 视频鉴定报告：{v['name']}")
    lines.append("")
    lines.append(f"- 检测对象：`{v['name']}`")
    lines.append(f"- 时长：{v['duration_seconds']}s ｜ 总帧数：{v['total_frames']} ｜ 原始 fps：{v['fps']}")
    lines.append(f"- 采样：每 {result['sampling']['interval']} 帧取 1，共抽 "
                 f"{result['sampling']['sampled']} 帧（上限 {result['sampling']['max_frames']}）")
    lines.append(f"- 模式：{'快速预览（跳过深度学习模型）' if result['mode'] == 'fast' else '完整（含 AIGC / TruFor）'}")
    lines.append(f"- 参与检测的工具：{len(verdict.get('tools_used', []))} 类（逐帧复用）")
    lines.append(f"- **视频级鉴定结论：{zh_level}（{risk}）**")
    lines.append("")

    lines.append("## 一、结论")
    lines.append("")
    lines.append(zh_desc)
    if verdict["reasons"]:
        lines.append("")
        lines.append("聚合判定依据（来自触发帧，标注了帧号与时间点）：")
        for r in verdict["reasons"]:
            lines.append(f"- {r}")
    if verdict["high_risk_frames"]:
        lines.append("")
        lines.append(f"⚠ **高风险帧**：{', '.join(verdict['high_risk_frames'])}")
    if verdict["flagged_frames"]:
        lines.append(f"触发判定的帧（可疑及以上）：{', '.join(verdict['flagged_frames'])}")
    lines.append("")

    lines.append("## 二、逐帧明细")
    lines.append("")
    lines.append("| 帧号 | 时间点(s) | 风险等级 | 触发理由摘要 |")
    lines.append("|------|----------|----------|--------------|")
    for fr in result.get("frames", []):
        lvl = fr["risk_level"]
        z, _ = RISK_TEXT.get(lvl, (lvl, ""))
        reason = "；".join(fr.get("reasons", []))[:60] or "—"
        lines.append(f"| {fr['frame_id']} | {fr['timestamp']} | {z} | {reason} |")
    lines.append("")

    lines.append("## 三、各档统计")
    lines.append("")
    lines.append(f"- 高风险（high_risk）：{bd.get('high_risk', 0)} 帧")
    lines.append(f"- 可疑（suspicious）：{bd.get('suspicious', 0)} 帧")
    lines.append(f"- 可信（credible）：{bd.get('credible', 0)} 帧")
    lines.append(f"- 无法判定（inconclusive）：{bd.get('inconclusive', 0)} 帧")
    if verdict.get("error_frames"):
        lines.append(f"- 分析失败（error）：{len(verdict['error_frames'])} 帧 "
                     f"（{', '.join(verdict['error_frames'])}）")
    lines.append("")

    lines.append("## 四、建议")
    lines.append("")
    lines.append(ADVICE.get(risk, ADVICE["inconclusive"]))
    lines.append("")

    lines.append("## 五、能力边界（诚实声明）")
    lines.append("")
    lines.append("- **抽帧覆盖率**：只分析采样帧，快速闪过（< 采样间隔）的编辑可能漏检；"
                 "对剪辑密集的视频请提高 `--fps-sample` 或 `--max-frames`。")
    lines.append("- **CPU 速度**：full 模式每帧都要加载 AIGC / TruFor 模型，单视频可能需数分钟；"
                 "预览用 `--fast`，正式复核再跑 full。")
    lines.append("- **帧证据复用统一格式**：每帧的逐工具证据写在 `outputs/<tool>_frame_*.json`，"
                 "source_asset_id 即帧文件名，与图片流水线完全一致，可单独追溯。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由模板从逐帧工具证据 JSON 拼装，结论可追溯到 `outputs/` 下的帧证据文件；"
                 "算法结论不是法律鉴定意见。*")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="BeautyProof 视频模态报告连接器")
    ap.add_argument("--input", required=True, help="输入视频路径（mp4）")
    ap.add_argument("--fast", action="store_true", help="跳过深度学习模型（预览）")
    ap.add_argument("--max-frames", type=int, default=30, help="最多抽帧数（默认 30）")
    ap.add_argument("--fps-sample", type=float, default=1.0, help="抽帧采样率（默认 1.0）")
    ap.add_argument("--report-only", action="store_true",
                    help="只根据已有的 analysis_video_<名>.json 重新生成报告，不重跑分析")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(TOOLS))
    import video_tool  # noqa: E402

    out_json = REPO / "outputs" / f"analysis_video_{Path(args.input).stem}.json"

    if args.report_only and out_json.exists():
        result = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        try:
            result = video_tool.analyze_video(
                args.input, fast=args.fast, max_frames=args.max_frames,
                fps_sample=args.fps_sample, verbose=True)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[run_video] 错误：{e}")
            return 1

    report = build_video_report(result)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rep_path = REPORT_DIR / f"report_video_{result['video']['stem']}.md"
    rep_path.write_text(report, encoding="utf-8")

    print(f"\n视频级报告：{rep_path}")
    print(f"聚合结果 JSON：{result.get('_output_file')}")
    print(f"视频级结论：{result['verdict']['risk_level']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

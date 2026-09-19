# -*- coding: utf-8 -*-
"""
AIGC 工具 —— 判断一张图是不是 AI 生成的，输出统一格式的证据 JSON

用法：
    python tools/aigc_tool.py <图片路径>

大白话：
    用开源模型 Organika/sdxl-detector 扫一遍图，
    它给一个 0~1 的分数：越接近 1，越像 AI 生成的图。

能证明什么：
    这张图「像不像 AI 生成的」。分数高 → 大概率是用 AI 画的/生成的。

不能证明什么（铁律）：
    模型会误判。分数只是「模型觉得」，不是「一定的事实」。
    真人画的也可能被误判成 AI，AI 生成的也可能蒙混过关。
"""
import json
import sys
from pathlib import Path


def run_aigc(image_path):
    """调 transformers 的 image-classification pipeline，判断是不是 AI 图

    返回: (ai_score, ai_label, available)
        available=False 表示模型没下下来 / 加载失败 —— 此时 ai_score 为 None，
        调用方应当按「无法判断」处理，而不是当成真的 0 分。
    """
    from transformers import pipeline

    try:
        # local_files_only=True：模型必须已在本地缓存（由 download_aigc_model.py 下好），
        # 不在就立刻报错降级，绝不自己联网下载（默认网络会被掐断，重试退避会卡死整条流水线）
        detector = pipeline(
            "image-classification",
            model="Organika/sdxl-detector",
            local_files_only=True,
        )
    except Exception as e:  # noqa: BLE001
        # 模型没下载成功 / 网络断了 / 缺依赖 —— 优雅降级，别让整条流水线崩
        return None, f"model_unavailable: {type(e).__name__}", False

    try:
        results = detector(str(image_path))
    except Exception as e:  # noqa: BLE001
        return None, f"inference_failed: {type(e).__name__}", False

    ai_score = None
    ai_label = None
    for r in results:
        label = r["label"].lower()
        if any(k in label for k in ["ai", "artificial", "fake", "generated", "synthetic"]):
            ai_score = float(r["score"])
            ai_label = r["label"]
            break

    if ai_score is None:
        # 没找到明确的 ai 标签，就用「非真实」来近似
        ai_score = 1.0 - float(results[0]["score"]) if len(results) == 2 else 0.0
        ai_label = "not_ai_label_found"

    return round(ai_score, 4), ai_label, True


def build_evidence(image_path, ai_score, ai_label, available=True):
    if not available:
        return {
            "tool": "aigc",
            "source_asset_id": Path(image_path).name,
            "observed": "AIGC 检测暂不可用（模型未下载成功 / 加载失败）",
            "cannot_prove": "模型未就绪，无法给出 AI 生成概率；本项按「无法判断」处理，不影响其他工具结论",
            "evidence": [
                {"aigc_score": None, "label": ai_label, "available": False},
            ],
        }
    if ai_score >= 0.7:
        verdict = "很可能是 AI 生成的图"
    elif ai_score >= 0.4:
        verdict = "有部分 AI 生成的痕迹，需人工复核"
    else:
        verdict = "看起来像真实拍摄/人工制作的图"
    return {
        "tool": "aigc",
        "source_asset_id": Path(image_path).name,
        "observed": f"AI 生成概率 {ai_score}（{verdict}）",
        "cannot_prove": "模型会误判：真人也常被误判为 AI，AI 生成的也可能蒙混过关；分数不是事实",
        "evidence": [
            {"aigc_score": ai_score, "label": ai_label, "available": True},
        ],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/aigc_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    ai_score, ai_label, available = run_aigc(image_path)
    report = build_evidence(image_path, ai_score, ai_label, available)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"aigc_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

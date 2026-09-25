# -*- coding: utf-8 -*-
"""
跨生成器零样本验证 —— v3 AIGC 模型对 Midjourney / Stable Diffusion / Flux 生成图的鲁棒性实测

用途：
    堵死评委「换个生成器就不灵」的质疑。
    用 v3（models/beautyproof_aigen_v3，MODEL_PRIORITY 置顶）对 MJ/SD/Flux 真图做
    零样本（不训练、只推理）验证，统计各生成器的 AI 召回率。

输入约定（队友负责）：
    data/ai_cross/
        mj/      ← Midjourney 图（2~3 张）
        sd/      ← Stable Diffusion 图（2~3 张）
        flux/    ← Flux 图（2~3 张）
    子文件夹名用于推断生成器；识别规则（不区分大小写、支持常见别名）：
        mj / midjourney            → Midjourney
        sd / stable-diffusion      → Stable Diffusion
        flux                       → Flux

判定口径：
    ai_prob = aigc_tool.run_aigc 给出的「整图由 AI 生成」概率（0~1）。
    ai_prob >= 0.5 记为「命中 AI」；>= 0.8 记为「高置信命中」。
    各生成器召回 = 该文件夹下命中张数 / 总张数。

健壮性：
    若 data/ai_cross/ 下没有任何 mj/sd/flux 子文件夹（或子文件夹为空），
    打印 "no images found, skipping run" 并正常退出（exit 0），不报错、不假跑。

运行（务必用项目 venv python）：
    C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe tools/_validate_crossgen.py
    # 可选：再跑一遍完整流水线拿综合判定（含 TruFor，较慢）
    ...python.exe tools/_validate_crossgen.py --full

输出：
    控制台打印表格 + 各生成器召回率；
    results/crossgen_validation.json 落盘（含每张图的 ai_prob / 判定）。
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

# 子文件夹名 → 展示用的生成器名（不区分大小写匹配）
GENERATOR_KEYS = {
    "mj": "Midjourney",
    "midjourney": "Midjourney",
    "sd": "Stable Diffusion",
    "stable-diffusion": "Stable Diffusion",
    "stable_diffusion": "Stable Diffusion",
    "flux": "Flux",
}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

HIT = 0.5      # 命中阈值
HIGH = 0.8     # 高置信阈值


def find_generators(input_dir):
    """扫描 input_dir，返回 {生成器展示名: Path(子文件夹)} 仅含确实存在且有图的文件夹"""
    found = {}
    if not input_dir.exists():
        return found
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        key = sub.name.lower()
        if key in GENERATOR_KEYS:
            imgs = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if imgs:
                found[GENERATOR_KEYS[key]] = (sub, sorted(imgs))
    return found


def classify(ai_prob):
    if ai_prob is None:
        return "无法判断"
    if ai_prob >= HIGH:
        return "AI生成(高置信)"
    if ai_prob >= HIT:
        return "AI生成(命中)"
    return "真实/未命中"


def run():
    ap = argparse.ArgumentParser(description="v3 跨生成器零样本验证")
    ap.add_argument("--input", default=str(REPO / "data" / "ai_cross"),
                    help="输入根目录，默认 data/ai_cross/")
    ap.add_argument("--full", action="store_true",
                    help="额外调用 pipeline.analyze(fast=False) 拿综合判定（含 TruFor，较慢）")
    ap.add_argument("--out", default=str(REPO / "results" / "crossgen_validation.json"),
                    help="结果落盘路径")
    args = ap.parse_args()

    input_dir = Path(args.input)
    gens = find_generators(input_dir)

    if not gens:
        print("no images found, skipping run")
        print("（未检测到 data/ai_cross/ 下的 mj/ sd/ flux/ 子文件夹或其中无图。）")
        print("请让队友把图分别放入 data/ai_cross/mj/、sd/、flux/ 后重跑本脚本。")
        return 0

    # 加载模型（复用 aigc_tool，自动选 v3）
    try:
        import aigc_tool  # noqa: E402
    except Exception as e:  # noqa: BLE001
        print(f"model load error: 无法导入 aigc_tool —— {type(e).__name__}: {e}")
        return 1

    pipeline = None
    if args.full:
        try:
            import pipeline as _pl  # noqa: E402
            pipeline = _pl
        except Exception as e:  # noqa: BLE001
            print(f"[--full] pipeline 不可用，跳过综合判定：{type(e).__name__}: {e}")

    print("=" * 78)
    print("BeautyProof v3 跨生成器零样本验证")
    print(f"模型：{aigc_tool.model_path()}")
    print(f"输入：{input_dir}")
    print("=" * 78)
    print(f"{'文件名':<48}{'生成器':<18}{'ai_prob':<10}{'判定'}")
    print("-" * 78)

    rows = []          # 明细
    summary = {}       # 每生成器统计
    for gen_name, (sub, imgs) in gens.items():
        hit = high = total = 0
        for p in imgs:
            ai_prob, label, available = aigc_tool.run_aigc(str(p))
            total += 1
            verdict = classify(ai_prob)
            if available and ai_prob is not None:
                if ai_prob >= HIT:
                    hit += 1
                if ai_prob >= HIGH:
                    high += 1
            # 可选：综合判定
            risk = None
            if pipeline is not None:
                try:
                    res = pipeline.analyze(str(p), fast=False, verbose=False)
                    risk = res.get("verdict", {}).get("risk_level")
                except Exception:  # noqa: BLE001
                    risk = "pipeline_error"
            rows.append({
                "file": p.name, "generator": gen_name,
                "ai_prob": ai_prob, "label": label,
                "available": available, "verdict": verdict,
                "pipeline_risk": risk,
            })
            print(f"{p.name[:46]:<48}{gen_name:<18}"
                  f"{(str(ai_prob) if ai_prob is not None else 'N/A'):<10}{verdict}")
        recall = (hit / total) if total else None
        high_recall = (high / total) if total else None
        summary[gen_name] = {
            "total": total, "hit": hit, "high": high,
            "recall_ge_0.5": round(recall, 4) if recall is not None else None,
            "high_conf_ge_0.8": round(high_recall, 4) if high_recall is not None else None,
        }

    print("-" * 78)
    print("各生成器召回率：")
    for gen_name, s in summary.items():
        r = s["recall_ge_0.5"]
        print(f"  {gen_name:<18} 召回(>=0.5) = {r if r is not None else 'N/A'} "
              f"({s['hit']}/{s['total']})  高置信(>=0.8) = {s['high_conf_ge_0.8']} ({s['high']}/{s['total']})")

    out = {
        "schema": "beautyproof/crossgen_validation@1",
        "model": aigc_tool.model_path(),
        "zero_shot": True,
        "note": "零样本跨生成器验证：v3 未针对 MJ/SD/Flux 训练，仅推理",
        "threshold_hit": HIT,
        "threshold_high": HIGH,
        "generators": summary,
        "details": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(run())

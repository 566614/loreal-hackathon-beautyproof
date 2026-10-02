# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队双检测器级联评测
"""
双检测器级联评测（cascade eval）—— AIGC 本域微调模型 × 频域工具

用法：
    python tools/cascade_eval.py
    python tools/cascade_eval.py --refit-spectral   # 顺带重标定频域

要回答的一个问题（也是答辩上最硬的那一句）：
    **在「自动下架误伤保持 0」不变的前提下，级联能把 AI 图的捕获率提高多少？**

为什么要级联（本项目的核心工程判断）：
    两条证据线彼此独立、失效模式不同：
      AIGC   = 本域微调 CNN，看的是「这张图的纹理统计像不像我们见过的 AI 美妆图」
               → 弱项：**跨生成器**。换到没见过的生成器/风格就掉（实测跨生成器召回仅 66.7%）。
      频域   = FFT 频谱统计，看的是「高频能量/频谱规整度像不像相机直出」
               → 强项：生成器无关（不关心是谁画的），弱项：**重压/强锐化真实图会误报**。
    单用任何一条都有系统性盲区；级联的意义是**用一条的长处补另一条的短处**，
    并且只有在两条一致时才自动下架 —— 这正是官方点名的「分级风险策略」。

铁律：
    - 模型必须 pin 到 beautyproof_aigen_v4 并**断言实际解析到的目录就是它**，
      否则静默回退到旧模型会输出一份"看起来通过"的假结果（9/26 踩过这个坑）。
    - 冻结图（从未参与 v1~v7 训练）才进评测；缺失文件记 missing，不进分母。
    - 自动下架（high_risk）的误伤数是本项目的红线指标，必须显式打印。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aigc_tool  # noqa: E402
import numpy as np  # noqa: E402
import spectral_calib as sc  # noqa: E402
from spectral_tool import extract_features, load_calib  # noqa: E402

PRODUCTION_MODEL = "beautyproof_aigen_v4"

# 频域双阈值（与 AIGC 的三档同理，数值由 results/spectral_calib.json 标定，
#  这里只定义"哪个算强信号/哪个算灰带"）
SPEC_STRONG = 0.8   # TEST 段：AI 召回 85% / 真实误报 0/10
SPEC_GRAY = 0.6     # TEST 段：AI 召回 90% / 真实误报 1/10

# AIGC 档位沿用 rule_engine.THRESHOLDS（不在这里另立一套）
AIGC_HARD_HIGH = 0.99
AIGC_HIGH_RISK = 0.9


def pin_model():
    """pin 生产模型并断言实际解析目录一致。"""
    aigc_tool.MODEL_PRIORITY = [PRODUCTION_MODEL] + [
        m for m in aigc_tool.MODEL_PRIORITY if m != PRODUCTION_MODEL
    ]
    p = aigc_tool.model_path()
    resolved = Path(p).resolve().name
    if resolved != PRODUCTION_MODEL:
        raise SystemExit(
            f"[FAIL] 模型 pin 失败：要 {PRODUCTION_MODEL}，实际解析到 {resolved}。"
            "拒绝继续（否则会输出一份基于旧模型的假评测）。"
        )
    print(f"[OK] 模型已 pin: {resolved}  ({p})")
    return p


def aigc_score(path):
    """返回 (分数, available, 标签)。分数 None 表示模型未就绪。

    注意：run_aigc 本身就返回 (ai_score, ai_label, available) 三元组，
    不要再拿它的结果去喂 ai_prob_from（那个函数是给 transformers 通用模型
    的 list[dict] 输出用的，喂三元组会报 'float' object is not subscriptable）。
    """
    try:
        res = aigc_tool.run_aigc(str(path))
    except Exception as e:
        print(f"   !! aigc 推理失败 {Path(path).name}: {e}")
        return None, False, "inference_error"
    score, label, ok = res
    return (float(score) if score is not None else None), bool(ok), str(label)


def build_table(repo_root):
    ai_rows, m1 = sc.collect(repo_root, ["data/ai_cross", "data/ai_cross_native", "data/ai_mj_sd_flux_heldout/flux"], "frozen_ai")
    real_rows, m2 = sc.collect(repo_root, ["data/realworld_heldout"], "frozen_real")
    fs_rows, m3 = sc.collect_fs(repo_root, "heldout_never_train")
    real_rows += fs_rows

    calib = load_calib(repo_root)
    if not calib:
        raise SystemExit("[FAIL] 未找到 results/spectral_calib.json，先跑 tools/spectral_calib.py")

    table = []
    missing = list(m1) + list(m2) + list(m3)
    for rows, kind in ((ai_rows, "ai"), (real_rows, "real")):
        for r in rows:
            rel = r["file"]
            src = repo_root / rel.split("/", 1)[-1] if False else None
            # 从 file 字段还原真实路径
            if rel.startswith("frozen_ai/"):
                sub = rel.split("/", 2)
                path = repo_root / "data" / sub[1] / sub[2]
            elif rel.startswith("frozen_real/"):
                sub = rel.split("/", 2)
                path = repo_root / "data" / sub[1] / sub[2]
            else:
                seg, fn = rel.split("/", 1)
                path = repo_root / "data" / "real_filter_selfie" / fn
            if not path.exists():
                missing.append({"file": rel, "note": "路径还原失败/文件不存在"})
                continue
            a_s, a_ok, a_lbl = aigc_score(path)
            sp_s, _ = sc_score(r["feat"], calib)
            table.append({
                "file": rel, "kind": kind,
                "aigc": a_s, "aigc_available": a_ok, "aigc_label": a_lbl,
                "spectral": sp_s,
                "hf_ratio": r["feat"]["hf_energy_ratio"],
                "flatness": r["feat"]["spectral_flatness"],
            })
    return table, missing


def sc_score(feat, calib):
    from spectral_tool import score_from_features
    s, _ = score_from_features(feat, calib)
    return s, None


def verdict(cfg, row):
    """按配置返回 'high_risk' / 'suspicious' / 'inconclusive'。"""
    a, s = row["aigc"], row["spectral"]
    if cfg == "aigc_only":
        if a is None:
            return "inconclusive"
        if a >= AIGC_HARD_HIGH:
            return "high_risk"
        if a >= AIGC_HIGH_RISK:
            return "suspicious"
        return "inconclusive"
    if cfg == "spectral_only":
        if s >= SPEC_STRONG:
            return "high_risk"
        if s >= SPEC_GRAY:
            return "suspicious"
        return "inconclusive"
    if cfg == "cascade_strict":
        # 只有两条证据都指向 AI 才自动下架
        if a is not None and a >= AIGC_HARD_HIGH and s >= SPEC_GRAY:
            return "high_risk"
        if a is not None and a >= AIGC_HARD_HIGH and s < SPEC_GRAY:
            # AIGC 极高但频域不支持：仍保 high_risk（不丢原有召回），但记为分歧
            return "high_risk"
        if (a is not None and a >= AIGC_HIGH_RISK and s >= SPEC_STRONG):
            return "high_risk"
        if (a is not None and a >= AIGC_HIGH_RISK) or s >= SPEC_GRAY:
            return "suspicious"
        return "inconclusive"
    raise ValueError(cfg)


def evaluate(cfg, table):
    ai = [r for r in table if r["kind"] == "ai"]
    real = [r for r in table if r["kind"] == "real"]
    n_ai = len(ai)
    n_real = len(real)
    if n_ai == 0 or n_real == 0:
        return {"config": cfg, "note": "样本缺失", "n_ai": n_ai, "n_real": n_real}

    ai_hr = [r for r in ai if verdict(cfg, r) == "high_risk"]
    ai_sus = [r for r in ai if verdict(cfg, r) == "suspicious"]
    ai_cap = [r for r in ai if verdict(cfg, r) in ("high_risk", "suspicious")]
    real_hr = [r for r in real if verdict(cfg, r) == "high_risk"]
    real_sus = [r for r in real if verdict(cfg, r) == "suspicious"]
    real_cap = [r for r in real if verdict(cfg, r) in ("high_risk", "suspicious")]

    return {
        "config": cfg,
        "n_ai": n_ai, "n_real": n_real,
        "ai_auto_takedown": len(ai_hr), "ai_auto_takedown_rate": round(len(ai_hr) / n_ai, 4),
        "ai_captured": len(ai_cap), "ai_capture_rate": round(len(ai_cap) / n_ai, 4),
        "real_auto_takedown_FP": len(real_hr), "real_fp_rate": round(len(real_hr) / n_real, 4),
        "real_flagged": len(real_cap), "real_flag_rate": round(len(real_cap) / n_real, 4),
        "ai_missed": [r["file"] for r in ai if verdict(cfg, r) == "inconclusive"],
        "real_FP_files": [r["file"] for r in real if verdict(cfg, r) == "high_risk"],
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    if "--refit-spectral" in sys.argv:
        print("== 重新标定频域 ==")
        sc.main()

    pin_model()
    print("== 逐图打分（AIGC v4 + 频域）==")
    table, missing = build_table(repo_root)
    print(f"   入表 {len(table)} 张（AI {sum(1 for r in table if r['kind']=='ai')} / "
          f"REAL {sum(1 for r in table if r['kind']=='real')}），缺失 {len(missing)}")

    cfgs = ["aigc_only", "spectral_only", "cascade_strict"]
    res = [evaluate(c, table) for c in cfgs]

    print("\n" + "=" * 96)
    print(f"{'配置':16s} {'AI自动下架':>12s} {'AI捕获(≥可疑)':>14s} {'真实误报(自动下架)':>20s} {'真实被标可疑':>14s}")
    print("-" * 96)
    for r in res:
        print(f"{r['config']:16s} "
              f"{r['ai_auto_takedown']:3d}/{r['n_ai']:<3d}={r['ai_auto_takedown_rate']:<6.3f} "
              f"{r['ai_captured']:4d}/{r['n_ai']:<3d}={r['ai_capture_rate']:<6.3f} "
              f"{r['real_auto_takedown_FP']:6d}/{r['n_real']:<3d}={r['real_fp_rate']:<6.3f}  "
              f"{r['real_flagged']:5d}/{r['n_real']:<3d}={r['real_flag_rate']:<6.3f}")
    print("=" * 96)

    base = res[0]
    casc = res[-1]
    d_cap = casc["ai_capture_rate"] - base["ai_capture_rate"]
    d_fp = casc["real_fp_rate"] - base["real_fp_rate"]
    print(f"\n级联 vs 仅AIGC：AI 捕获率 {base['ai_capture_rate']:.3f} → {casc['ai_capture_rate']:.3f} "
          f"（{d_cap:+.3f}）；自动下架误报 {base['real_fp_rate']:.3f} → {casc['real_fp_rate']:.3f}（{d_fp:+.3f}）")
    print(f"自动下架红线：仅AIGC 误伤 {base['real_auto_takedown_FP']} 张 / 级联 {casc['real_auto_takedown_FP']} 张"
          f"（n_real={casc['n_real']}）")

    out = {
        "schema": "beautyproof/cascade_eval@1",
        "production_model": PRODUCTION_MODEL,
        "model_pin_verified": True,
        "spectral_calib": "results/spectral_calib.json",
        "thresholds": {
            "aigc_hard_high": AIGC_HARD_HIGH, "aigc_high_risk": AIGC_HIGH_RISK,
            "spectral_strong": SPEC_STRONG, "spectral_gray": SPEC_GRAY,
        },
        "n_ai": len([r for r in table if r["kind"] == "ai"]),
        "n_real": len([r for r in table if r["kind"] == "real"]),
        "results": res,
        "headline": {
            "ai_capture_rate_aigc_only": base["ai_capture_rate"],
            "ai_capture_rate_cascade": casc["ai_capture_rate"],
            "delta_capture": round(d_cap, 4),
            "real_fp_rate_aigc_only": base["real_fp_rate"],
            "real_fp_rate_cascade": casc["real_fp_rate"],
            "delta_fp": round(d_fp, 4),
            "auto_takedown_fp_count": casc["real_auto_takedown_FP"],
        },
        "per_image": table,
        "missing": missing,
        "integrity": "冻结图（从未参与 v1~v7 训练）；模型已 pin 并断言；缺失文件不进分母",
    }
    p = repo_root / "results" / "cascade_eval.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

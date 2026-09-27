# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 数据集→识别→决策 闭环评测入口（赛题 S3）
"""
把多个数据集用统一的 pipeline.analyze() 跑成一份闭环评测报告。

这是赛题 S3 的「闭环评测入口」：把「数据集 → 识别 → 决策 → 处置」串成一份
可对外的量化报告，回答评分对照表的两件事：
  1) 多模态检测是否识别出真伪痕迹（risk_level 分布 + 方向命中率）
  2) Agent 是否给出风险预警与处置建议（disposition 处置建议分布）

用法：
    python tools/run_dataset_closedloop.py                # 全量跑（非 --fast）
    python tools/run_dataset_closedloop.py --fast         # 跳过 aigc+trufor，仅演示链路（≠真实指标）
    python tools/run_dataset_closedloop.py --sample 12    # 大模型集最多抽 12 张（真实/held-out 集始终全量）
    python tools/run_dataset_closedloop.py --datasets ai realworld_heldout   # 只跑指定数据集

口径纪律（项目铁律）：
    - 真实误报率（对外只报这一个"真实鲁棒性"数）= data/realworld_heldout 的误报率，
      该集永不参与训练。受控集（clean/tampered/ai）是真值已知的小样本，只证明「链路能判对」。
    - ai_cross_native 是跨生成器零样本召回：本系统只在「同生成器族」上做过微调，
      非 MJ/SD/Flux 族的外推能力仅到此，明确标注，不夸大。
    - 默认跑全量 aigc+trufor（非 --fast）。--fast 仅用于秒级演示链路，不用于对外指标。

产出：results/closedloop_eval.json
    - 每个数据集：n、四档风险分布、方向命中率（命中期望档）
    - realworld_heldout 的 误报率（真实图被误报为有问题）
    - ai_cross_native 的 零样本召回率
    - 全部样本的 处置建议分布（聚合 planner.decide_actions 的 disposition 计数）
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
RESULTS = REPO / "results"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".webp")

# 把 tools/ 加进 path，复用现成的 pipeline.analyze()（任务硬性要求：复用现有 pipeline）
sys.path.insert(0, str(TOOLS))
from pipeline import analyze  # noqa: E402

# 风险四档（固定顺序，与 rule_engine.RISK_LEVELS 保持一致）
RISK_LEVELS = ["high_risk", "suspicious", "credible", "inconclusive"]
ALARM = {"suspicious", "high_risk"}     # 判为「有问题」
CLEAN = {"credible", "inconclusive"}    # 判为「没抓到问题」

# 数据集注册表：
#   ground_truth  —— 该集真值
#   expect        —— 期望命中的风险档集合（用于算「方向命中率」）
#   bucket        —— controlled（受控）/ heldout_real（真 held-out 真实图）/ zero_shot_cross（跨生成器零样本）
#   always_full   —— 该集绝不抽样（真实误报率与零样本召回率必须全量才有意义）
DATASETS = {
    "clean": {
        "dir": "data/clean", "ground_truth": "clean",
        "expect": {"inconclusive", "credible"}, "bucket": "controlled",
        "always_full": False,
        "note": "受控真图（相机拍/自产实拍）。期望不误报（inconclusive/credible）。",
    },
    "tampered": {
        "dir": "data/tampered", "ground_truth": "tampered",
        "expect": {"suspicious", "high_risk"}, "bucket": "controlled",
        "always_full": False,
        "note": "受控篡改图（copy_move/splice/文字改写）。期望被报警（suspicious/high_risk）。",
    },
    "ai": {
        "dir": "data/ai", "ground_truth": "ai",
        "expect": {"high_risk"}, "bucket": "controlled",
        "always_full": False,
        "note": "受控 AI 生成图（即梦等本域微调覆盖族）。期望 high_risk（整图 AI 生成）。",
    },
    "realworld_heldout": {
        "dir": "data/realworld_heldout", "ground_truth": "real",
        "expect": {"inconclusive", "credible"}, "bucket": "heldout_real",
        "always_full": True,
        "note": "真 held-out 真实图（取自真实世界采集，绝不参与训练）。期望不误报；"
                "误报率=真实鲁棒性唯一对外口径。",
    },
    "ai_cross_native": {
        "dir": "data/ai_cross_native", "ground_truth": "ai",
        "expect": {"high_risk"}, "bucket": "zero_shot_cross",
        "always_full": True,
        "note": "跨生成器零样本集（同生成器族、非 MJ/SD/Flux 本域微调覆盖族）。"
                "算零样本召回；能力仅到此族，不夸大泛化。",
    },
}


def collect(folder: Path, sample=None, always_full=False):
    """列出目录下图片；超大集可抽样（真实/held-out 集始终全量）"""
    if not folder.exists():
        return []
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if sample and not always_full and len(imgs) > sample:
        return imgs[:sample]
    return imgs


def extract_key_scores(evidence):
    """从 evidence 抽关键分数（对齐 v4_eval.json 的 key_scores 口径）"""
    aigc = (evidence.get("aigc") or {}).get("evidence", [{}])[0]
    tru = (evidence.get("trufor") or {}).get("evidence", [{}])[0]
    ela = (evidence.get("ela") or {}).get("evidence", [{}])[0]
    ks = {}
    if aigc:
        if aigc.get("aigc_score") is not None:
            ks["aigc_score"] = round(aigc["aigc_score"], 4)
        if aigc.get("model"):
            ks["aigc_model"] = aigc["model"]
        if aigc.get("available") is not None:
            ks["aigc_available"] = bool(aigc["available"])
    if tru:
        if tru.get("trufor_score") is not None:
            ks["trufor_score"] = round(tru["trufor_score"], 4)
        if tru.get("tampered_area_ratio") is not None:
            ks["trufor_area_ratio"] = round(tru["tampered_area_ratio"], 4)
        if tru.get("available") is not None:
            ks["trufor_available"] = bool(tru["available"])
    reg = (ela.get("suspicious_regions") or [])
    if reg and reg[0].get("ela_score") is not None:
        ks["ela_top_score"] = round(reg[0]["ela_score"], 4)
    return ks


def run_one(image_path, fast, verbose, from_outputs=False):
    """对一张图跑完整 pipeline，规范化成闭环评测的一行

    from_outputs=True 时：不重跑 pipeline，直接读已落盘的
    outputs/analysis_<stem>.json（用于分批跑完后再聚合，避免重复触发
    环境的批量删除保护）。
    """
    if from_outputs:
        out = REPO / "outputs" / f"analysis_{image_path.stem}.json"
        if not out.exists():
            return {
                "file": image_path.name, "risk_level": "error",
                "hit": False, "key_scores": {}, "dispositions": [],
                "error": "outputs 中无该图分析结果（需先跑 pipeline.analyze）",
            }
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            return {
                "file": image_path.name, "risk_level": "error",
                "hit": False, "key_scores": {}, "dispositions": [],
                "error": f"读分析 JSON 失败：{type(e).__name__}: {e}",
            }
        risk = data.get("verdict", {}).get("risk_level", "error")
        evidence = data.get("evidence", {}) or {}
        actions = (data.get("agent") or {}).get("actions", []) or []
        dispositions = [a.get("action") for a in actions if a.get("action")]
        return {
            "file": image_path.name,
            "risk_level": risk,
            "hit": None,  # 由调用方按 expect 填
            "key_scores": extract_key_scores(evidence),
            "dispositions": dispositions,
            "reasons": data.get("verdict", {}).get("reasons", []),
        }
    try:
        res = analyze(str(image_path), fast=fast, verbose=verbose)
    except Exception as e:  # noqa: BLE001
        return {
            "file": image_path.name, "risk_level": "error",
            "hit": False, "key_scores": {}, "dispositions": [],
            "error": f"{type(e).__name__}: {e}",
        }
    risk = res.get("verdict", {}).get("risk_level", "error")
    evidence = res.get("evidence", {}) or {}
    actions = (res.get("agent") or {}).get("actions", []) or []
    dispositions = [a.get("action") for a in actions if a.get("action")]
    return {
        "file": image_path.name,
        "risk_level": risk,
        "hit": None,  # 由调用方按 expect 填
        "key_scores": extract_key_scores(evidence),
        "dispositions": dispositions,
        "reasons": res.get("verdict", {}).get("reasons", []),
    }


def evaluate_dataset(name, cfg, fast, sample, verbose, from_outputs=False):
    folder = REPO / cfg["dir"]
    imgs = collect(folder, sample=sample, always_full=cfg.get("always_full"))
    if not imgs:
        print(f"  [跳过] {name}: 目录无图或不存在（{folder}）", flush=True)
        return None

    expect = cfg["expect"]
    rows = []
    dist = {lv: 0 for lv in RISK_LEVELS}
    dist["error"] = 0
    n_hit = 0
    missing = []
    for p in imgs:
        print(f"  · {name}/{p.name} ...", end="", flush=True)
        t0 = time.time()
        row = run_one(p, fast, verbose, from_outputs=from_outputs)
        row["hit"] = row["risk_level"] in expect
        # ⚠️ 没跑到的图（error）绝不能计进指标分母：否则 --from-outputs 只聚合到 1/3
        # 时，会把「还有 2/3 没跑」显示成「召回 0%」——那是自欺，不是评测。
        # 正确做法：只统计真正跑过的图，未跑的单独记覆盖度。
        if row["risk_level"] == "error":
            missing.append(p.name)
            print(f" 未跑（不计入指标） {round(time.time()-t0,1)}s", flush=True)
            continue
        n_hit += int(row["hit"])
        dist[row["risk_level"]] = dist.get(row["risk_level"], 0) + 1
        rows.append(row)
        print(f" {row['risk_level']} ({'OK' if row['hit'] else 'MISS'}) "
              f"{round(time.time()-t0,1)}s", flush=True)

    n = len(rows)
    return {
        "dir": cfg["dir"],
        "ground_truth": cfg["ground_truth"],
        "bucket": cfg["bucket"],
        "expect_risk": sorted(expect),
        "n": n,
        "n_total": len(imgs),
        "n_evaluated": n,
        "n_not_run": len(missing),
        "missing_files": missing,
        "coverage": round(n / len(imgs), 4) if imgs else None,
        "sampled": (sample is not None and not cfg.get("always_full") and len(imgs) >= sample),
        "risk_distribution": dist,
        "direction_hit_rate": round(n_hit / n, 4) if n else None,
        "n_hit": n_hit,
        "samples": rows,
        "note": cfg.get("note", ""),
    }


def main():
    ap = argparse.ArgumentParser(description="数据集→识别→决策 闭环评测")
    ap.add_argument("--fast", action="store_true",
                    help="跳过 aigc+trufor（仅演示链路，不用于对外指标）")
    ap.add_argument("--sample", type=int, default=None,
                    help="超大受控集最多抽 N 张；realworld_heldout/ai_cross_native 始终全量")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="只跑指定数据集（默认全部）")
    ap.add_argument("--verbose", action="store_true", help="打印 pipeline 逐工具过程")
    ap.add_argument("--from-outputs", action="store_true",
                    help="不重跑 pipeline，直接读已落盘的 outputs/analysis_<stem>.json 聚合"
                         "（用于分批跑完 analyze 后生成闭环报告，避免重复触发批量删除保护）")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    selected = args.datasets or list(DATASETS.keys())

    report = {
        "schema": "beautyproof/closedloop_eval@1",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "fast": args.fast,
            "sample": args.sample,
            "from_outputs": args.from_outputs,
            "model_note": ("pipeline.analyze 全量 aigc+trufor（非 --fast）"
                           if not args.fast else "--fast：已跳过 aigc+trufor，仅供链路演示"),
            "pipeline": "tools/pipeline.analyze()",
        },
        "datasets": {},
        "false_positive_rate": {},
        "zero_shot_recall": {},
        "disposition_distribution": {},
        "summary": {},
    }

    print("===== BeautyProof 数据集闭环评测 =====", flush=True)
    if args.fast:
        print("  [注意] --fast 模式：跳过深度学习模型，指标仅演示链路，不可对外。", flush=True)

    for name in selected:
        if name not in DATASETS:
            print(f"  [忽略] 未知数据集 {name}", flush=True)
            continue
        print(f"\n--- 数据集：{name} ---", flush=True)
        ds = evaluate_dataset(name, DATASETS[name], args.fast, args.sample,
                               args.verbose, from_outputs=args.from_outputs)
        if ds is None:
            continue
        report["datasets"][name] = ds

        # 真实误报率：只从真 held-out 真实图集算（铁律：对外只报这个数）
        if DATASETS[name]["bucket"] == "heldout_real":
            n = ds["n"]
            fpr_alarm = round(sum(1 for r in ds["samples"]
                                  if r["risk_level"] in ALARM) / n, 4) if n else None
            fpr_high = round(sum(1 for r in ds["samples"]
                                 if r["risk_level"] == "high_risk") / n, 4) if n else None
            report["false_positive_rate"]["realworld_heldout"] = {
                "n": n,
                "false_positive_rate_alarm": fpr_alarm,
                "false_positive_rate_high_risk": fpr_high,
                "note": "真实图被误判为 suspicious/high_risk 的比例。对外唯一真实鲁棒性口径。",
            }

        # 跨生成器零样本召回率
        if DATASETS[name]["bucket"] == "zero_shot_cross":
            n = ds["n"]
            rec_hr = round(sum(1 for r in ds["samples"]
                               if r["risk_level"] == "high_risk") / n, 4) if n else None
            rec_alarm = round(sum(1 for r in ds["samples"]
                                  if r["risk_level"] in ALARM) / n, 4) if n else None
            report["zero_shot_recall"]["ai_cross_native"] = {
                "n": n,
                "recall_high_risk": rec_hr,
                "recall_alarm": rec_alarm,
                "note": "跨生成器零样本：本系统只在同生成器族微调，非 MJ/SD/Flux 族外推仅到此。",
            }

    # 处置建议分布（聚合 planner.decide_actions 的 disposition 计数）
    disp = {}
    for name, ds in report["datasets"].items():
        for r in ds["samples"]:
            for d in (r.get("dispositions") or []):
                disp[d] = disp.get(d, 0) + 1
    report["disposition_distribution"] = disp

    # 汇总
    total = sum(ds["n"] for ds in report["datasets"].values())
    controlled = [ds for ds in report["datasets"].values() if ds["bucket"] == "controlled"]
    n_ctrl_hit = sum(ds["n_hit"] for ds in controlled)
    n_ctrl = sum(ds["n"] for ds in controlled)
    fpr = report["false_positive_rate"].get("realworld_heldout", {})
    rec = report["zero_shot_recall"].get("ai_cross_native", {})
    report["summary"] = {
        "datasets_run": list(report["datasets"].keys()),
        "total_samples": total,
        "controlled_direction_hit_rate": round(n_ctrl_hit / n_ctrl, 4) if n_ctrl else None,
        "controlled_n_hit": n_ctrl_hit,
        "controlled_n": n_ctrl,
        "realworld_false_positive_rate": fpr.get("false_positive_rate_alarm"),
        "ai_cross_zero_shot_recall_high_risk": rec.get("recall_high_risk"),
        "disposition_kinds": len(disp),
        # 覆盖度：每个数据集「目录里有多少张 / 实际跑了多少张」。
        # 覆盖度 <1 时，本 json 的指标只代表跑过的那部分，不可当作全集结论对外。
        "coverage": {k: {"n_evaluated": v["n_evaluated"], "n_total": v["n_total"],
                         "coverage": v["coverage"]}
                     for k, v in report["datasets"].items()},
        "full_coverage": all(v["coverage"] == 1 for v in report["datasets"].values()),
    }

    out = RESULTS / "closedloop_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印汇总
    print("\n===== 闭环评测汇总 =====", flush=True)
    for name, ds in report["datasets"].items():
        print(f"  {name:18s} n={ds['n']:>3}  命中率={ds['direction_hit_rate']}  "
              f"分布={ {k: ds['risk_distribution'][k] for k in RISK_LEVELS} }", flush=True)
    if fpr:
        print(f"  真实误报率(held-out)={fpr.get('false_positive_rate_alarm')} "
              f"(high_risk={fpr.get('false_positive_rate_high_risk')})", flush=True)
    if rec:
        print(f"  跨生成器零样本召回(high_risk)={rec.get('recall_high_risk')} "
              f"(alarm={rec.get('recall_alarm')})", flush=True)
    print(f"  处置建议分布={disp}", flush=True)
    print(f"\n  已写 {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

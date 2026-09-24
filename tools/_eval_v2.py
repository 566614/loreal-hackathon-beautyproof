# -*- coding: utf-8 -*-
"""v1 vs v2 对比评估：重点看「真实美妆图误报率」有没有降下来。

诚实原则：把「训练过的图」和「没见过的图（验证集）」分开统计。
在训练过的图上评分会虚高，不能代表真实能力，所以验证集那一行才是关键。
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import aigc_tool  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg")
VAL_RATIO, SEED = 0.2, 42


def files(folder):
    p = REPO / "data" / folder
    if not p.exists():
        return []
    return sorted(q for q in p.glob("*") if q.suffix.lower() in EXTS)


def split(fs, ratio, rng):
    idx = list(range(len(fs)))
    rng.shuffle(idx)
    n = max(1, int(round(len(fs) * ratio)))
    v = set(idx[:n])
    return [fs[i] for i in idx if i not in v], [fs[i] for i in idx if i in v]


def main():
    ai = files("ai")
    rx = files("real_xhs")
    old = files("real") + files("clean") + files("tampered")
    real_all = rx + old

    rng = random.Random(SEED)
    _, ai_val = split(ai, VAL_RATIO, rng)
    _, re_val = split(real_all, VAL_RATIO, rng)
    val_names = {p.name for p in ai_val} | {p.name for p in re_val}

    v2_dir = str(REPO / "models" / "beautyproof_aigen_v2")
    aigc_tool._load_ft_model(v2_dir)

    rows = []
    for f in ai + real_all:
        s, _, _ = aigc_tool.run_aigc_timm(str(f), v2_dir)
        rows.append({"file": f.name, "label": "ai" if f in ai else "real",
                     "v2": round(float(s), 4), "held_out": f.name in val_names})

    # v1 分数：复用已有的两个结果文件，避免重跑
    v1 = {}
    for p, key in ((REPO / "results" / "real_xhs_qualitycheck.json", "ai_prob"),
                   (REPO / "results" / "capability_check.json", None)):
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in (d if key else d.get("per_image", [])):
            v1[r["file"]] = r.get(key or "ai_prob")
    for r in rows:
        r["v1"] = v1.get(r["file"])

    def stat(sub, name, key):
        sub = [r for r in sub if r.get(key) is not None]
        if not sub:
            return {"set": name, "n": 0}
        fp = sum(1 for r in sub if r["label"] == "real" and r[key] >= 0.9)
        n_real = sum(1 for r in sub if r["label"] == "real")
        fn = sum(1 for r in sub if r["label"] == "ai" and r[key] < 0.5)
        n_ai = sum(1 for r in sub if r["label"] == "ai")
        return {"set": name, "n": len(sub),
                "real_false_positive_rate": round(fp / n_real, 4) if n_real else None,
                "fp": fp, "n_real": n_real,
                "ai_recall": round((n_ai - fn) / n_ai, 4) if n_ai else None,
                "n_ai": n_ai}

    held = [r for r in rows if r["held_out"]]
    out = {
        "held_out_validation": {"v1": stat(held, "未训练(验证集)", "v1"),
                                "v2": stat(held, "未训练(验证集)", "v2")},
        "all_real_xhs": {"v1": stat([r for r in rows if r["file"] in {p.name for p in rx}],
                                    "真实美妆74张(含训练过的)", "v1"),
                         "v2": stat([r for r in rows if r["file"] in {p.name for p in rx}],
                                    "真实美妆74张(含训练过的)", "v2")},
        "all_109": {"v1": stat(rows, "全部109张", "v1"), "v2": stat(rows, "全部109张", "v2")},
        "per_image": sorted(rows, key=lambda r: -r["v2"]),
    }
    (REPO / "results" / "aigen_v2_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== v1 vs v2（真实类误报率越低越好，AI 召回越高越好）===")
    for k in ("held_out_validation", "all_real_xhs", "all_109"):
        a, b = out[k]["v1"], out[k]["v2"]
        print(f"\n[{k}]")
        for tag, s in (("v1", a), ("v2", b)):
            if not s.get("n"):
                print(f"  {tag}: 无数据")
                continue
            print(f"  {tag}: n={s['n']}  真实误报 {s['fp']}/{s['n_real']}"
                  f" ({s['real_false_positive_rate']})  AI召回 {s['ai_recall']}")
    print("\n已保存 results/aigen_v2_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

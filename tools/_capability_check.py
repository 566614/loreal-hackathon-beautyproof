# -*- coding: utf-8 -*-
"""当前 AIGC 模型能力画像：把「训练过的图」和「没见过的图」分开算。

为什么要分开：v1 训练时用了 35 张里的 28 张，只留 7 张没参与训练。
如果在全部 35 张上打分，准确率会虚高（模型背过答案），不能代表真实能力。
所以这里复现 train_aigen.py 的分割方式，分别统计：
    - in-sample（训练过的图）：仅参考，不算数
    - held-out（没见过的 7 张）：勉强算数，但样本太小
    - 真实世界（56 张 XHS）：这才是真相
"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import aigc_tool  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg")


def img_files(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS)


def main():
    model_dir = REPO / "models" / "beautyproof_aigen"
    aigc_tool._load_ft_model(str(model_dir))

    d = REPO / "data"
    ai = img_files(d / "ai")
    real = img_files(d / "real")
    clean = img_files(d / "clean")
    tamp = img_files(d / "tampered")
    real_all = real + clean + tamp

    # 复现 train_aigen.py 的分割：按排序取最后 N 张作验证
    ai_tr, ai_val = ai[:-4], ai[-4:]
    real_tr, real_val = real_all[:-3], real_all[-3:]

    def run(files, label):
        out = []
        for f in files:
            score, lab, ok = aigc_tool.run_aigc_timm(str(f), str(model_dir))
            out.append({"file": f.name, "label": label, "ai_prob": round(float(score), 4)})
        return out

    rows = []
    rows += run(ai_tr, "ai")
    rows += run(real_tr, "real")
    rows += run(ai_val, "ai")
    rows += run(real_val, "real")

    # 标记是否参与训练
    train_names = {p.name for p in ai_tr} | {p.name for p in real_tr}
    for r in rows:
        r["seen_in_training"] = r["file"] in train_names

    def summarize(subset, name):
        if not subset:
            return {"set": name, "n": 0}
        correct = sum(1 for r in subset
                      if (r["ai_prob"] >= 0.5) == (r["label"] == "ai"))
        fp = sum(1 for r in subset if r["label"] == "real" and r["ai_prob"] >= 0.9)
        n_real = sum(1 for r in subset if r["label"] == "real")
        return {
            "set": name,
            "n": len(subset),
            "accuracy": round(correct / len(subset), 4),
            "false_positive_rate_on_real": round(fp / n_real, 4) if n_real else None,
            "fp": fp, "n_real": n_real,
        }

    seen = [r for r in rows if r["seen_in_training"]]
    held = [r for r in rows if not r["seen_in_training"]]

    result = {
        "model": "beautyproof_aigen (v1)",
        "controlled_35": {
            "in_sample_seen": summarize(seen, "训练过的图(28)"),
            "held_out_7": summarize(held, "没见过的图(7)"),
            "all_35": summarize(rows, "全部35张"),
        },
        "per_image": sorted(rows, key=lambda r: -r["ai_prob"]),
    }

    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "capability_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 受控集 35 张（实验室条件）===")
    for k in ("in_sample_seen", "held_out_7", "all_35"):
        s = result["controlled_35"][k]
        print(f"  {s['set']}: n={s['n']} acc={s['accuracy']} "
              f"真实误报={s['fp']}/{s['n_real']} ({s['false_positive_rate_on_real']})")
    print("\n已保存 results/capability_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

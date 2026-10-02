# -*- coding: utf-8 -*-
# 来源：原创 —— BeautyProof 团队标定脚本
"""
频域工具标定 + 冻结集评测（spectral calibration & frozen evaluation）

用法：
    python tools/spectral_calib.py            # 标定并写 results/spectral_calib.json + spectral_eval.json
    python tools/spectral_calib.py --refit    # 强制重标定

═══ 标定协议（为什么这么切）═══
第一版协议（DEV 集 = data/ai + data/ai_aug vs data/real + data/clean）**已废弃**，
因为实测发现它标出来的阈值在冻结集上完全不可用（冻结 AUC 仅 0.586，AI 召回 0.18 / 误报 0.20）。
根因有二，都写在这里避免后人重犯：
  ① DEV 的真实图（data/real 10 张 + data/clean 2 张）几乎都是"干净棚拍/工作室"图，
     而真实场景里的小红书图、滤镜自拍都被平台重度压缩过 —— 频谱形态完全不同，
     用前者定阈值必然偏。
  ② 真实图只有 12 张，"最大间隔"阈值法在这种样本量下极不稳定。

第二版协议（本脚本采用）：
  池子 = 全部**从未参与 v1~v7 任何训练**的冻结图
         AI   : data/ai_cross(26) + data/ai_cross_native(12) + data/ai_mj_sd_flux_heldout/flux(1) = 39
         REAL : data/realworld_heldout(10) + data/real_filter_selfie 的 heldout_never_train 段(10) = 20
  按 seed 20261002 **对半切**：
      CAL  段 → 定极性、逻辑回归权重、决策线
      TEST 段 → 只做最终评测，**全程不参与任何拟合**
  ⚠️ 诚实边界：TEST 段只有约 20 AI / 10 REAL，样本小、置信区间宽，
     任何数字都要带 n 一起讲，不能只报一个百分比。答辩时必须同时给出 n。

纪律与本仓库其它评测工具一致（见 eval_realworld_heldout.py）：
    ① 冻结图永不训练；TEST 段永不调参；
    ② 缺失文件记 missing，绝不进分母；
    ③ 输出 n_total / n_evaluated / coverage，不允许"没跑"伪装成"跑过"。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_tool import FEATURE_KEYS, extract_features, score_from_features  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_SEED = 20261002
TARGET_REAL_FP = 0.10   # CAL 上真实图误报率上限（宁漏勿伤，与灰带策略一致）
MIN_TEST_PER_CLASS = 4  # 每类少于这个数就明确标注"样本不足，结论仅供参考"


def _list(d):
    p = Path(d)
    if not p.is_dir():
        return []
    return sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXT and not f.name.startswith("_")])


def collect(repo_root, rel_dirs, tag):
    rows, missing = [], []
    for d in rel_dirs:
        files = _list(repo_root / d)
        if not files:
            missing.append({"dir": d, "note": "目录不存在或为空"})
        for f in files:
            try:
                rows.append({"file": f"{tag}/{Path(d).name}/{f.name}", "feat": extract_features(f)})
            except Exception as e:
                missing.append({"dir": d, "file": f.name, "note": f"提取失败: {e}"})
    return rows, missing


def collect_fs(repo_root, segment):
    fs_dir = repo_root / "data" / "real_filter_selfie"
    sp_p = fs_dir / "SPLIT_MANIFEST.json"
    if not sp_p.exists():
        return [], [{"note": "SPLIT_MANIFEST.json 不存在"}]
    sp = json.loads(sp_p.read_text(encoding="utf-8"))
    rows, missing = [], []
    for fn in sp.get(segment, []):
        f = fs_dir / fn
        if not f.exists():
            missing.append({"file": fn, "note": "文件不存在"})
            continue
        try:
            rows.append({"file": f"fs_{segment}/{fn}", "feat": extract_features(f)})
        except Exception as e:
            missing.append({"file": fn, "note": f"提取失败: {e}"})
    return rows, missing


def auc(pos, neg):
    """Mann-Whitney U 求 AUC（并列取平均秩）。输入越大越像 AI。"""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    rsum_pos = sum(r[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_logistic(X, y, l2=1.0, iters=400, lr=0.5):
    """极简逻辑回归（numpy 手写，不引入新依赖）。带 L2 正则防过拟合。"""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) / n
        reg = np.zeros_like(w)
        reg[:-1] = l2 * w[:-1] / n
        w -= lr * (g + reg)
    return w


def matrix(rows, keys):
    out = []
    for r in rows:
        vals = []
        for k in keys:
            v = r["feat"].get(k)
            vals.append(v if v is not None else np.nan)
        out.append(vals)
    return np.array(out, dtype=float)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    calib_p = repo_root / "results" / "spectral_calib.json"
    if calib_p.exists() and "--refit" not in sys.argv:
        print(f"已存在标定文件，跳过（要重标定请加 --refit）: {calib_p}")
        return 0

    ai_rows, m1 = collect(repo_root, ["data/ai_cross", "data/ai_cross_native", "data/ai_mj_sd_flux_heldout/flux"], "frozen_ai")
    real_rows, m2 = collect(repo_root, ["data/realworld_heldout"], "frozen_real")
    fs_rows, m3 = collect_fs(repo_root, "heldout_never_train")
    real_rows = real_rows + fs_rows
    print(f"冻结池：AI {len(ai_rows)} 张 / REAL {len(real_rows)} 张")
    for m in (m1 + m2 + m3):
        if m:
            print("   ! ", m)

    if len(ai_rows) < 8 or len(real_rows) < 8:
        print("冻结池样本太少，中止（不写标定文件）")
        return 2

    rng = np.random.default_rng(SPLIT_SEED)
    ai_idx = rng.permutation(len(ai_rows))
    real_idx = rng.permutation(len(real_rows))
    ai_cal = [ai_rows[i] for i in ai_idx[: len(ai_rows) // 2]]
    ai_test = [ai_rows[i] for i in ai_idx[len(ai_rows) // 2:]]
    real_cal = [real_rows[i] for i in real_idx[: len(real_rows) // 2]]
    real_test = [real_rows[i] for i in real_idx[len(real_rows) // 2:]]
    print(f"   CAL : AI {len(ai_cal)} / REAL {len(real_cal)}")
    print(f"   TEST: AI {len(ai_test)} / REAL {len(real_test)}")

    # ---- 1) 在 CAL 上逐特征定极性（AUC 离 0.5 越远越有信息量；<0.5 则反号）----
    print("\n== CAL 上拟合：逐特征极性 + 逻辑回归权重 ==")
    keys = []
    polarity = {}
    per_feat = {}
    for k in FEATURE_KEYS:
        pos = [r["feat"][k] for r in ai_cal if r["feat"].get(k) is not None]
        neg = [r["feat"][k] for r in real_cal if r["feat"].get(k) is not None]
        a = auc(pos, neg)
        if a is None:
            print(f"   {k:24s} 不可用")
            continue
        pol = 1.0 if a >= 0.5 else -1.0
        if abs(a - 0.5) < 0.05:
            print(f"   {k:24s} CAL AUC={a:.4f} 离随机太近 → 丢弃该特征")
            continue
        keys.append(k)
        polarity[k] = "high_is_ai" if pol > 0 else "low_is_ai"
        per_feat[k] = {"cal_auc_raw": round(a, 4), "cal_auc_effective": round(max(a, 1 - a), 4)}
        print(f"   {k:24s} CAL raw AUC={a:.4f} → 极性 {polarity[k]:11s} 有效 AUC={max(a, 1-a):.4f}")

    if len(keys) < 2:
        print("可用特征不足 2 个，中止（不写标定文件）")
        return 2

    def build(rows):
        X = matrix(rows, keys)
        for j, k in enumerate(keys):
            col = X[:, j]
            med = np.nanmedian(col)
            col = np.where(np.isnan(col), med, col)
            X[:, j] = col
        # 标准化用 CAL 的均值标准差，避免信息泄漏
        return X

    Xtr = build(ai_cal + real_cal)
    ytr = np.array([1] * len(ai_cal) + [0] * len(real_cal), dtype=float)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    w = fit_logistic((Xtr - mu) / sd, ytr)

    calib_features = {}
    for j, k in enumerate(keys):
        # 把标准化后的权重反解回"原始特征"的等效阈值，供 explain/审计阅读
        calib_features[k] = {
            "polarity": polarity[k],
            "thr": round(float(mu[j]), 6),
            "scale": round(float(sd[j]), 6),
            "weight": round(abs(float(w[j])), 4),
            "signed_weight": round(float(w[j]), 4),
        }
    print(f"   逻辑回归权重: " + json.dumps({k: calib_features[k]['signed_weight'] for k in keys}, ensure_ascii=False))

    def prob(rows):
        X = (build(rows) - mu) / sd
        z = np.clip(X @ w[:-1] + w[-1], -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    # ---- 2) 在 CAL 上定决策线：真实误报 <= TARGET_REAL_FP 时最大化 AI 召回 ----
    pc, pr = prob(ai_cal), prob(real_cal)
    cands = sorted(set([round(float(x), 3) for x in list(pc) + list(pr)] + [0.5]))
    best = None
    for thr in cands:
        fp = float(np.mean(pr >= thr))
        rec = float(np.mean(pc >= thr))
        if fp <= TARGET_REAL_FP and (best is None or rec > best[1]):
            best = (thr, rec, fp)
    if best is None:
        thr, rec, fp = 1.01, 0.0, 0.0
    else:
        thr, rec, fp = best
    print(f"   决策线 thr={thr} → CAL AI 召回={rec:.4f} / 真实误报={fp:.4f}")

    calib = {
        "schema": "beautyproof/spectral_calib@3",
        "features": calib_features,
        "decision_thr": round(float(thr), 4),
        "normalization": {"mean": [round(float(x), 8) for x in mu],
                          "std": [round(float(x), 8) for x in sd],
                          "bias": round(float(w[-1]), 6)},
        "feature_order": keys,
        "calibrated_on": (
            f"冻结池对半切 CAL 段（seed={SPLIT_SEED}）：AI {len(ai_cal)} 张 / REAL {len(real_cal)} 张；"
            f"目标真实误报<={TARGET_REAL_FP}；TEST 段全程未参与拟合"
        ),
        "cal_metrics": {"ai_recall": round(rec, 4), "real_fp_rate": round(fp, 4)},
        "per_feature_cal": per_feat,
        "note": "本分数是频域倾向分（规则+逻辑回归），不是「AI 生成概率」，不可与 aigc_tool 的模型概率混读",
    }
    calib_p.parent.mkdir(exist_ok=True)
    calib_p.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写标定: {calib_p}")

    # ---------- 3) TEST 段评测（只读）----------
    print("\n== TEST 段评测（不参与拟合）==")
    pt, prt = prob(ai_test), prob(real_test)

    def stat(p, is_ai):
        n = len(p)
        if n == 0:
            return {"n_evaluated": 0, "note": "无样本"}
        hit = int(np.sum(p >= thr))
        d = {
            "n_evaluated": n,
            "mean_score": round(float(np.mean(p)), 4),
            "median_score": round(float(np.median(p)), 4),
            "n_ge_thr": hit,
        }
        d["recall" if is_ai else "fp_rate"] = round(hit / n, 4)
        return d

    test_ai, test_real = stat(pt, True), stat(prt, False)
    test_auc = auc(list(pt), list(prt))
    full_auc = auc(list(prob(ai_rows)), list(prob(real_rows)))

    print(f"   TEST AI   {json.dumps(test_ai, ensure_ascii=False)}")
    print(f"   TEST REAL {json.dumps(test_real, ensure_ascii=False)}")
    print(f"   TEST AUC  = {test_auc:.4f}")
    print(f"   全冻结池 AUC（参考值，含 CAL）= {full_auc:.4f}")

    small = (test_ai.get("n_evaluated", 0) < MIN_TEST_PER_CLASS
             or test_real.get("n_evaluated", 0) < MIN_TEST_PER_CLASS)

    eval_out = {
        "schema": "beautyproof/spectral_eval@2",
        "calib_file": "results/spectral_calib.json",
        "split_seed": SPLIT_SEED,
        "decision_thr": round(float(thr), 4),
        "protocol": (
            "冻结池（全部从未参与 v1~v7 训练）按 seed 对半切：CAL 定极性/权重/决策线，"
            "TEST 只做评测。TEST 样本量小，数字必须带 n 一起引用。"
        ),
        "cal": {"ai": {"n_evaluated": len(ai_cal), "ai_recall": round(rec, 4)},
                "real": {"n_evaluated": len(real_cal), "real_fp_rate": round(fp, 4)}},
        "test": {"ai": test_ai, "real": test_real, "auc": round(test_auc, 4) if test_auc else None},
        "full_frozen_pool_reference": {
            "n_ai": len(ai_rows), "n_real": len(real_rows), "auc": round(full_auc, 4) if full_auc else None,
            "note": "含 CAL 段，仅作参考，不可当测试结果引用",
        },
        "sample_size_warning": "TEST 段每类样本 <%d，置信区间宽" % MIN_TEST_PER_CLASS if small else None,
        "per_feature_cal": per_feat,
        "missing": m1 + m2 + m3,
        "integrity": "TEST 段未参与任何拟合；缺失文件已排除在分母外",
    }
    out_p = repo_root / "results" / "spectral_eval.json"
    out_p.write_text(json.dumps(eval_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写评测: {out_p}")
    if small:
        print("⚠️ TEST 段样本量偏小，答辩引用时必须同时说出 n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

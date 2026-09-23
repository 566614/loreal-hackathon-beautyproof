# -*- coding: utf-8 -*-
"""评测「冻结 CLIP-ViT-L/14 特征」在美妆图真伪判别上的能力（严格留一交叉验证）。

用法: python tools/eval_clip_head.py

诚实性设计（样本少，极易自欺，所以每一步都刻意防作弊）：
  1. **留一交叉验证（LOO）**：每张图都当一次"没见过的测试图"，只用其余样本做参照。
  2. **零训练方案优先**：CLIP 特征 + kNN / 类原型，不训练任何参数。
     35 样本 < 768 维，"训"一个线性头是欠定的 → 线性探针改为 PCA 降到 16 维后再回归，
     并明确标注它只是对照，不是主结论。
  3. **捷径排查（最关键）**：真实图是品牌素材 JPEG、AI 图是即梦 PNG，
     格式/分辨率/压缩质量本身就是可被利用的捷径。所以跑两套对比：
       · raw        —— 原图直送
       · normalized —— 统一尺寸 + 统一 JPEG q90 重编码，抹掉格式线索
     若 raw 高而 normalized 崩 → 学的是"哪一批的 JPEG"，不是真假。
  4. **与 MobileNetV3 公平对打**：在同一批 7 张留出图上比（那 7 张是
     ai_17~20 + tampered_01~03，MobileNetV3 训练时没见过，得分 7/7）。

评分口径：kNN 与原型都输出「AI 侧相似度 − 真实侧相似度」的**间隔分数**，
这样 ACC（取符号）与 AUC（用间隔排序）口径一致；1-NN 不再用"最近邻相似度"当分数
（那个量不构成决策函数，AUC 会失真）。

指标：ACC（准确率）+ AUC（阈值无关，小样本比 ACC 稳）。
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

CLIP_DIR = REPO / "models" / "clip-vit-large-patch14"
CACHE_DIR = REPO / "outputs" / "_clip_cache"
EXTS = (".png", ".jpg", ".jpeg")

# MobileNetV3 微调时留出的验证集（训练脚本 tools/train_aigen.py 里 ai[-4:] + real_all[-3:]）
HOLDOUT_NAMES = {
    "ai_17.png", "ai_18.png", "ai_19.png", "ai_20.png",
    "tampered_01_copy_move.png", "tampered_02_splice.png", "tampered_03_text_edit.png",
}


def list_files(folder):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS)


def reencode(img, size=512, quality=90):
    """统一尺寸 + 统一 JPEG 重编码 —— 抹掉格式/分辨率/压缩质量这些"捷径线索"。"""
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def extract_features(paths, model, proc, normalize_shortcut):
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if normalize_shortcut:
            img = reencode(img)
        inputs = proc(images=img, return_tensors="pt")
        # 显式走「视觉塔 → 投影层」：transformers 5.x 的 get_image_features
        # 改成返回对象而非张量，显式调用子模块不受版本 API 变动影响。
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            f = model.visual_projection(vision_out.pooler_output)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f[0].cpu().numpy())
    return np.stack(feats)


def load_or_extract(tag, paths, model, proc, norm):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"feats_{tag}.npy"
    if f.exists():
        X = np.load(f)
        if X.shape[0] == len(paths):
            print(f"  [缓存] 特征复用 {f.name}", flush=True)
            return X
    X = extract_features(paths, model, proc, norm)
    np.save(f, X)
    return X


def _margin_and_pred(X_tr, y_tr, x, k):
    """用参照集给出「AI 间隔分」= 最近 k 个 AI 的平均相似度 − 最近 k 个真实图的平均相似度。"""
    sims = X_tr @ x
    ai = sims[y_tr == 1]
    re = sims[y_tr == 0]
    kk = min(k, len(ai), len(re))
    s_ai = np.sort(ai)[-kk:].mean() if kk else -1.0
    s_re = np.sort(re)[-kk:].mean() if kk else -1.0
    return float(s_ai - s_re)


def loo_knn(X, y, k=1):
    n = len(y)
    preds, scores = [], []
    for i in range(n):
        mask = np.arange(n) != i
        s = _margin_and_pred(X[mask], y[mask], X[i], k)
        scores.append(s)
        preds.append(1 if s > 0 else 0)
    return np.array(preds), np.array(scores)


def loo_prototype(X, y):
    n = len(y)
    preds, scores = [], []
    for i in range(n):
        mask = np.arange(n) != i
        cent = {}
        for c in (0, 1):
            sel = mask & (y == c)
            v = X[sel].mean(axis=0)
            cent[c] = v / (np.linalg.norm(v) + 1e-9)
        s = float(X[i] @ cent[1] - X[i] @ cent[0])
        scores.append(s)
        preds.append(1 if s > 0 else 0)
    return np.array(preds), np.array(scores)


def loo_pca_logreg(X, y, ncomp=16, l2=1e-3, steps=600, lr=0.05):
    """对照用的线性探针：先在训练折上 PCA 降到 16 维，再训 L2 逻辑回归。

    为什么不用原始 768 维：35 样本 < 768 维，直接回归会退化到只学偏置
    （第一版就是这么坏掉的：AUC 0.000，等价于全判一类）。
    """
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(y)
    preds, scores = [], []
    for i in range(n):
        mask = (torch.arange(n) != i).numpy()
        Xtr_np, ytr_np = X[mask], y[mask]
        mu = Xtr_np.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xtr_np - mu, full_matrices=False)
        P = Vt[:ncomp].T                                   # (768, ncomp)
        Ztr = torch.tensor((Xtr_np - mu) @ P, dtype=torch.float32)
        z = torch.tensor((X[i] - mu) @ P, dtype=torch.float32)
        ytr = torch.tensor(ytr_np, dtype=torch.float32)
        w = torch.zeros(Ztr.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(steps):
            opt.zero_grad()
            loss = lossf(Ztr @ w + b, ytr) + l2 * (w ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = float((z @ w + b).item())
        scores.append(s)
        preds.append(1 if s > 0 else 0)
    return np.array(preds), np.array(scores)


def auc(y, score):
    """秩和法 AUC（含并列处理）"""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


METHODS = [
    ("1-NN（零训练）", lambda X, y: loo_knn(X, y, 1)),
    ("5-NN（零训练）", lambda X, y: loo_knn(X, y, 5)),
    ("类原型（零训练）", lambda X, y: loo_prototype(X, y)),
    ("PCA16+逻辑回归（对照）", lambda X, y: loo_pca_logreg(X, y)),
]


def run_protocol(tag, X, y, names):
    print(f"===== 协议：{tag} =====", flush=True)
    block = {}
    for name, fn in METHODS:
        pred, score = fn(X, y)
        acc = float((pred == y).mean())
        a = auc(y, score)
        block[name] = {"acc": round(acc, 4), "auc": round(a, 4)}
        print(f"  {name:24s} ACC {acc:.3f}   AUC {a:.3f}", flush=True)

    # 类原型的间隔分布（看有没有清晰间隙）
    _, proto = loo_prototype(X, y)
    s_ai, s_re = proto[y == 1], proto[y == 0]
    sep = bool(s_ai.min() > s_re.max())
    print(f"  类原型间隔：AI 侧 min {s_ai.min():+.3f} / 非AI 侧 max {s_re.max():+.3f}"
          f"  → {'完全分开' if sep else '有重叠'}", flush=True)
    print("", flush=True)
    return block, {"ai_min": round(float(s_ai.min()), 4),
                   "real_max": round(float(s_re.max()), 4),
                   "separated": sep}, proto


def eval_holdout7(X, y, names):
    """只在 MobileNetV3 那 7 张留出图上评（用其余 28 张做参照），做公平对打。"""
    idx = [i for i, nm in enumerate(names) if nm in HOLDOUT_NAMES]
    if not idx:
        return None
    res = {}
    for name, fn in METHODS:
        # 对每个留出样本，用"非留出"的 28 张做参照
        preds, scores, labels = [], [], []
        for k_name in (name,):
            for i in idx:
                mask = np.array([j not in idx for j in range(len(y))])
                Xtr, ytr = X[mask], y[mask]
                if "kNN" in k_name:
                    kk = 1 if k_name.startswith("1") else 5
                    s = _margin_and_pred(Xtr, ytr, X[i], kk)
                elif "原型" in k_name:
                    v1 = Xtr[ytr == 1].mean(0); v1 = v1 / (np.linalg.norm(v1) + 1e-9)
                    v0 = Xtr[ytr == 0].mean(0); v0 = v0 / (np.linalg.norm(v0) + 1e-9)
                    s = float(X[i] @ v1 - X[i] @ v0)
                else:
                    # 对照法直接在 28 张上训一次 PCA16+逻辑回归，再判这 7 张
                    mu = Xtr.mean(0)
                    U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
                    P = Vt[:16].T
                    Ztr = torch.tensor((Xtr - mu) @ P, dtype=torch.float32)
                    ytr_t = torch.tensor(ytr, dtype=torch.float32)
                    w = torch.zeros(Ztr.shape[1], requires_grad=True)
                    b = torch.zeros(1, requires_grad=True)
                    opt = torch.optim.Adam([w, b], lr=0.05)
                    lf = torch.nn.BCEWithLogitsLoss()
                    for _ in range(600):
                        opt.zero_grad()
                        loss = lf(Ztr @ w + b, ytr_t) + 1e-3 * (w ** 2).sum()
                        loss.backward(); opt.step()
                    with torch.no_grad():
                        s = float((torch.tensor((X[i] - mu) @ P, dtype=torch.float32) @ w + b).item())
                preds.append(1 if s > 0 else 0)
                scores.append(s)
                labels.append(int(y[i]))
        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        res[name] = {"acc": round(correct / len(idx), 4), "n": len(idx)}
        print(f"  {name:24s} 留出 7 张命中 {correct}/{len(idx)}", flush=True)
    return res


def main():
    if not (CLIP_DIR / "model.safetensors").exists():
        print(f"CLIP 未下载：{CLIP_DIR}\n先跑 python tools/download_clip.py")
        return 1

    from transformers import CLIPImageProcessor, CLIPModel

    print("加载 CLIP-ViT-L/14（CPU）...", flush=True)
    proc = CLIPImageProcessor.from_pretrained(str(CLIP_DIR))
    model = CLIPModel.from_pretrained(str(CLIP_DIR), low_cpu_mem_usage=True)
    model.eval()
    print("加载完成\n", flush=True)

    real = list_files(REPO / "data" / "real")
    ai = list_files(REPO / "data" / "ai")
    clean = list_files(REPO / "data" / "clean")
    tampered = list_files(REPO / "data" / "tampered")

    paths = ai + real + clean + tampered
    y = np.array([1] * len(ai) + [0] * (len(real) + len(clean) + len(tampered)))
    names = [p.name for p in paths]
    print(f"样本：AI {len(ai)} / 非 AI {len(real)+len(clean)+len(tampered)}"
          f"（real {len(real)} + clean {len(clean)} + tampered {len(tampered)}）= {len(paths)} 张\n", flush=True)

    report = {"n_ai": len(ai), "n_nonai": len(real) + len(clean) + len(tampered), "protocols": {}}

    for tag, norm in (("raw", False), ("normalized", True)):
        label = {"raw": "raw（原图，未做捷径控制）",
                 "normalized": "normalized（统一尺寸 + JPEG q90，封堵格式捷径）"}[tag]
        X = load_or_extract(tag, paths, model, proc, norm)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        block, margin, proto = run_protocol(label, X, y, names)
        report["protocols"][tag] = block
        report[f"prototype_margin_{tag}"] = margin
        if tag == "raw":
            report["per_image_raw"] = [
                {"file": names[i], "label": "ai" if y[i] == 1 else "real",
                 "proto_margin": round(float(proto[i]), 4)} for i in range(len(y))]

    # 与 MobileNetV3 在同一 7 张留出图上的公平对打（用 raw 特征）
    X = load_or_extract("raw", paths, model, proc, False)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    print("===== 公平对打：同一批 7 张留出图（ai_17~20 + tampered_01~03）=====", flush=True)
    ho = eval_holdout7(X, y, names)
    report["holdout7"] = ho
    print("", flush=True)

    fin = REPO / "results" / "aigen_finetune.json"
    if fin.exists():
        old = json.loads(fin.read_text(encoding="utf-8"))
        report["mobilenetv3_baseline"] = {
            "val_accuracy": old.get("val_accuracy"),
            "val_n": 7,
            "note": "MobileNetV3 训练 28 / 验证同一批 7 张；CLIP 是零训练 + 该 7 张不参与参照",
        }
        print(f"MobileNetV3（训练 28 / 验证同一批 7 张）验证集准确率 = {old.get('val_accuracy')}", flush=True)

    out = REPO / "results" / "clip_head_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

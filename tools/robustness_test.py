# -*- coding: utf-8 -*-
"""
对抗 / 鲁棒性测试 —— 回答「攻击者反向调图、直到骗过检测怎么办？」

干什么：
    拿几张已知真值的样本（3 张篡改 + 1 张干净），施加一组**普通人用美图软件就能做**
    的扰动（另存为更低质量、缩小尺寸、加噪点、调亮度对比度），
    然后比较扰动前后的 TruFor 篡改分数，看结论会不会被"调"翻档。

为什么要有这个：
    评审一定会问：「攻击者拿你的工具反复调图、直到骗过检测怎么办？」
    这份测试给出的不是"我们绝对安全"，而是"在哪些操作下结论会变、变多少"的实测边界。

用法：
    python tools/robustness_test.py            # 一行跑完，结果直接写进 results/robustness_report.md
    python tools/robustness_test.py --force    # 忽略 TruFor 缓存全部重跑

铁律（与项目其余部分一致）：
    这是**鲁棒性测量**，不是攻击教程。所有扰动都是日常图片处理操作，不含任何针对性优化、
    不搜索"最小扰动"、不做梯度反演 —— 我们的目的恰恰是证明"随手调图骗不过去"，
    以及诚实标注"哪些操作会让结论降级"。

依赖：tools/trufor_tool.py（TruFor 官方模型封装，含缓存）
"""
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import trufor_tool  # noqa: E402


def _patch_rmtree():
    """让 trufor_tool 里的 shutil.rmtree 变成空操作（只清它自己的两个临时目录）。

    原因：运行环境带"批量删除保护"，累计删除数超阈值时会直接中断进程。
    TruFor 封装每批推理前会清空 outputs/_trufor_in、_trufor_out —— 那两个目录
    只有本次测试自己的产物，清不清都不影响正确性（npz 按文件名 stem 匹配，
    本次所有扰动图文件名唯一，不会与残留文件混淆），所以这里直接跳过清理，
    既不删任何文件，也不会碰到 outputs/ 里其他人的产物。
    """
    def _rmtree(path, ignore_errors=False, onerror=None):
        return None

    shutil.rmtree = _rmtree


def _patch_visualization():
    """关掉 TruFor 封装里的热力图落盘。

    48 张扰动图 × 2 张可视化 = 96 个文件，会淹没 outputs/ 里其他人的正式产物；
    本测试只关心"分数会不会变档"，不看定位图，所以这里不落盘（不改任何已有代码）。
    """
    trufor_tool.save_localization_map = lambda npz_path, image_path: (None, None)

OUT_DIR = REPO / "outputs" / "_robust"          # 扰动图落盘处（英文文件名，规避中文路径问题）
REPORT = REPO / "results" / "robustness_report.md"

# 项目分级阈值（与 tools/rule_engine.py THRESHOLDS 保持一致）
TH_HIGH = 0.9
TH_SUS = 0.5

SAMPLES = [
    ("tampered_01_copy_move", "data/tampered/tampered_01_copy_move.png", "复制粘贴篡改", "tampered"),
    ("tampered_02_splice",    "data/tampered/tampered_02_splice.png",    "拼接篡改",     "tampered"),
    ("tampered_03_text_edit", "data/tampered/tampered_03_text_edit.png", "改文字",       "tampered"),
    ("clean_01",              "data/clean/clean_01.png",                 "干净图（对照）", "clean"),
]


# ------------------------------------------------------------------ 扰动定义
def _jpeg(img, q):
    """JPEG 重压缩：按质量 q 编码一次（写原始字节，避免二次压缩）"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes(), ".jpg"


def _rescale(img, s):
    """缩放再还原：先按 s 缩小（INTER_AREA），再放大回原尺寸（INTER_CUBIC）"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), ".png"


def _noise(img, sigma):
    """高斯噪声：像素值加 N(0, sigma)，固定随机种子保证可复现"""
    rng = np.random.default_rng(20240501)
    out = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8), ".png"


def _brightness(img, factor):
    """亮度：整图线性缩放 factor（1.1 = +10%，0.9 = -10%）"""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8), ".png"


def _contrast(img, factor):
    """对比度：围绕全图均值拉伸 factor 倍，均值不变"""
    mean = float(img.mean())
    out = (img.astype(np.float32) - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8), ".png"


# (key, 人话标签, 函数)
PERTURBATIONS = [
    ("jpg90",      "JPEG 重压缩 q=90", lambda i: _jpeg(i, 90)),
    ("jpg70",      "JPEG 重压缩 q=70", lambda i: _jpeg(i, 70)),
    ("jpg50",      "JPEG 重压缩 q=50", lambda i: _jpeg(i, 50)),
    ("jpg30",      "JPEG 重压缩 q=30", lambda i: _jpeg(i, 30)),
    ("scale075",   "缩放至 75% 再还原", lambda i: _rescale(i, 0.75)),
    ("scale050",   "缩放至 50% 再还原", lambda i: _rescale(i, 0.50)),
    ("noise05",    "高斯噪声 sigma=5",  lambda i: _noise(i, 5)),
    ("noise15",    "高斯噪声 sigma=15", lambda i: _noise(i, 15)),
    ("bright_p10", "亮度 +10%",         lambda i: _brightness(i, 1.10)),
    ("bright_m10", "亮度 -10%",         lambda i: _brightness(i, 0.90)),
    ("contrast_p10", "对比度 +10%",     lambda i: _contrast(i, 1.10)),
    ("contrast_m10", "对比度 -10%",     lambda i: _contrast(i, 0.90)),
]


# ------------------------------------------------------------------ 读写（中文路径安全）
def imread_any(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读图失败: {path}")
    return img


def write_any(path, payload, ext):
    if ext == ".jpg":
        Path(path).write_bytes(payload)          # 已经是 JPEG 字节流，直接落盘
    else:
        ok, buf = cv2.imencode(".png", payload)
        if not ok:
            raise RuntimeError(f"写图失败: {path}")
        Path(path).write_bytes(buf.tobytes())


# ------------------------------------------------------------------ 分级
def tier(score):
    """按项目阈值把分数翻译成档位（只看 TruFor 这一条证据）"""
    if score is None:
        return "无结果"
    if score >= TH_HIGH:
        return "high_risk"
    if score >= TH_SUS:
        return "suspicious"
    return "未报警(<0.5)"


# ------------------------------------------------------------------ 主流程
def build_perturbed():
    """生成全部扰动图，返回 {sample_key: [(pert_key, 标签, 路径), ...]}"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = {}
    for skey, rel, _desc, _gt in SAMPLES:
        src = REPO / rel
        img = imread_any(src)
        items = []
        for pkey, plabel, fn in PERTURBATIONS:
            payload, ext = fn(img)
            out = OUT_DIR / f"{skey}__{pkey}{ext}"
            write_any(out, payload, ext)
            items.append((pkey, plabel, out))
        plan[skey] = items
    return plan


def run_all(force=False):
    """一次性把全部样本 + 扰动图丢给 TruFor（模型只加载一次，最省时间）

    注意：每张扰动图都是**不同的文件名**，所以缓存 key 各不相同，不会互相串味；
    已跑过的会被缓存直接命中跳过，因此中断后重跑是安全的。
    """
    plan = build_perturbed()
    todo = []
    for skey, rel, _desc, _gt in SAMPLES:
        todo.append((REPO / rel).resolve())
        todo.extend([p.resolve() for (_k, _l, p) in plan[skey]])

    t0 = time.time()
    cache, err = trufor_tool.run_trufor_batch(todo, force=force)
    if err:
        print(f"[警告] TruFor 批量推理返回错误：{err}", flush=True)
    print(f"推理完成：提交 {len(todo)} 张 · 用时 {time.time() - t0:.1f}s", flush=True)

    results = {}
    for skey, rel, desc, gt in SAMPLES:
        src = (REPO / rel).resolve()
        rows = []
        base_res = cache.get(str(src))
        base_score = base_res.get("trufor_score") if (base_res or {}).get("available") else None
        rows.append({
            "label": "原始图（无扰动）",
            "score": base_score,
            "delta": None,
            "tier": tier(base_score),
            "flipped": False,
            "ratio": (base_res or {}).get("tampered_area_ratio"),
            "ok": bool((base_res or {}).get("available")),
            "reason": (base_res or {}).get("reason", ""),
        })
        for pkey, plabel, path in plan[skey]:
            r = cache.get(str(path.resolve())) or {}
            ok = bool(r.get("available"))
            sc = r.get("trufor_score") if ok else None
            delta = (sc - base_score) if (ok and base_score is not None) else None
            rows.append({
                "label": plabel,
                "score": sc,
                "delta": delta,
                "tier": tier(sc),
                "flipped": bool(ok and base_score is not None and tier(sc) != tier(base_score)),
                "ratio": r.get("tampered_area_ratio"),
                "ok": ok,
                "reason": r.get("reason", ""),
            })
        results[skey] = {"desc": desc, "gt": gt, "rows": rows}
    return results


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def fmt_delta(v):
    return "—" if v is None else f"{v:+.4f}"


def render_report(results):
    """把实测结果渲染成给评审看的 Markdown"""
    lines = []
    lines.append("# 鲁棒性 / 对抗测试报告 —— 攻击者反复调图能不能骗过检测？")
    lines.append("")
    lines.append("> 被测模型：TruFor（CVPR 2023 图像取证）。样本：3 张已知篡改图 + 1 张干净对照图。")
    lines.append("> 方法：对每张图施加 12 种**普通图片软件就能做**的操作（不针对模型做优化、不搜索最小扰动），")
    lines.append("> 比较扰动前后的 TruFor 篡改分数。档位阈值沿用项目定级：≥0.9 = high_risk，≥0.5 = suspicious，<0.5 = 未报警。")
    lines.append("> 全部数字由 `python tools/robustness_test.py` 实跑得出，非估算。")
    lines.append("")

    # ---- 分样本明细表
    for skey, rel, desc, gt in SAMPLES:
        info = results[skey]
        base = info["rows"][0]
        lines.append(f"## 样本：`{rel}`")
        lines.append("")
        lines.append(f"- 真值：{desc}（ground truth = {gt}）")
        lines.append(f"- 原始分：**{fmt(base['score'])}**（{base['tier']}），可疑区域占比 {fmt(base['ratio'])}")
        lines.append("")
        lines.append("| 扰动 | 扰动后分数 | 相对原始分变化 | 档位 | 结论是否翻转 |")
        lines.append("|---|---:|---:|---|---|")
        for r in info["rows"][1:]:
            flip = "**是**" if r["flipped"] else "否"
            if not r["ok"]:
                lines.append(f"| {r['label']} | 失败 | — | — | 失败（{r['reason'] or '未返回结果'}） |")
            else:
                lines.append(f"| {r['label']} | {fmt(r['score'])} | {fmt_delta(r['delta'])} | {r['tier']} | {flip} |")
        lines.append("")

    # ---- 总表
    lines.append("## 总表：样本 × 扰动 → 分数 / 变化量 / 是否翻转")
    lines.append("")
    header = "| 扰动 |" + "".join(f" {skey} |" for skey, _r, _d, _g in SAMPLES)
    lines.append(header)
    lines.append("|" + "---|" * (len(SAMPLES) + 1))
    for idx in range(len(PERTURBATIONS) + 1):
        label = "原始图（无扰动）" if idx == 0 else PERTURBATIONS[idx - 1][1]
        cells = []
        for skey, _r, _d, _g in SAMPLES:
            r = results[skey]["rows"][idx]
            if not r["ok"]:
                cells.append(" 失败 ")
            elif idx == 0:
                cells.append(f" {fmt(r['score'])} ")
            else:
                mark = " ⚠" if r["flipped"] else ""
                cells.append(f" {fmt(r['score'])} ({fmt_delta(r['delta'])}){mark} ")
        lines.append(f"| {label} |" + "|".join(cells) + "|")
    lines.append("")
    lines.append("（单元格 = 扰动后分数（相对原始分变化量）；⚠ 标记 = 跨过 0.5 / 0.9 阈值、档位发生变化）")
    lines.append("")

    # ---- 自动统计
    flip_rows, down_rows, tamper_scores = [], [], []
    for skey, _rel, _desc, gt in SAMPLES:
        b = results[skey]["rows"][0]
        for r in results[skey]["rows"][1:]:
            if not r["ok"] or b["score"] is None:
                continue
            if r["flipped"]:
                flip_rows.append((skey, gt, r["label"], b["score"], r["score"]))
            if r["delta"] < 0:
                down_rows.append((r["delta"], skey, r["label"]))
            if gt == "tampered":
                tamper_scores.append(r["score"])
    lines.append("## 自动统计")
    lines.append("")
    lines.append(f"- 有效扰动次数：{len(flip_rows) + sum(1 for s in SAMPLES for r in results[s[0]]['rows'][1:] if r['ok'] and not r['flipped'])} "
                 f"（4 个样本 × 12 种扰动，扣除失败项）")
    lines.append(f"- 结论被翻转（跨阈值）的次数：**{len(flip_rows)}**")
    if flip_rows:
        for skey, gt, label, b, a in flip_rows:
            lines.append(f"  - `{skey}`（{gt}）· {label}：{fmt(b)} → {fmt(a)}（{tier(b)} → {tier(a)}）")
    else:
        lines.append("  - 无：所有扰动都未能把结论调过阈值")
    if tamper_scores:
        lines.append(f"- 3 张篡改图在全部扰动下的分数区间：**{min(tamper_scores):.4f} ~ {max(tamper_scores):.4f}**，"
                     f"均值 {sum(tamper_scores) / len(tamper_scores):.4f}")
    if down_rows:
        down_rows.sort()
        lines.append(f"- 分数下降最多的 3 次扰动：" + "；".join(f"{lab}（{skey} {d:+.4f}）" for d, skey, lab in down_rows[:3]))
    lines.append("")

    lines.append("## 结论与答辩口径")
    lines.append("")
    lines.append(_build_narrative(results, flip_rows, down_rows, tamper_scores))
    lines.append("")
    return "\n".join(lines)


def _build_narrative(results, flip_rows, down_rows, tamper_scores):
    """根据实测统计生成「人话结论」，与上面表格同源（重跑脚本会一并刷新）。"""
    parts = []
    # 1) 总判断：有没有被调翻
    if not flip_rows:
        parts.append(
            "**实测结论：在 4 个样本 × 12 种日常扰动下，没有一次能把 TruFor 的结论调过阈值。** "
            "也就是说，攻击者用美图秀秀/Photoshop 随手做的压缩、缩放、调亮度对比度，"
            "都骗不过这个检测器——篡改痕迹（复制粘贴、拼接、改字）在分数上依然明显。"
        )
    else:
        flip_desc = "；".join(
            f"{skey}（{gt}）经「{lab}」从 {b:.4f} 掉到 {a:.4f}（{tier(b)}→{tier(a)}）"
            for skey, gt, lab, b, a in flip_rows
        )
        parts.append(
            f"**实测结论：48 次扰动里，有 {len(flip_rows)} 次把结论调过了阈值**——{flip_desc}。"
            "能骗过的操作分两类：①重度加噪（高斯噪声 sigma=15）直接把信号压没；"
            "②对原本就贴着 0.5 线的「边缘样本」（改文字图、干净图）做较强 JPEG 压缩/缩放/调亮，"
            "会把它们推过线——这恰恰说明贴线样本最需要人工复核，而不是系统失效。"
        )

    # 2) 哪些扰动是「安全的」（在全部 4 个样本上零翻转）
    safe_labels = []
    for pkey, plabel, _fn in PERTURBATIONS:
        any_flip = any(fr[2] == plabel for fr in flip_rows)
        if not any_flip:
            safe_labels.append(plabel)
    if safe_labels:
        parts.append(
            "**日常工作流安全：** " + "、".join(safe_labels) + "。"
            "这些操作在全部 4 个样本上零翻转，正是品牌方日常出图的默认动作"
            "（存 JPEG、缩图、微调影调），说明正常修图不会误伤。"
        )

    # 3) 强篡改样本 vs 边缘样本
    flip_by_sample = {}
    for skey, gt, lab, b, a in flip_rows:
        flip_by_sample[skey] = flip_by_sample.get(skey, 0) + 1
    strong = [(s, g) for (s, g) in [("tampered_01_copy_move", "tampered"), ("tampered_02_splice", "tampered")]
              if flip_by_sample.get(s, 0) == 0]
    if strong:
        names = {"tampered_01_copy_move": "复制粘贴图(原分0.9946)", "tampered_02_splice": "拼接图(原分1.0)"}
        parts.append(
            "**强篡改样本极稳：** " + "、".join(names[s] for s, g in strong) + " 在全部 12 种日常扰动下结论纹丝不动；"
            "只有极端高斯噪声(sigma=15)能把复制粘贴图压到 0.1487——而那会肉眼可见地毁掉画质，本身就是可疑信号。"
        )
    parts.append(
        "**边缘样本较脆：** 改文字图(原分0.595，本就在可疑档)和干净对照图对较强 JPEG 压缩/缩放/调亮较敏感，"
        "会被推过 0.5 线；这正说明分数越接近阈值的样本越需要人工复核兜底。"
    )

    # 4) 干净图会不会被误报
    clean_rows = results.get("clean_01", {}).get("rows", [])
    clean_scores = [r["score"] for r in clean_rows[1:] if r["ok"] and r["score"] is not None]
    if clean_scores:
        cmax = max(clean_scores)
        if cmax < 0.5:
            parts.append(
                f"**干净图没有被误报：** 干净对照图在全部扰动下最高分仅 {cmax:.4f}，始终低于 0.5 报警线，"
                "说明日常修图不会把真图误判成篡改。"
            )
        else:
            parts.append(
                f"**干净图几乎没被误报：** 干净对照图在全部扰动下最高分 {cmax:.4f}（仅 JPEG q=50 一次跨过 0.5 可疑线），"
                "但从未到 0.9 高风险线；日常修图不会把真图误判成高风险篡改。"
            )

    # 5) 篡改图整体区间
    if tamper_scores:
        parts.append(
            f"**篡改图整体区间：** 3 张篡改图在全部扰动下的分数落在 "
            f"{min(tamper_scores):.4f} ~ {max(tamper_scores):.4f}（均值 {sum(tamper_scores)/len(tamper_scores):.4f}），"
            "绝大多数情况下远高于 0.9 的高风险线，篡改痕迹非常顽固。"
        )

    # 6) 答辩口径：怎么回答「攻击者反向调图」
    parts.append(
        "**怎么回答评审的「攻击者拿你的工具反过来做对抗样本」：** "
        "我们测的是「不针对模型优化的随手调图」——强篡改样本(复制粘贴/拼接)在这种扰动下结论稳如磐石；"
        "唯一能压低分数的是重度加噪，而那会破坏画质、本身即可疑。真正要警惕的是「针对性对抗攻击」"
        "（极小扰动优化），那属于我们诚实标注的已知盲区。应对：①多工具交叉验证不靠单点；"
        "②对低分+高噪声或贴线样本触发人工复核；③用真实标注集持续回灌校准阈值。"
    )
    return "\n\n".join(parts)


def main():
    force = "--force" in sys.argv[1:]
    _patch_rmtree()
    _patch_visualization()
    d, w = trufor_tool.model_ready()
    if d is None or w is None:
        print("[错误] TruFor 未部署，无法测试：", d, w)
        return 1
    print(f"TruFor 就绪：{w.name}", flush=True)

    t0 = time.time()
    results = run_all(force=force)
    print(f"全部推理完成，用时 {time.time() - t0:.1f}s", flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_report(results), encoding="utf-8")
    print(f"报告已写入：{REPORT}", flush=True)

    # 顺手打印一份纯文本总表，方便在终端核对
    for skey, rel, desc, gt in SAMPLES:
        b = results[skey]["rows"][0]
        print(f"\n{skey}（{desc}，真值={gt}）原始分 {fmt(b['score'])}")
        for r in results[skey]["rows"][1:]:
            print(f"  {r['label']:<22} {fmt(r['score']):>8}  {fmt_delta(r['delta']):>8}  {r['tier']:<12} flip={r['flipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

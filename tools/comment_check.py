# -*- coding: utf-8 -*-
# 来源：原创实现 —— 评论真实性特征工程与规则判定为 BeautyProof 团队原创
"""
评论区真实性核验 —— 补齐官方三场景之②，并命中官方点名加分项「矩阵级异常模式识别」

用法：
    python tools/comment_check.py --case cc_01_flood
    python tools/comment_check.py --all
    python tools/comment_check.py --text-file comments.txt     # 每行一条评论

为什么需要它（场景定位）：
    官方给的三个场景里，我们原本只做了「种草内容核验」和「AI 视觉素材鉴伪」，
    缺「评论区真实性核验」。这一场景的特点是**完全不需要图像模型**：
    灌水的本质是「文本模式高度重复 + 缺少个人细节 + 情感一边倒 + 发布节奏异常」，
    全部可用确定性统计特征刻画 —— 这在评审眼里是很干净的一类贡献（规则可复核、无黑箱）。

七项特征（全部可解释、可人工核对）：
    1. 完全重复率      同一句话一字不差反复出现
    2. 近似重复率      3-gram Jaccard 相似度 ≥ 0.75 的评论对占比（抓「换了标点/emoji 的复读」）
    3. 短评率          ≤15 字的评论占比（真实用户会写细节，水军只刷一句）
    4. 无细节率        不含数字/时间/肤质/使用感受等个人化信息的评论占比
    5. 情感单一度      只出现正面词、零负面词的评论占比
    6. 模板集中度      最高频 4-gram 开头短语占比
    7. 节奏爆发度      相邻评论时间间隔的变异系数倒数（批量刷评间隔极短且高度一致）

另有跨内容维度（矩阵级异常，官方点名加分项）：
    8. 跨帖重复率      同一条评论文本出现在多少条不同内容下 —— 正常评论区几乎不会这样

⚠️ 铁律（必须出现在每份输出里）：
    - 这些全是**统计倾向**，不是「这条评论是水军」的证明。真实评论区也可能很短、也可能清一色好评。
    - 本工具**不能**证明某条评论由真人所写、也不能证明整个账号是水军账号。
    - 判定只到「可疑 / 转人工复核」，绝不自动封号 —— 封号是不可逆动作，必须人工。
"""
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── 特征阈值（在 data/comment_cases.json 标注集上标定，勿随手改）──
THRESHOLDS = {
    "near_dup_sim": 0.75,        # 近似重复判定相似度
    "short_len": 15,             # 短评长度阈值（字）
    "high_risk_score": 0.75,     # 组合分 ≥ 此 → 高风险（仍需人工，仅优先级最高）
    "suspicious_score": 0.45,    # 组合分 ≥ 此 → 可疑，转人工
    "burstiness_cv": 0.35,       # 间隔变异系数 < 此 → 视为批量爆发
    "cross_post_repeat_min": 3,  # 同文案出现在 ≥N 条内容下 → 矩阵级异常
}

WEIGHTS = {
    "dup_exact_rate": 1.00,
    "near_dup_rate": 1.20,
    "short_rate": 0.60,
    "no_detail_rate": 1.00,
    "positive_uniformity": 0.90,
    "template_concentration": 0.80,
    "burstiness": 0.70,
}

# 强信号阈值：单条命中即算一条独立证据。
# 为什么需要"证据计数"而不只是加权平均（这是实测踩出来的）：
#   第一版只有加权平均。cc_02 那条灌水样本 short_rate / no_detail_rate /
#   positive_uniformity / template_concentration / burstiness **五项全部拉满**，
#   但因为近似重复率被标点稀释成 0.25，加权平均后只剩 0.69 → 判成 suspicious。
#   但取证逻辑应该是「多条互相独立的证据同时命中 = 强」，而不是「求平均」。
#   所以补一套显式的强信号阈值 + 计数规则（阈值全部写在代码里可复核）。
STRONG_SIGNALS = {
    "dup_exact_rate": 0.30,           # 三成以上一字不差重复
    "near_dup_rate": 0.30,            # 三成以上评论对高度相似
    "no_detail_rate": 0.85,           # 绝大多数评论零个人化细节
    "positive_uniformity": 0.90,      # 九成以上只有正面词、零负面
    "template_concentration": 0.50,   # 半数以上复用同一开头模板
    "burstiness": 0.80,               # 发布节奏异常整齐
}

# ── 词表（公开可查的常见表达，用于统计倾向，不是情感模型）──
POSITIVE_WORDS = [
    "好", "很好", "不错", "喜欢", "爱了", "推荐", "回购", "好用", "惊艳", "绝绝子",
    "yyds", "真香", "划重点", "闭眼入", "冲", "不踩雷", "宝藏", "绝了", "封神",
    "亲妈", "无限回购", "已经回购三次", "空瓶", "囤了", "性价比高", "超出预期",
]
NEGATIVE_WORDS = [
    "不好", "一般", "失望", "踩雷", "别买", "浪费", "难用", "油腻", "假白",
    "闷痘", "过敏", "刺痛", "没效果", "智商税", "劝退", "后悔",
]
# 个人化细节：数字 / 时间 / 肤质 / 使用部位或感受 —— 真实用户常带，水军常不带
DETAIL_PATTERNS = [
    re.compile(r"\d"),                                   # 任意数字（色号、天数、克数、价格）
    re.compile(r"(天|周|月|年|小时|分钟|早上|晚上|睡前|晨起|去年|今年)"),
    re.compile(r"(油皮|干皮|混合皮|混油|敏感肌|痘痘肌|中性皮|暗沉|毛孔粗大)"),
    re.compile(r"(鼻翼|下巴|额头|两颊|法令纹|黑眼圈|毛孔|出油|卡粉|起皮)"),
    re.compile(r"(上脸|涂上|抹上|推开|吸收|闷|搓泥|假白|不服帖|很润|很干)"),
]
# 开头模板（抓「同句式复读」）
TEMPLATE_N = 4


def _norm(s):
    """归一化：去空白 + 去标点 + 去符号（含 emoji）。

    ⚠️ 去标点/符号是必须的（实测）：水军最常见的伪装就是在复读时加标点或 emoji
    （「绝了」→「绝了!!」「绝了👀」）。第一版用字符类正则过滤，漏了标点且有转义警告，
    导致 3-gram 完全不同、相似度被严重低估（0.25 而非接近 1.0），整条判罚被拖成 suspicious。
    改用 unicodedata 类别判断：P*=标点、S*=符号（emoji 绝大多数落在 S 或 So），
    比维护一个越来越长的字符类可靠，也不会再出现 invalid escape sequence 警告。
    """
    s = str(s or "")
    out = []
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        out.append(ch)
    return "".join(out)


def _ngrams(s, n=TEMPLATE_N):
    s = _norm(s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def _jaccard(a, b):
    A, B = _ngrams(a), _ngrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _has_detail(text):
    return any(p.search(text) for p in DETAIL_PATTERNS)


def _sentiment(text):
    t = str(text or "")
    pos = sum(1 for w in POSITIVE_WORDS if w in t)
    neg = sum(1 for w in NEGATIVE_WORDS if w in t)
    return pos, neg


def analyze_comments(comments, timestamps=None, cross_post_index=None, min_n=5):
    """核心分析。返回特征 + 组合分 + 四档结论。

    参数：
        comments        评论列表（str 或 {"text":..., "ts":..., "account":...}）
        timestamps      可选，与 comments 等长的发布时间戳（秒）
        cross_post_index 可选，{评论文本: 出现过的内容条数}，用于矩阵级异常
        min_n           少于这么多条不出结论（样本太少，统计无意义）
    """
    items = []
    for i, c in enumerate(comments):
        if isinstance(c, dict):
            text = c.get("text", "")
            ts = c.get("ts", (timestamps or [None] * len(comments))[i] if timestamps else None)
            acct = c.get("account")
        else:
            text, ts, acct = c, (timestamps or [None] * len(comments))[i] if timestamps else None, None
        items.append({"text": str(text or ""), "ts": ts, "account": acct,
                      "norm": _norm(text)})

    n = len(items)
    base = {
        "comment_count": n,
        "min_n_required": min_n,
        "thresholds": THRESHOLDS,
    }
    if n < min_n:
        base.update({
            "verdict": "inconclusive",
            "headline": f"评论只有 {n} 条，少于 {min_n} 条，不出结论",
            "reasons": [f"样本量不足（{n} < {min_n}），任何比例型指标都不可靠，转人工看一眼即可"],
            "score": None,
        })
        return base

    # 1) 完全重复率
    norms = [it["norm"] for it in items if it["norm"]]
    dup_exact_rate = 1.0 - (len(set(norms)) / len(norms)) if norms else 0.0

    # 2) 近似重复率（评论对的占比，n≤120 时全对比，超过则抽样避免 O(n^2) 爆炸）
    pairs = []
    idx = list(range(n))
    if n <= 120:
        for a in range(n):
            for b in range(a + 1, n):
                pairs.append((a, b))
    else:
        step = n // 120 + 1
        for a in range(0, n, step):
            for b in range(a + step, n, step):
                pairs.append((a, b))
    near_hit = 0
    for a, b in pairs:
        if _jaccard(items[a]["text"], items[b]["text"]) >= THRESHOLDS["near_dup_sim"]:
            near_hit += 1
    near_dup_rate = near_hit / len(pairs) if pairs else 0.0

    # 3) 短评率
    short_rate = sum(1 for it in items if len(_norm(it["text"])) <= THRESHOLDS["short_len"]) / n

    # 4) 无细节率
    no_detail_rate = sum(1 for it in items if not _has_detail(it["text"])) / n

    # 5) 情感单一度：只有正面、零负面
    uni = 0
    for it in items:
        pos, neg = _sentiment(it["text"])
        if pos > 0 and neg == 0:
            uni += 1
    positive_uniformity = uni / n

    # 6) 模板集中度：最高频开头 n-gram 占比
    grams = {}
    for it in items:
        for g in _ngrams(it["text"]):
            grams[g] = grams.get(g, 0) + 1
    template_concentration = (max(grams.values()) / n) if grams else 0.0

    # 7) 节奏爆发度：间隔变异系数的倒数（越集中越像批量）
    ts = sorted(x for x in (it["ts"] for it in items) if x is not None)
    burstiness = None
    if len(ts) >= 3:
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        gaps = [g for g in gaps if g >= 0]
        if gaps and sum(gaps) > 0:
            cv = (statistics.pstdev(gaps) / (sum(gaps) / len(gaps))) if len(gaps) > 1 else 0.0
            burstiness = max(0.0, min(1.0, 1.0 - cv))   # 间隔越整齐 → 越接近 1

    # 8) 跨帖重复率（矩阵级异常）
    cross_repeat_max = 0
    cross_repeat_ratio = 0.0
    if cross_post_index:
        counts = [cross_post_index.get(it["norm"], 1) for it in items]
        cross_repeat_max = max(counts) if counts else 0
        cross_repeat_ratio = sum(1 for c in counts if c >= THRESHOLDS["cross_post_repeat_min"]) / n

    feats = {
        "dup_exact_rate": round(dup_exact_rate, 4),
        "near_dup_rate": round(near_dup_rate, 4),
        "short_rate": round(short_rate, 4),
        "no_detail_rate": round(no_detail_rate, 4),
        "positive_uniformity": round(positive_uniformity, 4),
        "template_concentration": round(template_concentration, 4),
        "burstiness": (round(burstiness, 4) if burstiness is not None else None),
        "cross_post_repeat_max": cross_repeat_max,
        "cross_post_repeat_ratio": round(cross_repeat_ratio, 4),
    }

    # ── 组合分（透明加权，权重写在代码里可复核）──
    parts = {}
    wsum = 0.0
    for k, w in WEIGHTS.items():
        v = feats.get(k)
        if v is None:
            continue
        parts[k] = round(float(v), 4)
        wsum += w
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / wsum if wsum else 0.0
    # 矩阵级异常是"多内容共用同一批评论"这种结构性证据，命中就直接抬到高权重
    if cross_repeat_max >= THRESHOLDS["cross_post_repeat_min"]:
        score = min(1.0, score + 0.25)

    reasons = []
    if feats["near_dup_rate"] >= 0.15:
        reasons.append(f"近似重复评论占比 {feats['near_dup_rate']:.0%}"
                       f"（相似度≥{THRESHOLDS['near_dup_sim']}，已忽略标点与 emoji 差异），存在复读式模板")
    if feats["dup_exact_rate"] >= 0.20:
        reasons.append(f"一字不差的重复评论占比 {feats['dup_exact_rate']:.0%}")
    if feats["no_detail_rate"] >= 0.70:
        reasons.append(f"{feats['no_detail_rate']:.0%} 的评论不含任何个人化细节"
                       f"（数字/时间/肤质/使用感受全无）")
    if feats["positive_uniformity"] >= 0.80:
        reasons.append(f"{feats['positive_uniformity']:.0%} 的评论只有正面词、零负面词，情感一边倒")
    if feats["template_concentration"] >= 0.30:
        reasons.append(f"最常见的开头短语被 {feats['template_concentration']:.0%} 的评论复用")
    if burstiness is not None and feats["burstiness"] >= 1 - THRESHOLDS["burstiness_cv"]:
        reasons.append("评论发布节奏异常整齐（间隔变异系数低），像批量投放")
    if cross_repeat_max >= THRESHOLDS["cross_post_repeat_min"]:
        reasons.append(f"**矩阵级异常**：同一条评论文本出现在 {cross_repeat_max} 条不同内容下，"
                       f"正常用户不会跨这么多条内容发同一句话")

    if not reasons:
        reasons.append("未见明显的复读/模板/情感一边倒/节奏异常特征")

    # ── 强信号计数：多条互相独立的证据同时命中，比任何单一分数都可靠 ──
    fired = [k for k, t in STRONG_SIGNALS.items()
             if feats.get(k) is not None and feats.get(k) >= t]
    matrix_hit = cross_repeat_max >= THRESHOLDS["cross_post_repeat_min"]

    if matrix_hit:
        # 结构性证据：同一批评论跨多条内容复用，这是 botnet 的典型签名，
        # 强度足以单独进入高风险（但仍只到「转人工」，绝不自动封号）。
        verdict = "high_risk"
        headline = "评论区呈现矩阵级异常（跨内容复用同一批评论），建议立即人工复核"
    elif len(fired) >= 4:
        verdict = "high_risk"
        headline = f"评论区有 {len(fired)} 条独立特征同时指向批量灌水，建议立即人工复核"
    elif len(fired) >= 3 or score >= THRESHOLDS["suspicious_score"]:
        verdict = "suspicious"
        headline = "评论区存在可疑模式，建议人工看一眼"
    else:
        verdict = "credible"
        headline = "评论区未见明显灌水模式"

    if fired:
        reasons.append(f"命中 {len(fired)} 项独立强信号：{'、'.join(fired)}")

    return {
        "comment_count": n,
        "features": feats,
        "score": round(score, 4),
        "strong_signals_fired": fired,
        "n_strong_signals": len(fired),
        "matrix_anomaly": matrix_hit,
        "verdict": verdict,
        "headline": headline,
        "reasons": reasons,
        "thresholds": THRESHOLDS,
        "weights": WEIGHTS,
        "strong_signal_thresholds": STRONG_SIGNALS,
    }


def analyze_account(posts):
    """账号画像（虚假种草账号画像，官方点名加分项）。

    posts: [{"text":..., "published_at":...}, ...] 同一账号的多条内容
    输出：文案模板复用率、发布间隔规律性、平均篇幅、平均正面词密度。
    ⚠️ 同样只是统计倾向，不能证明账号是水军。
    """
    texts = [_norm(p.get("text", "")) for p in posts if p.get("text")]
    n = len(texts)
    if n < 2:
        return {"post_count": n, "verdict": "inconclusive",
                "note": "同一账号至少要有 2 条内容才能做复用率与节奏分析"}

    # 文案模板复用：两两 3-gram Jaccard 相似度均值
    sims = []
    for a in range(n):
        for b in range(a + 1, n):
            sims.append(_jaccard(texts[a], texts[b]))
    reuse = sum(sims) / len(sims) if sims else 0.0

    ts = sorted(p["published_at"] for p in posts if p.get("published_at"))
    regularity = None
    if len(ts) >= 3:
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        if gaps and sum(gaps) > 0:
            cv = statistics.pstdev(gaps) / (sum(gaps) / len(gaps))
            regularity = max(0.0, min(1.0, 1.0 - cv))

    lens = [len(t) for t in texts]
    pos_density = []
    for t in texts:
        pos, _ = _sentiment(t)
        pos_density.append(pos)

    feats = {
        "post_count": n,
        "text_reuse_rate": round(reuse, 4),
        "publish_regularity": (round(regularity, 4) if regularity is not None else None),
        "mean_length": round(sum(lens) / len(lens), 1),
        "length_cv": (round(statistics.pstdev(lens) / (sum(lens) / len(lens)), 4) if len(lens) > 1 else None),
        "mean_positive_hits": round(sum(pos_density) / len(pos_density), 3),
    }
    flags = []
    if reuse >= 0.5:
        flags.append(f"同一账号文案两两相似度均值 {reuse:.0%}，高度模板化")
    if regularity is not None and regularity >= 0.7:
        flags.append(f"发布间隔高度规律（变异系数低），疑似定时批量投放")
    if feats["mean_positive_hits"] >= 3:
        flags.append(f"平均每条命中 {feats['mean_positive_hits']:.1f} 个正面词，情感表述一边倒")

    return {
        **feats,
        "flags": flags,
        "verdict": "suspicious" if flags else "credible",
        "cannot_prove": "账号画像只是统计倾向，不能证明该账号由水军运营；封号是不可逆动作，必须人工决策",
    }


def build_evidence(comments, source_id="inline", timestamps=None, cross_post_index=None, account_posts=None):
    res = analyze_comments(comments, timestamps, cross_post_index)
    acct = analyze_account(account_posts) if account_posts else None
    return {
        "tool": "comment",
        "source_asset_id": source_id,
        "observed": (f"评论区核验：{res['comment_count']} 条评论，组合分 "
                     f"{res.get('score')}，结论 {res['verdict']}。{res['headline']}"),
        "cannot_prove": (
            "以上全部是统计倾向，不是「这条评论是水军」的证明。"
            "真实评论区也可能普遍很短、或清一色好评（品牌方组织过试用活动时尤其如此）。"
            "本工具不能证明评论由真人所写，也不能证明整个账号是水军账号。"
            "结论只到「转人工复核」，绝不自动封号。"
        ),
        "evidence": [{
            **{k: v for k, v in res.items() if k != "thresholds"},
            "thresholds": THRESHOLDS,
            "weights": WEIGHTS,
            "account_profile": acct,
        }],
    }


def load_cases():
    d = json.loads((REPO / "data" / "comment_cases.json").read_text(encoding="utf-8"))
    cases = d.get("cases") if isinstance(d, dict) else d
    if isinstance(cases, dict):
        cases = list(cases.values())
    return d if isinstance(d, dict) else {}, cases or []


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/comment_check.py --case ID | --all | --text-file F")
        return 1

    if "--all" in sys.argv:
        meta, cases = load_cases()
        rows = []
        ok = 0
        print(f"{'case':26s} {'期望':10s} {'实得':10s} {'分数':>6s}  判定")
        print("-" * 78)
        for c in cases:
            r = analyze_comments(c["comments"], c.get("timestamps"), c.get("cross_post_index"))
            exp = c["expected_verdict"]
            got = r["verdict"]
            hit = (exp == got)
            ok += 1 if hit else 0
            rows.append({
                "id": c["id"], "expected": exp, "got": got, "hit": hit,
                "score": r.get("score"), "n": r["comment_count"],
                "features": r.get("features"),
            })
            print(f"{c['id']:26s} {exp:10s} {got:10s} {str(r.get('score')):>6s}  {'✅' if hit else '❌'}")
        n = len(cases)
        print("-" * 78)
        print(f"方向命中 {ok}/{n} = {ok/n:.1%}")
        out = {
            "schema": "beautyproof/comment_eval@1",
            "data_provenance": meta.get("_provenance"),
            "disclaimer": meta.get("_disclaimer"),
            "n_cases": n, "direction_hit": ok, "direction_rate": round(ok / n, 4) if n else None,
            "rows": rows,
        }
        p = REPO / "results" / "comment_eval.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写: {p}")
        return 0

    if "--case" in sys.argv:
        cid = sys.argv[sys.argv.index("--case") + 1]
        _, cases = load_cases()
        hit = [c for c in cases if c["id"] == cid]
        if not hit:
            print(f"没找到 {cid}")
            return 1
        c = hit[0]
        rep = build_evidence(c["comments"], c["id"], c.get("timestamps"), c.get("cross_post_index"))
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        out = REPO / "outputs" / f"comment_{c['id']}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存到: {out}")
        return 0

    if "--text-file" in sys.argv:
        f = Path(sys.argv[sys.argv.index("--text-file") + 1])
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        rep = build_evidence(lines, f.stem)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print("用法: python tools/comment_check.py --case ID | --all | --text-file F")
    return 1


if __name__ == "__main__":
    sys.exit(main())

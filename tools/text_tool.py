# -*- coding: utf-8 -*-
"""
文案体检工具 —— 给一段种草文 / 评论区文案做检查，输出统一格式的证据 JSON

用法：
    python tools/text_tool.py "姐妹们这个真的绝了，七天美白，亲测有效！"
    python tools/text_tool.py --file 文案.txt          # 一个文件 = 一段文案
    python tools/text_tool.py --comments 评论.txt      # 一行一条评论，查刷量特征

大白话：
    前面六个工具都在「看图」，这个工具开始「读字」。
    它做三件事：
      1. 查违禁宣称 —— 化妆品广告不能说的话（有法律条文依据，是硬规则）
      2. 看写作特征 —— 像不像 AI 批量生成的（只是提示，不作判定）
      3. 查评论刷量 —— 一堆评论是不是同一个模板套出来的

能证明什么：
    这段文字里出现了哪些《广告法》《化妆品监督管理条例》点名的敏感宣称，
    出现在第几个字、上下文是什么 —— 可逐条核对，可解释。

不能证明什么（铁律）：
    ❌ 不能证明这段文案是 AI 写的。写作特征只是统计现象，真人也会这么写，
       AI 也能写得很像人。这一项**永远只提示、不判定、不参与自动定级**。
    ❌ 不能证明这段文案是假的 / 是水军。刷量特征同样只是现象。
    ❌ 命中敏感词 ≠ 违法。是否构成违法由监管部门认定，这里只做「值得人工看一眼」的提示。

为什么这么保守：
    图像侧我们能用「归一化后指标不降反升」证明模型学的是真伪信号；
    文本侧我们没有同等强度的验证，所以只给可核对的硬规则，
    不给自己没验证过的东西编一个概率分。分数没校准过，就不说它是概率。
"""
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- 违禁宣称词典
# 依据（中国大陆，化妆品品类）：
#   《广告法》第 9 条   —— 禁用"国家级/最高级/最佳"等绝对化用语
#   《广告法》第 17 条  —— 非医疗广告不得涉及疾病治疗功能
#   《化妆品监督管理条例》第 43 条 —— 化妆品广告不得明示或暗示产品具有医疗作用
#   《化妆品监督管理条例》第 43 条 —— 不得含有虚假或者引人误解的内容、不得夸大功效
#
# ⚠️ 收录原则：宁可少收、不可乱收。每个词都能对上上面某一条才收进来。
#    命中只是"值得人工核对"的提示，不等于违法认定。
CLAIM_LEXICON = {
    "medical": {
        "label": "明示或暗示医疗作用",
        "basis": "《化妆品监督管理条例》第43条 / 《广告法》第17条",
        "severity": "high",
        "words": [
            "治疗", "治愈", "疗效", "根治", "根治", "消除炎症", "消炎", "抗菌", "杀菌",
            "灭菌", "抗病毒", "药用", "药效", "处方", "麻醉", "解毒", "抗过敏",
            "祛痘印", "祛疤", "去疤", "牛皮癣", "湿疹", "皮炎", "激素", "抗生素",
            "医用", "临床验证", "遵医嘱",
        ],
    },
    "absolute": {
        "label": "绝对化用语",
        "basis": "《广告法》第9条",
        "severity": "medium",
        "words": [
            "国家级", "世界级", "最高级", "最佳", "最好", "最优", "最有效", "最强",
            "第一品牌", "全球第一", "全网第一", "独一无二", "唯一指定", "绝无仅有",
            "100%有效", "百分之百", "永久", "永远", "彻底", "完全去除", "绝不反弹",
            "零风险", "无副作用",
        ],
    },
    "promise": {
        "label": "夸大功效 / 效果时限承诺",
        "basis": "《化妆品监督管理条例》第43条（不得夸大功效）",
        "severity": "medium",
        "words": [
            "立竿见影", "瞬间", "即刻见效", "一秒", "一分钟", "三天美白", "七天美白",
            "三天祛斑", "七天祛斑", "一周祛皱", "一夜回春", "无效退款", "无效退全款",
            "根治", "永不反弹", "一次见效", "永久美白", "永久去皱",
        ],
    },
}

# AI 写作常见特征词（弱信号，只统计不判定）
AI_STYLE_MARKERS = [
    "首先", "其次", "再次", "再者", "此外", "另外", "同时", "综上", "总之",
    "总而言之", "因此", "由此可见", "值得注意的是", "值得一提", "一方面", "另一方面",
    "不仅", "而且", "需要指出的是", "总的来说", "换句话说",
]

# 美妆种草营销套话（弱信号）
TEMPLATE_MARKERS = [
    "姐妹们", "宝子们", "集美们", "家人们", "闭眼入", "冲", "绝绝子", "yyds",
    "无限回购", "亲测有效", "真的绝了", "谁用谁知道", "空瓶", "回购N次",
    "学生党", "贫民窟", "平替", "天花板", "宝藏", "避雷",
]

# 具体细节信号（有这些更像真人真实体验）
DETAIL_NUMBER = re.compile(r"\d+(\.\d+)?")
DETAIL_TIME = re.compile(r"(第?\d+\s*(天|周|月|年|次|小时|分钟))|(早上|晚上|睡前|晨起|两周|一个月)")
DETAIL_SHADE = re.compile(r"[A-Za-z]*\d{2,3}\s*(号|色)?|(色号)|(#\w+)")


def scan_claims(text):
    """逐条扫违禁宣称词典，返回命中列表（含位置与上下文，可人工核对）"""
    hits = []
    for category, spec in CLAIM_LEXICON.items():
        for word in spec["words"]:
            start = 0
            while True:
                idx = text.find(word, start)
                if idx < 0:
                    break
                left = max(0, idx - 12)
                right = min(len(text), idx + len(word) + 12)
                hits.append({
                    "category": category,
                    "category_label": spec["label"],
                    "basis": spec["basis"],
                    "severity": spec["severity"],
                    "word": word,
                    "position": idx,
                    "context": text[left:right],
                })
                start = idx + len(word)
    hits.sort(key=lambda x: (-{"high": 2, "medium": 1}.get(x["severity"], 0), x["position"]))
    return hits


def ai_style_features(text):
    """统计写作特征 —— 全是弱信号，只用于提示，不用于判定"""
    if not text:
        return {}

    # 按标点断句
    sentences = [s for s in re.split(r"[。！？!?；;\n]", text) if s.strip()]
    lengths = [len(s) for s in sentences]

    n = len(text)
    per100 = (lambda c: round(c * 100 / max(n, 1), 2))

    ai_hits = sum(text.count(w) for w in AI_STYLE_MARKERS)
    tpl_hits = sum(text.count(w) for w in TEMPLATE_MARKERS)

    if len(lengths) >= 2:
        mean = sum(lengths) / len(lengths)
        var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        cv = round((var ** 0.5) / mean, 3) if mean else 0.0
    else:
        mean = lengths[0] if lengths else 0
        cv = 0.0

    return {
        "char_count": n,
        "sentence_count": len(sentences),
        "avg_sentence_len": round(mean, 1),
        "sentence_len_cv": cv,
        "ai_marker_per_100": per100(ai_hits),
        "template_marker_per_100": per100(tpl_hits),
        "has_number": bool(DETAIL_NUMBER.search(text)),
        "has_time_detail": bool(DETAIL_TIME.search(text)),
        "has_shade_detail": bool(DETAIL_SHADE.search(text)),
    }


def comment_flood_features(lines):
    """评论区刷量特征 —— 同样只是现象，不判定"""
    items = [l.strip() for l in lines if l.strip()]
    if len(items) < 2:
        return {}
    unique = len(set(items))
    short = sum(1 for l in items if len(l) <= 15)
    return {
        "comment_count": len(items),
        "unique_count": unique,
        "unique_ratio": round(unique / len(items), 3),
        "short_comment_ratio": round(short / len(items), 3),
        "note": "重复率高 / 短评占比高只是刷量的常见现象，也可能是评论区本来就这么聊，不能据此判定刷量",
    }


def build_evidence(text, source_id="inline", mode="single"):
    lines = text.splitlines() if mode == "comments" else [text]

    claims = scan_claims(text)
    style = ai_style_features(text)
    flood = comment_flood_features(lines) if mode == "comments" else {}

    high = sum(1 for c in claims if c["severity"] == "high")
    medium = sum(1 for c in claims if c["severity"] == "medium")
    if high:
        claim_level = "high"
    elif medium:
        claim_level = "medium"
    else:
        claim_level = "none"

    if claims:
        top = claims[0]
        observed = (f"文案共 {style.get('char_count', 0)} 字，命中 {len(claims)} 处敏感宣称"
                    f"（高风险 {high} 处、中风险 {medium} 处）。"
                    f"最扎眼的一处：「{top['word']}」（{top['category_label']}，依据 {top['basis']}），"
                    f"上下文 …{top['context']}…")
    else:
        observed = f"文案共 {style.get('char_count', 0)} 字，未命中违禁宣称词典中的任何词条。"

    if flood:
        observed += f" 评论 {flood['comment_count']} 条，去重后 {flood['unique_count']} 条（重复率 {1 - flood['unique_ratio']:.0%}）。"

    return {
        "tool": "text",
        "source_asset_id": source_id,
        "observed": observed,
        "cannot_prove": (
            "文案工具只能核对「有没有出现法律点名的敏感宣称」，"
            "不能证明这段文案是 AI 写的、也不能证明它是假的或水军刷的；"
            "写作特征与重复率仅为统计提示，未校准为概率，不参与自动定级"
        ),
        "evidence": [{
            "mode": mode,
            "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
            "claim_violations": claims,
            "claim_violation_count": len(claims),
            "claim_high_count": high,
            "claim_medium_count": medium,
            "claim_risk_level": claim_level,
            "ai_style": style,
            "comment_flood": flood,
            "calibrated": False,
        }],
    }


def main():
    args = sys.argv[1:]
    if not args:
        print("用法: python tools/text_tool.py \"文案内容\"")
        print("      python tools/text_tool.py --file 文案.txt")
        print("      python tools/text_tool.py --comments 评论.txt   # 一行一条")
        return 1

    mode = "single"
    if args[0] == "--file":
        p = Path(args[1])
        if not p.exists():
            print(f"找不到文件: {p}")
            return 1
        text = p.read_text(encoding="utf-8")
        source_id = p.name
    elif args[0] == "--comments":
        p = Path(args[1])
        if not p.exists():
            print(f"找不到文件: {p}")
            return 1
        text = p.read_text(encoding="utf-8")
        source_id = p.name
        mode = "comments"
    else:
        text = " ".join(args)
        source_id = "inline"

    report = build_evidence(text, source_id=source_id, mode=mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

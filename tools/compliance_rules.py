# -*- coding: utf-8 -*-
# 来源：法律法规公开条文 + 公开处罚案例（逐条附来源链接）；代码实现为 BeautyProof 团队原创
"""
合规法条引擎 —— 把「文案体检命中敏感词」升级成「可交付法务的合规风险清单」

用法：
    python tools/compliance_rules.py --text "文案内容"
    python tools/compliance_rules.py --post xs_01
    python tools/compliance_rules.py --all      # 扫 data/xhs_posts.json 全部真实文案

为什么要做这个（这是本模块存在的唯一理由）：
    原来的 text_tool 已经能查出「命中了哪个敏感词、依据是哪条法规」，
    但对**品牌方/法务最关心的两个问题**没有回答：
      ① 这条宣称在法规上到底违不违？违的是第几条？
      ② 违了会怎么处理？可能罚多少钱？
    只报「你命中了『治疗』这个词」没有可操作性；报「条例第37条禁止明示暗示医疗作用，
    顶格处罚区间 20~100 万元，相似案例：某连锁药店标注『美白淡斑』而备案仅『修护保湿』
    被罚 2000 元」—— 这才是法务能直接用的东西。

⚠️ 铁律（必须写进每一份输出）：
    本模块是**风险提示**，不是法律意见。是否构成违法、怎么罚，由监管机构认定。
    引用条款与案例均来自公开渠道，但法规会修订，落地前请由法务复核最新条文。

数据来源（全部可公开核验）：
    《化妆品监督管理条例》全文与条文解读见下方 law_ref；
    《中华人民共和国广告法》第九条/第十七条/第二十八条；
    《反不正当竞争法》第八条（虚假宣传）；
    《消费者权益保护法》第五十五条（退一赔三）；
    处罚案例见 case_ref。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_tool import scan_claims  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# ── 法条知识库 ────────────────────────────────────────────────
# penalty 用文字描述而不是数字区间上限，因为金额上限常随情节/上次违法数浮动，
# 写死一个数字容易被当成"确定会罚这么多"，反而误导。
LAW_CATALOG = {
    "medical_claim": {
        "法名": "《化妆品监督管理条例》第三十七条第（二）项",
        "要点": "化妆品标签禁止明示或者暗示产品具有医疗作用",
        "处罚": "由市场监管部门责令改正；拒不改正的，处违法所得三倍以下罚款；"
                "情节严重的，吊销化妆品经营许可证；"
                "构成犯罪的，依法追究刑事责任",
        "law_ref": "https://www.gov.cn/zhengce/content/2020-06/29/content_5522593.htm",
        "case_ref": "https://m.cqn.com.cn/zj/content/2026-08/03/content_9166816.htm",
        "case": "南京某化妆品店在身体乳包装标注「美白淡斑」，而产品备案仅为「修护保湿」，"
                "依《条例》第四十三条与《广告法》第二十八条被罚款 2000 元",
        "note": "「暗示医疗作用」的边界由监管认定，例如「祛痘」「抗炎」「修复屏障受损」"
                "常被按医疗用语处理，本工具只做提示。",
    },
    "efficacy_claim": {
        "法名": "《化妆品监督管理条例》第二十二条",
        "要点": "化妆品功效宣称应当有充分的科学依据，"
                "且应当在国家药监局指定网站公布功效宣称所依据的文献资料与数据摘要",
        "处罚": "未按规定公开功效宣称依据的，由市场监管部门责令改正；"
                "情节严重的，处违法所得三倍以下罚款",
        "law_ref": "https://www.gov.cn/zhengce/content/2020-06/29/content_5522593.htm",
        "case_ref": "https://big5.china.com.cn/gate/big5/news.china.com.cn/2026-04/13/content_118433065.shtml",
        "case": "屈臣氏武汉门店宣传「92% 原液含量」，实测为 0.414%，被认定为虚假宣传并处罚",
        "note": "本工具无法联网核验品牌方的功效依据是否已在药监局公示，"
                "因此这一条只提示「该宣称需要能拿出公开文献与数据摘要」。",
    },
    "special_category": {
        "法名": "《化妆品监督管理条例》第十六条",
        "要点": "祛斑美白、防晒、防脱发、染发烫发等属特殊化妆品，须取得注册证；"
                "普通化妆品不得宣称具有上述功效",
        "处罚": "以普通化妆品冒充特殊化妆品销售的，责令改正、没收违法所得，"
                "并处违法所得三倍以下罚款；情节严重的，吊销许可证",
        "law_ref": "https://www.gov.cn/zhengce/content/2020-06/29/content_5522593.htm",
        "case_ref": "https://www.yingde.gov.cn/yqydscjd/gkmlpt/content/2/2120/post_2120970.html",
        "case": "英德市一处销售「止脱育发」套盒，实际为普通化妆品，被没收并处罚款合计 1.14 万元",
        "note": "关键点是**备案类别**与宣称是否匹配 —— 需拿产品备案信息才能最终认定。",
    },
    "absolute_claim": {
        "法名": "《广告法》第九条",
        "要点": "广告不得使用「国家级」「最高级」「最佳」等绝对化用语",
        "处罚": "由市场监管部门责令停止发布广告，责令广告主在相应范围内消除影响，"
                "处广告费用三倍以上五倍以下的罚款，情节严重的吊销营业执照、吊销广告发布登记证件",
        "law_ref": "https://www.gov.cn/guoqing/2021-10/29/content_5647633.htm",
        "case_ref": None,
        "case": None,
        "note": "《化妆品监督管理条例》第四十三条亦有对应表述，两处可并用。",
    },
    "false_ad": {
        "法名": "《广告法》第二十八条 / 《反不正当竞争法》第八条",
        "要点": "广告对商品的性能、功能、产地、用途、质量、成分、价格、生产者、"
                "有效期限、销售状况、曾获荣誉等信息作虚假或引人误解的宣传",
        "处罚": "《广告法》：处广告费用三倍以上五倍以下罚款，情节严重的吊销营业执照；"
                "《反不正当竞争法》：处二十万元以上一百万元以下罚款，"
                "情节严重的处一百万元以上二百万元以下罚款，可吊销营业执照",
        "law_ref": "https://www.gov.cn/guoqing/2021-10/29/content_5647633.htm",
        "penalty_note": "这是本模块里金额上限最高的一档：虚假宣传 20~100 万元，"
                        "情节严重 100~200 万元并可吊销执照；"
                        "消费者还可依《消费者权益保护法》第五十五条主张退一赔三。",
        "case_ref": "https://big5.china.com.cn/gate/big5/news.china.com.cn/2026-04/13/content_118433065.shtml",
        "case": "屈臣氏武汉门店「92% 原液含量」宣传不实，属虚假宣传，"
                "依《反不正当竞争法》第八条作出行政处罚",
        "note": "处罚基数是「广告费用」或「违法所得」，不是销售额 —— 这是很多品牌最容易算错的地方。",
    },
    "timed_promise": {
        "法名": "《广告法》第九条 / 《化妆品监督管理条例》第四十三条",
        "要点": "不得对功效作绝对化或违反科学、常识的承诺；"
                "「见效快」「永久」「一次见效」等属效果时限承诺，缺少科学依据即涉嫌夸大",
        "处罚": "同 absolute_claim / false_ad 两档",
        "law_ref": "https://www.gov.cn/guoqing/2021-10/29/content_5647633.htm",
        "case_ref": None,
        "case": None,
        "note": "化妆品属普通商品，不能宣称疾病治疗作用；「见效速度」需有临床评价支撑。",
    },
}

# text_tool 的 CLAIM_LEXICON 键 → 法条知识库键 的映射
# ⚠️ 键名必须与 tools/text_tool.py 里 CLAIM_LEXICON 的键逐字对齐（medical/absolute/promise），
#    对不上会静默退化成"无法定档"，所以下面有自检。
CATEGORY_TO_LAW = {
    "medical": "medical_claim",
    "promise": "timed_promise",
    "absolute": "absolute_claim",
    # 标签兜底（若将来 text_tool 改了 label 但没改 category，仍能命中）
    "明示或暗示医疗作用": "medical_claim",
    "绝对化用语": "absolute_claim",
    "夸大功效 / 效果时限承诺": "timed_promise",
}

# 特殊化妆品功效词（《条例》第十六条）—— 需要产品注册证才能宣称
SPECIAL_CATEGORY_PAT = re.compile(
    r"(祛斑|美白|防晒|防脱发|防脱|育发|染发|烫发|脱毛|去角质|清洁\s*黑头)"
)

# 行业侧可引用的量化背景（用于商业价值页，来源见 docs/合规与行业数据引用.md）
INDUSTRY_FACTS = [
    {
        "fact": "2025 年中国化妆品全渠道交易额 11042.45 亿元，同比增长 2.83%",
        "use": "说明市场体量足够大，值得投入内容鉴真",
        "source": "https://dzb.xfrb.com.cn/Html/2026-05-18/74159.html",
    },
    {
        "fact": "行业平均营销费用率从 2021 年 33.36% 升至 2025 年 45.53%",
        "use": "营销投入越大，虚假宣称的动机越强，鉴伪的 ROI 越高",
        "source": "https://dzb.xfrb.com.cn/Html/2026-05-18/74159.html",
    },
    {
        "fact": "2025 年 12315 受理美妆类投诉约 25 万条，涉虚假宣传、资质、售后安全",
        "use": "说明虚假宣传已经是真实的监管风险，不是假设",
        "source": "https://thepaper.cn/newsDetail_forward_33152973",
    },
    {
        "fact": "46.43% 的消费者认为护肤品行业存在「效果不佳或夸大功效、虚假宣传」",
        "use": "信任赤字直接反映在消费者端",
        "source": "https://www.iimedia.cn/108723.html",
    },
]


def enrich_claims(text):
    """把 text_tool 的命中结果升级成带法条、处罚区间、真实案例的合规清单。"""
    # 自检：词典里每个 category 都必须有对应法条，否则会出现"命中了但定不了档"的静默失效
    try:
        from text_tool import CLAIM_LEXICON
        missing = [c for c in CLAIM_LEXICON if c not in CATEGORY_TO_LAW]
        if missing:
            print(f"[WARN] text_tool 词典里有 category 未配置法条: {missing}", file=sys.stderr)
    except Exception:
        pass

    claims = scan_claims(text)
    out = []
    for c in claims:
        item = dict(c)
        law_key = CATEGORY_TO_LAW.get(c.get("category")) or CATEGORY_TO_LAW.get(c.get("category_label"))
        law = LAW_CATALOG.get(law_key)
        if law:
            item["law_key"] = law_key
            item["法条"] = law["法名"]
            item["法条要点"] = law["要点"]
            item["处罚"] = law["处罚"]
            item["law_ref"] = law["law_ref"]
            if law.get("case"):
                item["真实案例"] = law["case"]
                item["case_ref"] = law.get("case_ref")
            if law.get("penalty_note"):
                item["处罚说明"] = law["penalty_note"]
            if law.get("note"):
                item["适用边界"] = law["note"]
        out.append(item)
    return out


def special_category_findings(text):
    """特殊化妆品（《条例》第十六条）提示。

    刻意**不挂在每条命中上**——第一版把它塞进每条 claim 的「附加提示」，
    结果同一句「美白属特殊化妆品」在 4 条命中上重复刷屏，报告没法看。
    现在独立成条，一个词只报一次。
    """
    found = []
    for m in SPECIAL_CATEGORY_PAT.finditer(text):
        found.append({
            "type": "special_category",
            "word": m.group(1),
            "法条": LAW_CATALOG["special_category"]["法名"],
            "要点": LAW_CATALOG["special_category"]["要点"],
            "处罚": LAW_CATALOG["special_category"]["处罚"],
            "真实案例": LAW_CATALOG["special_category"]["case"],
            "law_ref": LAW_CATALOG["special_category"]["law_ref"],
            "case_ref": LAW_CATALOG["special_category"]["case_ref"],
            "适用边界": LAW_CATALOG["special_category"]["note"],
        })
    # 去重（同一功效词只报一次）
    seen, uniq = set(), []
    for f in found:
        if f["word"] in seen:
            continue
        seen.add(f["word"])
        uniq.append(f)
    return uniq


def exposure_report(items, special=None):
    """风险敞口：把命中条目换算成法务关心的「可能承担什么责任」。"""
    special = special or []
    keys = {i.get("law_key") for i in items if i.get("law_key")}
    if special:
        keys.add("special_category")
    tiers = []
    if "false_ad" in keys:
        tiers.append({
            "档位": "虚假宣传（最高档）",
            "依据": LAW_CATALOG["false_ad"]["法名"],
            "处罚": "20~100 万元；情节严重 100~200 万元并可吊销营业执照",
            "额外": "消费者可依《消费者权益保护法》第五十五条主张退一赔三",
            "source": LAW_CATALOG["false_ad"]["law_ref"],
        })
    if "medical_claim" in keys or "efficacy_claim" in keys:
        tiers.append({
            "档位": "化妆品宣称违规",
            "依据": LAW_CATALOG["medical_claim"]["法名"],
            "处罚": "责令改正；拒不改正处违法所得三倍以下罚款；情节严重吊销许可证",
            "额外": "最高可涉刑事责任",
            "source": LAW_CATALOG["medical_claim"]["law_ref"],
        })
    if "special_category" in keys:
        tiers.append({
            "档位": "特殊化妆品无证宣称",
            "依据": LAW_CATALOG["special_category"]["法名"],
            "处罚": "没收违法所得并处违法所得三倍以下罚款；情节严重吊销许可证",
            "额外": "实际案例：罚没合计 1.14 万元",
            "source": LAW_CATALOG["special_category"]["law_ref"],
        })
    if "absolute_claim" in keys or "timed_promise" in keys:
        tiers.append({
            "档位": "绝对化用语 / 效果承诺",
            "依据": LAW_CATALOG["absolute_claim"]["法名"],
            "处罚": "责令停止发布、消除影响，处广告费用三至五倍罚款；情节严重吊销执照",
            "额外": "这是最容易被平台自动拦截、也最容易改的一类",
            "source": LAW_CATALOG["absolute_claim"]["law_ref"],
        })
    return {
        "命中条目数": len(items),
        "特殊化妆品提示": [f["word"] for f in special],
        "风险档位": tiers,
        "最高一档": tiers[0]["档位"] if tiers else None,
        "口径说明": (
            "处罚基数是「广告费用」或「违法所得」，不是销售额 —— 这是品牌方最常算错的地方。"
            "本报告是风险提示，不是法律意见，是否违法与如何处罚由监管机构认定。"
        ),
    }


def build_evidence(text, source_id="inline"):
    items = enrich_claims(text)
    special = special_category_findings(text)
    rep = exposure_report(items, special)
    return {
        "tool": "compliance",
        "source_asset_id": source_id,
        "observed": (
            f"合规体检：命中 {len(items)} 处敏感宣称"
            + (f"，另有 {len(special)} 处特殊化妆品功效需核对注册证" if special else "")
            + f"，对应 {len(rep['风险档位'])} 个法规风险档位"
            + (f"，最高一档为「{rep['最高一档']}」" if rep["最高一档"] else "，未构成明显法规风险")
        ),
        "cannot_prove": (
            "本模块是风险提示，不是法律意见。是否构成违法、如何处罚由监管机构认定，"
            "落地前请由法务复核最新法规条文与本地方案。"
            "本模块也无法联网核验产品备案类别、功效依据是否已在药监局公示、"
            "广告费用与违法所得金额，因此**不能据此计算确切罚款金额**。"
        ),
        "evidence": [{
            "claim_violations": items,
            "claim_violation_count": len(items),
            "special_category_findings": special,
            "exposure": rep,
            "industry_facts": INDUSTRY_FACTS,
        }],
    }


def load_posts():
    """读 data/xhs_posts.json。

    ⚠️ 踩过的坑：该文件顶层是 {"_readme": ..., "posts": [...]}，不是数组。
       直接 `for p in json.loads(...)` 会拿到字符串键然后报
       `TypeError: string indices must be integers`。这里统一收口。
    """
    d = json.loads((REPO / "data" / "xhs_posts.json").read_text(encoding="utf-8"))
    posts = d.get("posts") if isinstance(d, dict) else d
    if isinstance(posts, dict):
        posts = list(posts.values())
    return posts or []


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/compliance_rules.py --text \"文案\" | --post ID | --all")
        return 1

    if "--all" in sys.argv:
        posts = load_posts()
        rows = []
        for p in posts:
            items = enrich_claims(p["text"])
            special = special_category_findings(p["text"])
            rep = exposure_report(items, special)
            rows.append({
                "id": p["id"], "platform": p.get("platform"), "title": p.get("title", "")[:40],
                "claim_count": len(items),
                "special": [f["word"] for f in special],
                "tiers": [t["档位"] for t in rep["风险档位"]],
                "top_tier": rep["最高一档"],
            })
            print(f"{p['id']:8s} 命中 {len(items):2d} 处 | 档位: {', '.join(t['档位'] for t in rep['风险档位']) or '无'}")
        out = REPO / "results" / "compliance_scan.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({
            "schema": "beautyproof/compliance_scan@1",
            "scanned": len(posts),
            "rows": rows,
            "note": "全部为公开可见的真实种草文案，逐条可回溯到原文；仅作合规风险提示",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写: {out}")
        return 0

    text = None
    sid = "inline"
    if "--text" in sys.argv:
        i = sys.argv.index("--text")
        text = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    elif "--post" in sys.argv:
        i = sys.argv.index("--post")
        pid = sys.argv[i + 1]
        posts = load_posts()
        hit = [p for p in posts if p.get("id") == pid]
        if not hit:
            print(f"没找到 {pid}")
            return 1
        text, sid = hit[0]["text"], pid

    if text is None:
        print("用法: python tools/compliance_rules.py --text \"文案\" | --post ID | --all")
        return 1

    rep = build_evidence(text, sid)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    out = REPO / "outputs" / f"compliance_{sid}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存到: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

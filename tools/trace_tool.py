# -*- coding: utf-8 -*-
# 来源：原创实现 —— 跨平台跨内容比对与溯源为 BeautyProof 团队原创
"""
跨平台跨内容比对与溯源 —— 命中官方点名加分项「跨平台跨内容比对与溯源」

用法：
    python tools/trace_tool.py --case tr_01
    python tools/trace_tool.py --all
    python tools/trace_tool.py --text "文案" --ocr-file outputs/ocr_xxx.json

为什么做这个（这是本工具存在的唯一理由）：
    前面所有工具都在看「单张图本身」或「单段文案本身」。
    真正的伪造里最常见、也最难被单图工具抓到的是**内容搬运**：
    把别人小红书上的图download下来，配自己写的文案，声称"原相机实拍""品牌方提供"。
    单看图、单独看文案都没问题，只有**把图上的水印账号和文案声称的来源放在一起比**，
    才会露出矛盾。

三条溯源线：
    ① 水印账号 vs 文案声称来源
       小红书/抖音图上普遍带发布者水印（形如 @某某_小红书）。
       若文案声称"品牌官方提供/官网原图/品牌方素材"，而图上水印是**个人账号**，
       则该图**不可能**来自文案声称的来源 → 需核实授权。
    ② 品牌/产品名一致性
       文案里点名的产品或品牌，是否真的出现在图上（OCR 可抄到的字里）。
    ③ 跨内容矩阵
       同一账号多条内容：水印账号是否一致、文案模板是否高度复用。
       一人多号发高度雷同图文 = 矩阵号，这是官方点名的「矩阵级异常模式识别」。

⚠️ 铁律（本工具最容易越界的地方，必须写进每份输出）：
    发现第三方水印 **不等于** 认定盗图。真实场景里至少有四种正当解释：
      ① 博主自己截图自己的笔记（带自己水印完全正常）
      ② 品牌方转载博主的投稿图（授权链条存在，只是不在图上体现）
      ③ 用了平台官方素材库（带平台水印）
      ④ 二次创作但已获授权（口头/私信授权，无书面凭证）
    所以本工具的结论**只到「需核实授权」**，绝不输出「盗图已证实」。
    跨平台取证要落到平台投诉/司法鉴定，本工具只负责把矛盾点指出来。
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ── 水印识别 ──────────────────────────────────────────────────
# 小红书 / 抖音 / 微博 图上常见的发布者水印形态
WATERMARK_PATTERNS = [
    re.compile(r"@([^\s@，,。:：]{1,20})"),                    # @账号名
    re.compile(r"([一-龥A-Za-z0-9_]{1,20})[的の]?(小红书|抖音|快手|微博|视频号)"),
    re.compile(r"^(?:来自)?(.*?)(?:的小红书|的抖音)$"),
]
# 平台官方的图库水印（这类反而说明是官方素材，不算矛盾）
PLATFORM_OFFICIAL = re.compile(r"(官方|品牌|旗舰|企业号|官方素材|素材库|beian|备案)")

# ── 文案里"来源声称"的识别 ────────────────────────────────────
SOURCE_CLAIM_PATTERNS = {
    "brand_official": re.compile(r"(品牌官方|官方提供|品牌方提供|官方素材|旗舰店|官方图|品牌方素材|官方授权)"),
    "own_shot": re.compile(r"(原相机|原图直出|实拍|无滤镜|未修图|一手实拍|本人拍摄)"),
    "official_site": re.compile(r"(官网|官方商城|天猫|京东|官方渠道)"),
}

# ── 品牌/产品名候选（中文品牌常见形态 + 英文品牌）──────────────
# 品牌名只认拉丁字母品牌（OCR 抄到的包装英文/拼音商标）。
# ⚠️ 第一版还有一条「中文品牌后缀（…牌/…记/…堂）」的正则，实测是误报源：
#    「这张是品牌方提供的」被抽出品牌名「这张是品牌」。中文品牌后缀词太常见、
#    无法用词表穷举，故**整条删掉**，品牌比对降为辅助信息，主判据交给品类词。
BRAND_PAT = re.compile(r"([A-Z][A-Za-z]{2,15})")

# ── 品类词（关键：比品牌名可靠得多）──────────────────────────
# 为什么必须比品类（实测踩出来的）：
#   第一版只比品牌名，结果两个问题：
#     ① 「文案说珂润浸润保湿面霜、图上是美即烟酰胺原液」这种**最典型的图文错配**
#        检测不到 —— 因为「珂润浸润保湿面霜」不带「牌/记/堂」等品牌后缀，
#        品牌正则压根没抽到东西。
#     ② 反过来「图上有 MIST 保湿喷雾、文案没提 MIST」被判成中风险 ——
#        但**产品图上印着品牌名本来就是正常的**，这条是纯误报。
#   改法：主判据换成**品类是否对得上**（文案说面霜、图上是原液 = 硬矛盾），
#   品牌名只作辅助参考，且「图上有品牌」不再单独作为风险。
PRODUCT_CATEGORIES = [
    "粉底液", "气垫", "粉饼", "散粉", "定妆", "遮瑕", "BB霜", "CC霜", "素颜霜",
    "面霜", "精华水", "精华液", "原液", "安瓶", "水乳", "乳液", "面膜", "眼霜",
    "防晒", "口红", "唇釉", "唇膏", "眉笔", "眼影", "腮红", "香水", "身体乳",
    "护手霜", "喷雾", "洁面", "洗面奶", "卸妆", "膏霜", "乳霜",
]


def find_categories(text):
    t = str(text or "")
    return {c for c in PRODUCT_CATEGORIES if c in t}


def parse_ocr(ocr_evidence):
    """从 ocr 证据里抽出文字行。兼容两种形状：list[dict] 或 {"evidence": [...]}。"""
    if isinstance(ocr_evidence, dict):
        lines = ocr_evidence.get("evidence") or []
    else:
        lines = ocr_evidence or []
    out = []
    for ln in lines:
        if isinstance(ln, dict) and ln.get("text"):
            out.append({"text": str(ln["text"]), "bbox": ln.get("bbox"),
                        "confidence": ln.get("confidence")})
    return out


def find_watermarks(ocr_lines):
    """找出图上的发布者水印。返回 [{handle, raw, bbox}]。"""
    hits = []
    for ln in ocr_lines:
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        if PLATFORM_OFFICIAL.search(t):
            continue                      # 官方图库水印不算矛盾，跳过
        for pat in WATERMARK_PATTERNS:
            m = pat.search(t)
            if m:
                handle = (m.group(1) or m.group(2) or "").strip()
                if handle:
                    hits.append({"handle": handle, "raw": t, "bbox": ln.get("bbox"),
                                 "confidence": ln.get("confidence")})
                break
    # 去重（同一个 handle 只留一条）
    seen, uniq = set(), []
    for h in hits:
        if h["handle"] in seen:
            continue
        seen.add(h["handle"])
        uniq.append(h)
    return uniq


def find_source_claims(text):
    out = {}
    for k, pat in SOURCE_CLAIM_PATTERNS.items():
        m = pat.search(text)
        if m:
            out[k] = {"matched": m.group(0), "position": m.start()}
    return out


def find_brands(text):
    out = set()
    for m in BRAND_PAT.finditer(text or ""):
        out.add((m.group(1) or "").strip())
    return {b for b in out if len(b) >= 3}


def compare_text_ocr(text, ocr_lines):
    """线②：品类是否对得上（主判据）+ 品牌名一致性（辅助）。

    ⚠️ 「图上出现文案未提及的品牌」**不再**作为风险：产品图上印着品牌名本来就正常，
       第一版把它当 medium，5 例里错了 2 例，纯误报源。
    """
    ocr_text = " ".join((ln.get("text") or "") for ln in ocr_lines)
    t_brands, o_brands = find_brands(text), find_brands(ocr_text)
    t_cats, o_cats = find_categories(text), find_categories(ocr_text)
    return {
        "文案品类": sorted(t_cats),
        "图上品类": sorted(o_cats),
        "品类冲突": bool(t_cats and o_cats and not (t_cats & o_cats)),
        "文案品牌": sorted(t_brands),
        "图上品牌": sorted(o_brands),
        "两边都有品牌": sorted(t_brands & o_brands),
        "只在文案品牌": sorted(t_brands - o_brands),
        "图上文字摘要": ocr_text[:120],
    }


def trace_one(text, ocr_evidence, ocr_conf_min=0.5):
    """对单条内容（文案 + 配图 OCR）做一次溯源核查。"""
    ocr_lines = [ln for ln in parse_ocr(ocr_evidence)
                 if (ln.get("confidence") or 0) >= ocr_conf_min]
    watermarks = find_watermarks(ocr_lines)
    claims = find_source_claims(text)
    brand_cmp = compare_text_ocr(text, ocr_lines)

    findings = []
    risk = "low"

    # ① 水印账号 vs 来源声称
    if watermarks and (claims.get("brand_official") or claims.get("official_site")):
        handles = "、".join("@" + w["handle"] for w in watermarks)
        findings.append({
            "类型": "水印与来源声称矛盾",
            "严重度": "high",
            "说明": f"文案声称「{claims.get('brand_official', {}).get('matched') or claims.get('official_site', {}).get('matched')}」"
                    f"来自品牌官方/官网，但图上能读到发布者水印 {handles}。"
                    f"该图不可能来自文案声称的来源。",
            "为什么重要": "这是内容搬运的典型特征 —— 单看图、单独看文案都发现不了，"
                          "只有把两者放在一起比对才露出来。",
            "⚠️ 边界": "发现第三方水印不等于认定盗图：博主截图自己的笔记、"
                      "品牌转载投稿图、平台素材库都可能有水印。结论只到「需核实授权」。",
        })
        risk = "high"
    elif watermarks and claims.get("own_shot"):
        findings.append({
            "类型": "水印与「实拍」声称并存（需注意，非必然矛盾）",
            "严重度": "medium",
            "说明": f"文案声称「{claims['own_shot']['matched']}」，图上同时有发布者水印 "
                    f"{'、'.join('@' + w['handle'] for w in watermarks)}。"
                    f"若为博主自己发布则两者一致；若声称是他人素材则需核实授权。",
            "⚠️ 边界": "这条是提示不是矛盾 —— 小红书博主自己发的笔记本来就带自己的水印。",
        })
        if risk == "low":
            risk = "medium"

    # ② 品类冲突：文案说面霜、图上是原液 —— 这是最典型的图文错配/混用他牌素材
    if brand_cmp["品类冲突"]:
        findings.append({
            "类型": "文案品类与图上不一致",
            "严重度": "medium",
            "说明": f"文案在说「{'、'.join(brand_cmp['文案品类'])}」，"
                    f"但图上可读文字指向「{'、'.join(brand_cmp['图上品类'])}」"
                    f"（OCR：{brand_cmp['图上文字摘要'] or '无'}）。",
            "为什么重要": "常见于「挂着 A 的名卖 B 的图」或图文错配；"
                          "比品牌名比对可靠，因为品牌名可能因包装/字体 OCR 认不出，品类词认得出。",
            "⚠️ 边界": "一套水乳套装图上同时出现多种品类词时会看起来冲突，需人工确认主体是哪件。",
        })
        if risk == "low":
            risk = "medium"
    elif brand_cmp["只在文案品牌"] and not brand_cmp["两边都有品牌"]:
        # 品牌名比对降级为「一并提示」，不单独抬风险
        findings.append({
            "类型": "文案点名的品牌未在图上可读文字里找到（仅供参考）",
            "严重度": "info",
            "说明": f"文案提到 {'、'.join(brand_cmp['只在文案品牌'])}，图上没读到。"
                    f"品牌名常因艺术字/遮挡/画面外而 OCR 认不出，所以这条只作提示、不单独定风险。",
        })

    verdict = {"low": "credible", "medium": "suspicious", "high": "high_risk"}[risk]

    return {
        "风险": risk,
        "verdict": verdict,
        "图上水印": [{"账号": "@" + w["handle"], "原文": w["raw"],
                      "置信度": w.get("confidence")} for w in watermarks],
        "文案来源声称": claims,
        "品牌比对": brand_cmp,
        "findings": findings,
        "headline": ({
            "high": "图上水印与文案声称的来源矛盾，需核实素材授权",
            "medium": "文案与图上信息存在对不上的地方，建议人工核对",
            "low": "未发现跨平台比对上的矛盾",
        })[risk],
    }


def trace_matrix(records):
    """线③：跨内容矩阵 —— 同一账号多条内容横向比对。

    records: [{"content_id":..., "text":..., "ocr_evidence":...}, ...]
    看两件事：水印账号是否一致（一人多号）、文案模板是否高度复用。
    """
    n = len(records)
    if n < 2:
        return {"content_count": n, "verdict": "inconclusive",
                "note": "至少要有 2 条内容才能做跨内容比对"}

    handles = []
    for r in records:
        ws = find_watermarks(parse_ocr(r.get("ocr_evidence")))
        handles.append(ws[0]["handle"] if ws else None)
    known = [h for h in handles if h]
    handle_variety = len(set(known))

    texts = [re.sub(r"\s+", "", r.get("text") or "") for r in records]

    def grams(s, k=4):
        s = re.sub(r"[^\w一-龥]", "", s)
        return {s[i:i + k] for i in range(max(0, len(s) - k + 1))} or ({s} if s else set())

    sims = []
    for a in range(n):
        for b in range(a + 1, n):
            A, B = grams(texts[a]), grams(texts[b])
            sims.append(len(A & B) / len(A | B) if (A and B) else 0.0)
    reuse = sum(sims) / len(sims) if sims else 0.0

    flags = []
    if known and handle_variety > 1:
        flags.append(f"同一批内容出现 {handle_variety} 个不同发布者水印"
                     f"（{'、'.join('@'+h for h in sorted(set(known)))}），疑为「一人多号」矩阵")
    if reuse >= 0.5:
        flags.append(f"文案两两相似度均值 {reuse:.0%}，高度模板化，疑为批量投放的同一套文案")
    if len(known) == 0:
        flags.append("这些图都没读到发布者水印，无法判断来源归属")

    return {
        "content_count": n,
        "水印账号": handles,
        "水印账号数": handle_variety,
        "文案复用率": round(reuse, 4),
        "flags": flags,
        "verdict": "suspicious" if flags else "credible",
        "cannot_prove": "矩阵分析只是统计倾向，不能证明账号矩阵是专门造假；"
                        "「一人多号」也可能是团队运营（品牌方多人共管一个品牌号）。",
    }


def build_evidence(text, ocr_evidence, source_id="inline", records=None):
    res = trace_one(text, ocr_evidence)
    mat = trace_matrix(records) if records else None
    if mat and mat.get("verdict") == "suspicious" and res["verdict"] == "credible":
        res["verdict"] = "suspicious"
        res["风险"] = "medium"
        res["headline"] = "单条内容未见矛盾，但同批内容存在矩阵级异常，建议人工复核"
    return {
        "tool": "trace",
        "source_asset_id": source_id,
        "observed": (f"跨平台溯源：读到 {len(res['图上水印'])} 个图上发布者水印，"
                     f"文案来源声称 {len(res['文案来源声称'])} 项，"
                     f"命中 {len(res['findings'])} 处比对矛盾 → {res['headline']}"),
        "cannot_prove": (
            "发现第三方水印不等于认定盗图：博主截图自己的笔记、品牌转载投稿图、"
            "平台素材库、二次创作已获授权，四种情况都会产生水印。"
            "本工具只负责把矛盾点指出来并提示「需核实授权」，"
            "绝不输出「盗图已证实」；跨平台取证要落到平台投诉或司法鉴定。"
        ),
        "evidence": [{**res, "matrix": mat}],
    }


def load_cases():
    d = json.loads((REPO / "data" / "trace_cases.json").read_text(encoding="utf-8"))
    cases = d.get("cases") if isinstance(d, dict) else d
    if isinstance(cases, dict):
        cases = list(cases.values())
    return d if isinstance(d, dict) else {}, cases or []


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/trace_tool.py --case ID | --all | --text T --ocr-file F")
        return 1

    if "--all" in sys.argv:
        meta, cases = load_cases()
        ok, rows = 0, []
        print(f"{'case':26s} {'期望':12s} {'实得':12s} 判定")
        print("-" * 66)
        for c in cases:
            r = trace_one(c["text"], c["ocr_lines"])
            exp = c["expected_verdict"]
            hit = exp == r["verdict"]
            ok += 1 if hit else 0
            rows.append({"id": c["id"], "expected": exp, "got": r["verdict"], "hit": hit,
                         "findings": [f["类型"] for f in r["findings"]],
                         "watermarks": [w["账号"] for w in r["图上水印"]]})
            print(f"{c['id']:26s} {exp:12s} {r['verdict']:12s} {'✅' if hit else '❌'}")
        n = len(cases)
        print("-" * 66)
        print(f"方向命中 {ok}/{n} = {ok/n:.1%}")
        p = REPO / "results" / "trace_eval.json"
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({
            "schema": "beautyproof/trace_eval@1",
            "data_provenance": meta.get("_provenance"),
            "disclaimer": meta.get("_disclaimer"),
            "n_cases": n, "direction_hit": ok, "direction_rate": round(ok / n, 4) if n else None,
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
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
        rep = build_evidence(c["text"], c["ocr_lines"], c["id"], c.get("records"))
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        out = REPO / "outputs" / f"trace_{c['id']}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存到: {out}")
        return 0

    if "--text" in sys.argv and "--ocr-file" in sys.argv:
        text = sys.argv[sys.argv.index("--text") + 1]
        of = Path(sys.argv[sys.argv.index("--ocr-file") + 1])
        ocr = json.loads(of.read_text(encoding="utf-8"))
        rep = build_evidence(text, ocr, of.stem)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print("用法: python tools/trace_tool.py --case ID | --all | --text T --ocr-file F")
    return 1


if __name__ == "__main__":
    sys.exit(main())

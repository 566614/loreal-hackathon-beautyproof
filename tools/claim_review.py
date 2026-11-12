"""Conservative extraction and comparison of explicit beauty claims."""

import json
import os
import re
import unicodedata
import uuid
from pathlib import Path


CLAUSES = re.compile(r"[^。！？!?；;\n]+")
QUOTES = re.compile(r'「[^」]*」|“[^”]*”|"[^"]*"')
NEGATED = re.compile(r"(?:并非|并不|不是|不能|不可|不得|不具备|不具有|不保证|无法|没有|未见|不会|无|不|非)$")
QUOTING = re.compile(r"(?:声称|宣称|写着|提到|引用|说法|评论|引述|禁止|禁用|辟谣|例如|比如)")
DISCLOSURE = re.compile(r"(?:AI|人工智能|生成式)(?:生成|合成|绘制)?\s*(?:示意(?:图)?|效果图|模拟(?:图)?|演示(?:图)?)|(?:示意图|效果图)\s*[（(]?\s*(?:AI|人工智能)(?:生成)?", re.I)
EFFICACY_PHOTO = re.compile(
    r"(?:真人|本人|实拍|实测|真实拍摄)[^，,。！？!?；;\n]{0,10}(?:前后对比|效果对比|功效对比|使用前后)"
    r"|(?:前后对比|使用前后)[^，,。！？!?；;\n]{0,10}(?:实拍|真人|本人)"
)
PRODUCT = re.compile(
    r"(?:产品(?:名称)?\s*[:：]\s*([^，,。；;\n]+)|【([^】]*(?:精华(?:液)?|面霜|眼霜|乳液|"
    r"粉底(?:液)?|口红|唇釉|防晒霜|面膜|洁面乳|化妆水))】)"
)
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
DURATION = re.compile(NUMBER + r"\s*(天|日|周)")
PRODUCT_VERSION = re.compile(r"(?<!\d)(?:20\d{2}版|第[一二三四五六七八九十\d]+代|新版|旧版)")
SHADE = re.compile(r"(?:色号|色號|#)\s*[:：#]?\s*([A-Za-z0-9]+)", re.I)
POPULATION = re.compile(r"敏感肌|混油皮|混干皮|干皮|油皮|受试者|成年人")
CONDITIONS = re.compile(r"即时|即刻|首次|连续使用|早晚|夜间|每[日天]|实验室|消费者自评|最高|平均|至少|至多")
METRICS = (
    ("水分提升", re.compile(r"(?:角质层)?(?:水分|含水量|含水率)\s*(?:提升|增加|提高)\s*" + NUMBER + r"\s*(%)")),
    ("天然来源比例", re.compile(NUMBER + r"\s*(%)\s*(?:为|的)?\s*天然来源")),
    ("天然来源比例", re.compile(r"天然来源(?:成分)?(?:比例|占比)?\s*(?:达|为|[:：])?\s*" + NUMBER + r"\s*(%)")),
    ("斑点减少", re.compile(r"(?:色斑|斑点)(?:面积|数量)?\s*(?:减少|下降|降低)\s*" + NUMBER + r"\s*(%)")),
    ("保湿时长", re.compile(r"保湿\s*(?:长达|持续)\s*" + NUMBER + r"\s*(小时|h|天|日)", re.I)),
    ("保湿时长", re.compile(NUMBER + r"\s*(小时|h)\s*(?:持久)?保湿", re.I)),
    ("规格", re.compile(r"(正装|赠品|小样|试用装)\s*(?:容量|净含量|规格)?\s*[:：]?\s*" + NUMBER + r"\s*(毫升|ml|克|g)", re.I)),
)
UNQUANTIFIED = re.compile(r"治愈|治疗|根治|修复|美白|祛斑|抗皱|淡纹|保湿|控油|祛痘")
UNITS = {"毫升": "ml", "克": "g", "小时": "h", "日": "天"}
REFERENCE_TYPES = {"efficacy_study", "product_label", "brand_claim", "method"}
REFERENCE_METADATA = ("source_type", "market", "product_version", "published", "method", "sample", "population", "period", "conditions")


def explicit_product(text, product):
    """Only bind a free-text name at a visible boundary, never a longer SKU."""
    return bool(product and re.search(r"(?<![\w])" + re.escape(product) + r"(?![A-Za-z\u4e00-\u9fff])", text))


def stance_at(text, start):
    """Classify explicit local negation or reported speech at a match."""
    before = re.split(r"[。！？!?；;，,\n]", text[:start])[-1].rstrip()
    action = re.sub(r"(?:宣称|提供|起到|用于|治疗|治愈|根治|预防|缓解)$", "", before).rstrip()
    if NEGATED.search(before) or NEGATED.search(action):
        return "negated"
    rest = text[start:]
    end = re.search(r"[。！？!?；;\n]", rest)
    tail = rest[:end.start()] if end else rest
    if re.search(r"(?:说法|说辞|观点|宣称)(?:并不属实|不属实|不成立|不实|错误|虚假)", tail):
        return "negated"
    for quote in QUOTES.finditer(text):
        if quote.start() < start < quote.end():
            context = text[max(0, quote.start() - 16):quote.start()] + text[quote.end():quote.end() + 12]
            if QUOTING.search(context) or re.search(r"不实|不成立|错误|不可信", context):
                return "quoted"
    if end and end.group() in "？?":
        return "question"
    return "asserted"


def is_illustration_disclosed(*sources):
    return any(stance_at(text, match.start()) == "asserted"
               for text in sources for match in DISCLOSURE.finditer(text))


def extract(text, product=None, source="body"):
    """Return original excerpts and only fields that can be attributed to them."""
    facts = []
    if not product:
        declared = {next(g for g in m.groups() if g).strip() for m in PRODUCT.finditer(text)}
        product = declared.pop() if len(declared) == 1 else None
    for clause in CLAUSES.finditer(text):
        raw = clause.group()
        line = unicodedata.normalize("NFKC", raw)
        marker = PRODUCT.search(raw)
        subject = (next(g for g in marker.groups() if g).strip() if marker
                   else product if explicit_product(raw, product) else None)
        durations = {float(m[1]) * (7 if m[2] == "周" else 1) for m in DURATION.finditer(line)}
        versions = set(PRODUCT_VERSION.findall(line))
        product_version = next(iter(versions)) if len(versions) == 1 else "ambiguous" if versions else None
        days = next(iter(durations)) if len(durations) == 1 else None
        population = sorted(set(POPULATION.findall(line)))
        common = {"source": source, "quote": raw.strip(), "product": subject,
                  "product_version": product_version,
                  "duration_days": days, "population": population,
                  "conditions": sorted(set(CONDITIONS.findall(line))),
                  "span": [clause.start(), clause.end()]}
        quantified = []
        for metric, pattern in METRICS:
            for match in pattern.finditer(line):
                *qualifiers, value, unit = match.groups()
                label = qualifiers[0] if qualifiers else None
                unit = UNITS.get(unit.lower(), unit.lower())
                fact = dict(common, metric=metric, variant=label, value=float(value), unit=unit,
                            stance=stance_at(text, clause.start() + match.start()))
                if metric == "保湿时长" and unit == "天":
                    fact.update(value=fact["value"] * 24, unit="h")
                if metric in {"保湿时长", "规格", "天然来源比例"}:
                    fact["duration_days"] = None
                quantified.append(fact)
        facts.extend(quantified)
        for shade in SHADE.finditer(line):
            facts.append(dict(common, metric="色号", variant=None, value=shade.group(1).upper(),
                              unit="色号", stance=stance_at(text, clause.start() + shade.start())))
        if not quantified:
            for match in UNQUANTIFIED.finditer(line):
                facts.append(dict(common, metric=match.group(), variant=None, value=None, unit=None,
                                  stance=stance_at(text, clause.start() + match.start())))
        photo = EFFICACY_PHOTO.search(line) or re.search(r"原相机实拍|原图直出|未修图|无滤镜|实拍", line)
        if photo:
            facts.append(dict(common, metric="实拍或未经编辑宣称", variant=None, value=None, unit=None,
                              stance=stance_at(text, clause.start() + photo.start())))
    return facts


def enrich(facts, text, rows, source):
    """Use model attribution only when both the excerpt and subject occur together."""
    for fact in facts:
        if fact["product"]:
            continue
        matches = {
            row["product"].strip() for row in rows
            if isinstance(row.get("product"), str) and row["product"].strip()
            and explicit_product(fact["quote"], row["product"].strip())
            and fact["quote"] in row["quote"] and row["quote"] in text
        }
        if len(matches) == 1:
            fact["product"] = matches.pop()
    for row in rows:
        quote, metric = row["quote"], row.get("metric_text")
        if (row.get("stance") != "asserted" or not isinstance(metric, str)
                or not metric.strip() or metric not in quote
                or any(f["stance"] == "asserted" and f["metric"] in quote
                       and (f["quote"] in quote or quote in f["quote"]) for f in facts)):
            continue
        start = text.index(quote)
        if stance_at(text, start + quote.index(metric)) != "asserted":
            continue
        product = row.get("product")
        subject = product if isinstance(product, str) and explicit_product(quote, product.strip()) else None
        # Unsupported semantic metrics can request evidence. They do not
        # participate in automatic numeric comparisons or verdict promotion.
        facts.append({"source": source, "quote": quote, "product": subject,
                      "product_version": None,
                      "duration_days": None, "population": [], "conditions": [],
                      "span": [start, start + len(quote)], "metric": metric,
                      "variant": None, "value": None, "unit": None, "stance": "asserted",
                      "extraction": "model_excerpt"})
    return facts


def comparable(left, right):
    fields = ("product", "metric", "variant", "unit", "duration_days", "population", "conditions")
    if left["metric"] in {"水分提升", "斑点减少"} and left["duration_days"] is None:
        if not {"即时", "即刻"} & set(left["conditions"]):
            return False
    return (left.get("product_version") != "ambiguous" and right.get("product_version") != "ambiguous"
            and left.get("product_version") == right.get("product_version")
            and left["stance"] == right["stance"] == "asserted"
            and bool(left["product"]) and left["value"] is not None and right["value"] is not None
            and all(left[k] == right[k] for k in fields))


def context_conflicts(claims):
    """Find conflicting attributed values within one document."""
    conflicts = []
    for index, left in enumerate(claims):
        for right in claims[index + 1:]:
            if left["span"] == right["span"] or not comparable(left, right):
                continue
            if left["value"] != right["value"]:
                conflicts.append({"type": "正文同一宣称数值不一致", "first": left,
                                  "second": right, "reason": "同一产品和条件下的数值不同，请核对原文。"})
    return conflicts


def verified_catalog():
    """The reviewer maintains this catalog outside the web upload directory."""
    path = Path(os.environ.get("BP_VERIFIED_REFERENCES") or
                Path(__file__).resolve().parents[1] / ".local" / "knowledge" / "verified.json")
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("已核验依据清单须为数组")
    catalog = {}
    fields = ("id", "source", "excerpt", "product", "version", "locator", "reviewer")
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
            raise ValueError("依据清单须含 ID、来源、摘录、产品、版本、位置和审核人")
        ref = {k: row[k].strip() for k in fields}
        for key in REFERENCE_METADATA:
            if key in row:
                if not isinstance(row[key], str):
                    raise ValueError(f"资料字段 {key} 必须是文本")
                ref[key] = row[key].strip()
        if ref.get("source_type") and ref["source_type"] not in REFERENCE_TYPES:
            raise ValueError("资料类型无效")
        if ref["id"] in catalog:
            raise ValueError("已核验依据 ID 重复")
        catalog[ref["id"]] = ref
    return catalog


def add_verified_reference(data):
    """Save an explicitly reviewed excerpt and its provenance locally."""
    if not isinstance(data, dict) or data.get("reviewer_confirmed") is not True:
        raise ValueError("请确认已核对原始资料和适用范围")
    fields = ("source", "excerpt", "product", "version", "locator", "reviewer")
    if any(not isinstance(data.get(k), str) or not data[k].strip() for k in fields):
        raise ValueError("请完整填写资料来源、摘录、产品、版本、位置和审核人")
    ref = {k: data[k].strip() for k in fields}
    for key in REFERENCE_METADATA:
        if key in data:
            if not isinstance(data[key], str):
                raise ValueError(f"资料字段 {key} 必须是文本")
            ref[key] = data[key].strip()
    if ref.get("source_type") not in REFERENCE_TYPES:
        raise ValueError("请选择有效的资料类型")
    if any(len(value) > 10000 for value in ref.values()):
        raise ValueError("资料字段过长")
    ref["id"] = uuid.uuid4().hex
    path = Path(os.environ.get("BP_VERIFIED_REFERENCES") or
                Path(__file__).resolve().parents[1] / ".local" / "knowledge" / "verified.json")
    rows = list(verified_catalog().values())
    rows.append(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return ref


def assess(body, image, references, contradictions, *, verified_references=(), market=None):
    supplied = []
    for ref, verification in ([(r, "user_supplied") for r in references] +
                              [(r, "reviewer_verified") for r in verified_references]):
        for fact in extract(ref["excerpt"], ref.get("product"), "reference"):
            if verification == "reviewer_verified" and fact["product"] != ref["product"]:
                continue
            if verification == "reviewer_verified" and ref.get("product_version"):
                if fact.get("product_version") and fact["product_version"] != ref["product_version"]:
                    continue
                fact["product_version"] = ref["product_version"]
            supplied.append((ref, verification, fact))

    results = []
    for claim in body + image:
        if claim["stance"] != "asserted":
            continue
        side = "body_claim" if claim["source"] == "body" else "image_claim"
        conflicts = [c for c in contradictions if c.get(side) == claim or
                     (side == "body_claim" and claim["metric"] == "实拍或未经编辑宣称"
                      and "body_claim" not in c and
                      claim["span"][0] <= c.get("body_position", -1) < claim["span"][1])]
        matched = [{**{k: ref[k] for k in ("id", "source", "version", "locator", "reviewer", *REFERENCE_METADATA) if k in ref},
                    "quote": fact["quote"], "value": fact["value"], "unit": fact["unit"],
                    "verification": verification, "matches": fact["value"] == claim["value"]}
                   for ref, verification, fact in supplied if comparable(claim, fact)]
        verified = [m for m in matched if m["verification"] == "reviewer_verified"
                    and m.get("source_type") == "efficacy_study"
                    and bool(market) and m.get("market") == market
                    and bool(m.get("product_version")) and m["product_version"] == claim.get("product_version")
                    and m["product_version"] not in ("新版", "旧版", "ambiguous")
                    and all(not m.get(k) or m[k] in claim["quote"]
                            for k in ("method", "sample", "population", "period", "conditions"))]
        if conflicts or any(not m["matches"] for m in verified):
            status, missing = "存在冲突", []
        elif verified:
            status, missing = "有证据支持", []
        else:
            status = "证据不足"
            missing = (["产品或规格"] if not claim["product"] else []) + ["对应产品、指标及条件的原始功效依据"]
        results.append(dict(claim, status=status, missing=missing, references=matched, conflicts=conflicts))
    return results


def normalize_references(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("references 应是数组")
    for item in value:
        if (not isinstance(item, dict) or
                any(not isinstance(item.get(key), str) or not item[key].strip() for key in ("source", "excerpt")) or
                (item.get("product") is not None and not isinstance(item["product"], str))):
            raise ValueError("每份依据须含 source、excerpt，product 可选")
    return [{k: row[k] for k in ("source", "excerpt", "product") if k in row} for row in value]


def ocr_regions(lines, span):
    """Map offsets in newline-joined OCR to boxes in original pixels."""
    start = 0
    regions = []
    for line in lines:
        end = start + len(line["text"])
        if start < span[1] and end > span[0] and line.get("bbox"):
            regions.append({k: line[k] for k in ("text", "bbox", "confidence") if k in line})
        start = end + 1
    return regions


def ocr_lines(evidence):
    """Keep low-confidence OCR in raw evidence but out of claim comparisons."""
    return [row for row in evidence.get("ocr", {}).get("evidence", [])
            if isinstance(row, dict) and isinstance(row.get("text"), str)
            and row.get("confidence", 1) >= .70]

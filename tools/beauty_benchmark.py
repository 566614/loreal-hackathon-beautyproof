"""Validate authorized beauty samples and measure the same pipeline used by Flask."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(stream)


def _hash(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def local_file(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"缺少数据文件或文件不在数据目录内：{value}")
    return path


def validate(manifest):
    from PIL import Image, ImageOps

    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("清单须为非空 JSON 数组")
    root = manifest.parent
    ids, ownership, hashes, fingerprints, originals = set(), {}, {}, {}, set()
    edits = []
    for row in rows:
        name = row.get("id")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "group", "authorization")):
            raise ValueError("每项须有 id、group 和 authorization")
        if name in ids:
            raise ValueError(f"重复样本 ID：{name}")
        ids.add(name)
        split, kind = row.get("split"), row.get("kind")
        if split not in ("smoke", "train", "calibration", "blind") or kind not in ("real", "generated", "edited", "claim"):
            raise ValueError(f"{name} 的 split 或 kind 无效")
        reviewers = row.get("reviewers", [])
        if split == "blind" and (not isinstance(reviewers, list) or
                                  len({r.strip() for r in reviewers if isinstance(r, str) and r.strip()}) < 2):
            raise ValueError(f"{name} 的盲测标签需要两名独立标注者")
        if kind == "generated" and not (row.get("generator") and row.get("generator_version")):
            raise ValueError(f"{name} 缺少生成器家族或版本")
        if kind == "edited" and not (row.get("original_sha256") and row.get("edit")):
            raise ValueError(f"{name} 缺少原图哈希或编辑方式")
        if kind == "claim" and (not isinstance(row.get("claims"), list) or
                                type(row.get("crossmodal_conflict")) is not bool):
            raise ValueError(f"{name} 缺少宣称字段或图文冲突金标")
        for key in ("group", "identity", "campaign", "study", "original_sha256", "generator"):
            if not row.get(key):
                continue
            if not isinstance(row[key], str):
                raise ValueError(f"{name} 的 {key} 须为字符串")
            # Generator families may contain many groups, but blind families must be unseen.
            partition = (("blind" if split == "blind" else "development") if key == "generator"
                         else (split, row["group"]))
            owner = (key, row[key])
            if ownership.setdefault(owner, partition) != partition:
                raise ValueError(f"{name}：{key} 跨分区或来源组：{row[key]}")
        if kind == "edited":
            edits.append((name, split, row["group"], row["original_sha256"]))
        if not row.get("image"):
            if kind != "claim" or not row.get("text"):
                raise ValueError(f"{name} 缺少输入图片或正文")
            continue
        image = local_file(root, row["image"])
        digest = sha256(image)
        if digest != row.get("sha256"):
            raise ValueError(f"{name} 图片 SHA-256 不匹配")
        location = (split, row["group"])
        if hashes.setdefault(digest, location) != location:
            raise ValueError(f"{name} 图片跨分区或来源组重复")
        if ownership.setdefault(("original_sha256", digest), location) != location:
            raise ValueError(f"{name} 的 original_sha256 原图和衍生图跨分区或来源组")
        if kind in ("real", "generated"):
            originals.add((split, row["group"], digest))
        with Image.open(image) as raw:
            gray = list(ImageOps.grayscale(raw).resize((9, 8)).getdata())
            fingerprint = sum((gray[i] > gray[i + 1]) << (8 * (i // 9) + i % 9)
                              for i in range(72) if i % 9 != 8)
            for other, (other_id, other_split) in fingerprints.items():
                if split != other_split and (fingerprint ^ other).bit_count() <= 4:
                    raise ValueError(f"{name} 和 {other_id} 跨分区近重复，请复核")
            fingerprints.setdefault(fingerprint, (name, split))
            size = raw.size
        if row.get("mask"):
            if kind != "edited":
                raise ValueError(f"{name} 只有编辑图可标正掩码")
            mask = local_file(root, row["mask"])
            if sha256(mask) != row.get("mask_sha256"):
                raise ValueError(f"{name} 掩码 SHA-256 不匹配")
            with Image.open(mask) as raw:
                if raw.size != size or raw.convert("L").getextrema() != (0, 255):
                    raise ValueError(f"{name} 掩码须与原图同尺寸，且含前景和背景")
    for name, split, group, original_hash in edits:
        if (split, group, original_hash) not in originals:
            raise ValueError(f"{name} 的 original_sha256 找不到同组原图记录")
    return rows


def run(manifest, rows, llm):
    unchecked = [row["id"] for row in rows if row["split"] != "smoke" and row["kind"] in ("real", "generated", "edited")
                 and (row.get("visible_ai_label_reviewed") is not True or
                      row.get("visible_ai_label_present") is not False)]
    if unchecked:
        raise ValueError("评测图片须复核并去除可见 AI 标识：" + ", ".join(unchecked[:5]))
    import claim_review
    import pipeline

    catalog = claim_review.verified_catalog()
    reports = []
    for row in rows:
        image = local_file(manifest.parent, row["image"]) if row.get("image") else None
        result = pipeline.analyze(image, text=row.get("text"), product=row.get("product"),
                                  market=row.get("market"),
                                  references=row.get("references"), llm=llm,
                                  verified_references=[catalog[i] for i in row.get("reference_ids", [])],
                                  timeout=120, verbose=False, publish=False)
        audit = result["audit"]
        reports.append({"id": row["id"], "image_sha256": (result.get("image") or {}).get("sha256"),
                        "run_id": result["run_id"], "analysis": result["_output_file"],
                        "seconds": result["elapsed_seconds"], "checks": audit["checks"],
                        "signals": audit["image_signals"], "claims": audit["claims"],
                        "context_conflicts": audit.get("context_conflicts", []),
                        "semantic_findings": audit.get("semantic_findings", []),
                        "crossmodal": result["evidence"].get("crossmodal", {}).get("evidence", [{}])[0]})
    return reports


def rate(events):
    values = list(events)
    n, count = len(values), sum(values)
    if not n:
        return {"n": 0, "count": 0, "rate": None, "ci95": None}
    p, z = count / n, 1.959964
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return {"n": n, "count": count, "rate": p, "ci95": [max(0, center - width), min(1, center + width)]}


def groups(rows, predicate):
    result = defaultdict(list)
    for row, prediction in rows:
        result[row["group"]].append(predicate(row, prediction))
    return result.values()


def origin_report(rows, low, high):
    real = [(row, score) for row, score in rows if row["kind"] == "real"]
    fake = [(row, score) for row, score in rows if row["kind"] == "generated"]
    complete_real = [g for g in groups(real, lambda row, score: score) if all(score is not None for score in g)]
    complete_fake = [g for g in groups(fake, lambda row, score: score) if all(score is not None for score in g)]
    uncertain = lambda row, score: score is None or low <= score < high
    return {"false_positive": rate(any(score >= high for score in g) for g in complete_real),
            "recall": rate(all(score >= high for score in g) for g in complete_fake),
            "uncertain": rate(any(g) for g in groups(real + fake, uncertain)),
            "missing_scores": {"real": sum(score is None for _, score in real),
                               "generated": sum(score is None for _, score in fake)}}


def calibrate(rows, manifest_hash):
    origin = [(row, report["signals"]["origin"]) for row, report in rows if row["kind"] in ("real", "generated")]
    identities = {(s.get("model"), s.get("weights_sha256"), s.get("preprocessing")) for _, s in origin}
    if len(identities) != 1 or any(s.get("score") is None for _, s in origin):
        raise ValueError("校准要求固定同一个模型且所有校准图有分数")
    model, weights, preprocessing = identities.pop()
    if not (model and weights and preprocessing):
        raise ValueError("校准要求可追溯的权重哈希和预处理")
    scores = [(row, s["score"]) for row, s in origin]
    if {r["kind"] for r, _ in scores} != {"real", "generated"}:
        raise ValueError("校准同时需要实拍与整图生成")
    candidates = sorted({0.0, 1.0, *[s for _, s in scores]})
    chosen = None
    for high in candidates:
        metrics = origin_report(scores, high, high)
        if metrics["false_positive"]["rate"] <= .05 and metrics["recall"]["rate"] >= .80:
            key = (metrics["recall"]["rate"], -metrics["false_positive"]["rate"], high)
            if chosen is None or key > chosen[0]:
                chosen = key, high
    if chosen is None:
        raise ValueError("没有阈值同时满足校准目标")
    high = chosen[1]
    low = next(low for low in candidates if low <= high and
               origin_report(scores, low, high)["uncertain"]["rate"] <= .25)
    return {"model": model, "weights_sha256": weights, "preprocessing": preprocessing, "low": low,
            "high": high, "calibration_manifest_sha256": manifest_hash, "approved": False,
            "metrics": origin_report(scores, low, high)}


def evaluate(rows, profile, root):
    from PIL import Image
    import numpy as np

    visual = [(row, report["signals"]["origin"]) for row, report in rows if row["kind"] in ("real", "generated")]
    if profile and any(s.get("score") is not None and any(s.get(k) != profile[k] for k in
                       ("model", "weights_sha256", "preprocessing")) for _, s in visual):
        raise ValueError("模型版本、权重哈希或预处理与冻结阈值不一致")
    scores = [(row, s.get("score")) for row, s in visual]
    slices, families = defaultdict(list), defaultdict(list)
    for row, score in scores:
        for tag in row.get("tags", []):
            slices[tag].append((row, score))
        if row["kind"] == "generated":
            families[row["generator"]].append((row, score))
    def summarize(items):
        return origin_report(items, profile["low"], profile["high"])
    ious, clean, field_hits = defaultdict(list), [], [0, 0, 0]
    missing_edit_scores, missing_masks, untracked_edit_scores = 0, 0, 0
    edit_models = set()
    conflict_pairs, context_pairs = [], []
    context_model_unavailable = 0
    fields = ("source", "quote", "product", "product_version", "metric", "value", "unit", "duration_days", "variant",
              "population", "conditions")
    unsupported = 0
    for row, report in rows:
        edit = report["signals"]["editing"]
        # The mask threshold and image operating point are fixed before blind evaluation.
        score = edit.get("score")
        detected = score is not None and score >= .5
        if row["kind"] == "real" or row.get("mask"):
            if score is None:
                missing_edit_scores += 1
            else:
                identity = tuple(edit.get(k) for k in ("model", "weights_sha256", "preprocessing"))
                if not all(identity):
                    untracked_edit_scores += 1
                else:
                    edit_models.add(identity)
        if row["kind"] == "real":
            if score is not None:
                clean.append((row, detected))
        if row.get("mask"):
            predicted = edit.get("mask")
            iou = 0
            if detected and (not predicted or not Path(predicted).is_file()):
                missing_masks += 1
            if detected and predicted and Path(predicted).is_file():
                with Image.open(local_file(root, row["mask"])) as truth, Image.open(predicted) as output:
                    a, b = np.asarray(truth.convert("L")) > 127, np.asarray(output.convert("L")) > 127
                    if a.shape != b.shape:
                        raise ValueError(f"{row['id']} 的预测掩码尺寸与真值不符")
                    iou = float((a & b).sum() / (a | b).sum())
            ious[row["group"]].append(iou)
        if row["kind"] != "claim":
            continue
        if "context_conflict" in row:
            if type(row["context_conflict"]) is not bool:
                raise ValueError(f"{row['id']} 的 context_conflict 须为布尔值")
            if "context_conflicts" not in report or "semantic_findings" not in report:
                raise ValueError(f"{row['id']} 的预测缺少正文上下文结果，请重新运行")
            if report["checks"].get("semantic_review", {}).get("status") != "completed":
                context_model_unavailable += 1
            context_pairs.append((row["context_conflict"], bool(report.get("context_conflicts") or
                                                                  report.get("semantic_findings"))))
        for claim in report["claims"]:
            if claim["status"] == "有证据支持" and not any(
                    r.get("verification") == "reviewer_verified" and r.get("source_type") == "efficacy_study"
                    and r.get("source") and r.get("locator") and r.get("matches") is True
                    and row.get("market") and r.get("market") == row["market"]
                    and r.get("product_version") and r["product_version"] == claim.get("product_version")
                    for r in claim.get("references", [])):
                unsupported += 1
        expected = {(k, json.dumps(c[k], sort_keys=True), c["quote"]) for c in row["claims"]
                    for k in fields if c.get(k) not in (None, [])}
        predicted = {(k, json.dumps(c[k], sort_keys=True), c["quote"]) for c in report["claims"]
                     for k in fields if c.get(k) not in (None, [])}
        for index, count in enumerate((len(expected & predicted), len(expected), len(predicted))):
            field_hits[index] += count
        conflict_pairs.append((row["crossmodal_conflict"], bool(report["crossmodal"].get("contradictions"))))
    tp, labels, guesses = field_hits
    hit = sum(a and b for a, b in conflict_pairs)
    precision = sum(b for _, b in conflict_pairs)
    recall = sum(a for a, _ in conflict_pairs)
    means = [sum(v) / len(v) for v in ious.values()]
    iou = sum(means) / len(means) if means else None
    rng = random.Random(20261110)
    bootstrap = sorted(sum(rng.choices(means, k=len(means))) / len(means)
                       for _ in range(1000)) if means else []
    return {
        "origin": summarize(scores) if profile else None,
        "generator_families": {k: summarize(v)["recall"] for k, v in sorted(families.items())} if profile else {},
        "slices": {k: summarize(v) for k, v in sorted(slices.items())} if profile else {},
        "localization": {"mask_groups": len(ious), "mean_iou": iou,
                         "iou_ci95": [bootstrap[25], bootstrap[974]] if bootstrap else None,
                         "missing_scores": missing_edit_scores, "missing_masks": missing_masks,
                         "untracked_scores": untracked_edit_scores,
                         "models": [{"model": m, "weights_sha256": w, "preprocessing": p}
                                    for m, w, p in sorted(edit_models)],
                         "clean_false_positive": rate(any(g) for g in groups(clean, lambda _, flag: flag))},
        "claims": {"field_f1": 2 * tp / (labels + guesses) if labels + guesses else None,
                   "labels": labels, "predicted": guesses, "matched": tp, "unsupported": unsupported},
        "crossmodal": {"precision": rate(a for a, b in conflict_pairs if b),
                       "recall": rate(b for a, b in conflict_pairs if a), "cases": len(conflict_pairs)},
        "context": {"precision": rate(a for a, b in context_pairs if b),
                    "recall": rate(b for a, b in context_pairs if a), "cases": len(context_pairs),
                    "semantic_unavailable": context_model_unavailable},
        "unavailable": {tool: sum(r["checks"].get(tool, {}).get("status") == "model_unavailable" for _, r in rows)
                        for tool in ("aigc", "trufor", "ocr", "claim_model")},
        "seconds": [r["seconds"] for _, r in rows],
    }


def release_checks(metrics):
    origin = metrics["origin"]
    edit = metrics["localization"]
    claims = metrics["claims"]
    cross = metrics["crossmodal"]
    context = metrics["context"]
    families = metrics["generator_families"].values()
    return {
        "origin": bool(origin and origin["false_positive"]["n"] >= 100 and origin["recall"]["n"] >= 100
                       and len(families) >= 2 and all(g["n"] >= 50 and g["rate"] >= .70 for g in families)
                       and origin["false_positive"]["rate"] <= .05 and origin["recall"]["rate"] >= .80
                       and origin["uncertain"]["rate"] <= .25 and not any(origin["missing_scores"].values())
                       and metrics["unavailable"]["aigc"] == 0),
        "editing": bool(edit["mask_groups"] >= 60 and edit["clean_false_positive"]["n"] >= 100
                        and edit["mean_iou"] >= .40 and edit["clean_false_positive"]["rate"] <= .10
                        and edit["missing_scores"] == 0 and edit["missing_masks"] == 0
                        and edit["untracked_scores"] == 0 and len(edit["models"]) == 1
                        and metrics["unavailable"]["trufor"] == 0),
        "claims": bool(cross["cases"] >= 120 and (claims["field_f1"] or 0) >= .85 and claims["unsupported"] == 0),
        "crossmodal": bool(cross["cases"] >= 120 and (cross["precision"]["rate"] or 0) >= .90
                           and (cross["recall"]["rate"] or 0) >= .80),
        "context": bool(context["cases"] >= 60 and (context["precision"]["rate"] or 0) >= .90
                        and (context["recall"]["rate"] or 0) >= .80 and context["semantic_unavailable"] == 0),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("validate", "run", "calibrate", "evaluate"))
    cli.add_argument("manifest", type=Path)
    cli.add_argument("--split", choices=("smoke", "train", "calibration", "blind"), default="smoke")
    cli.add_argument("--predictions", type=Path)
    cli.add_argument("--profile", type=Path)
    cli.add_argument("--out", type=Path)
    cli.add_argument("--llm", action="store_true")
    args = cli.parse_args()
    manifest = args.manifest.resolve()
    try:
        rows = validate(manifest)
        digest = sha256(manifest)
        if args.action == "validate":
            result = {"manifest_sha256": digest, "samples": len(rows)}
        else:
            subset = [r for r in rows if r["split"] == args.split]
            if not subset:
                raise ValueError(f"清单没有 {args.split} 样本")
            if args.action == "run":
                result = {"manifest_sha256": digest, "split": args.split, "reports": run(manifest, subset, args.llm)}
            else:
                if not args.predictions:
                    raise ValueError("需要 --predictions")
                prediction = json.loads(args.predictions.read_text(encoding="utf-8"))
                if (prediction["manifest_sha256"], prediction["split"]) != (digest, args.split):
                    raise ValueError("预测清单哈希或分区不符")
                by_id = {r["id"]: r for r in prediction["reports"]}
                if len(by_id) != len(prediction["reports"]) or set(by_id) != {r["id"] for r in subset}:
                    raise ValueError("预测与样本 ID 不一一对应")
                pairs = [(r, by_id[r["id"]]) for r in subset]
                if any(r.get("sha256") != p["image_sha256"] for r, p in pairs):
                    raise ValueError("预测没有绑定原图哈希")
                if args.action == "calibrate":
                    if args.split != "calibration":
                        raise ValueError("只能从 calibration 分区选阈值")
                    result = calibrate(pairs, digest)
                else:
                    profile = json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else None
                    if profile and profile["calibration_manifest_sha256"] != digest:
                        raise ValueError("阈值所用的冻结清单哈希不符")
                    metrics = evaluate(pairs, profile, manifest.parent)
                    result = {"manifest_sha256": digest, "split": args.split, "profile": profile,
                              "metrics": metrics, "blind_checks": release_checks(metrics) if args.split == "blind" else None}
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.out:
            args.out.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        cli.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

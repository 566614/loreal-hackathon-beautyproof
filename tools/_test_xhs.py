# -*- coding: utf-8 -*-
"""Real-world stress test: run BeautyProof on the XHS lipstick beauty images
(extracted from MediaCrawler zip) + their ad copy. Model loaded once.
"""
import sys, json, zipfile
from pathlib import Path
sys.path.insert(0, str(Path("tools").resolve()))
import aigc_tool, c2pa_tool, hash_tool, text_tool

REPO = Path(".")
XHS_ZIP = r"C:\Users\Lenovo\Documents\xwechat_files\wxid_fwgf45apbp8422_3543\msg\file\2026-09\MediaCrawler数据.zip"
IMG_DIR = REPO / "outputs" / "_external_xhs"

# ---- image side ----
imgs = sorted([x for x in IMG_DIR.rglob("*") if x.suffix.lower() in (".webp", ".png")])
mdir = aigc_tool.model_path()
print(f"aigc model dir: {mdir}")
aigc_tool._load_ft_model(mdir)  # load once

img_rows = []
for img in imgs:
    ai_score, ai_label, avail = aigc_tool.run_aigc_timm(str(img), mdir)
    tc = c2pa_tool.read_tc260_aigc(str(img)) or {}
    aid = hash_tool.compute_asset_id(str(img))
    img_rows.append({
        "file": str(img.relative_to(IMG_DIR)),
        "note": img.parent.name,
        "aigc_score": ai_score,
        "aigc_label": ai_label,
        "tc260_present": bool(tc),
        "asset_id": aid[:12],
    })

scores = [r["aigc_score"] for r in img_rows]
print(f"\n=== IMAGE SIDE ({len(img_rows)} real-world beauty images) ===")
print(f"aigc min/max: {min(scores):.4f} / {max(scores):.4f}")
print(f"  false positive (>=0.9): {sum(1 for s in scores if s >= 0.9)}")
print(f"  uncertain   (0.5-0.9):  {sum(1 for s in scores if 0.5 <= s < 0.9)}")
print(f"  clean real  (<0.5):     {sum(1 for s in scores if s < 0.5)}")
print(f"  TC260 present: {sum(1 for r in img_rows if r['tc260_present'])} (expect 0)")

# ---- text side ----
z = zipfile.ZipFile(XHS_ZIP)
raw = z.read("MediaCrawler数据/xhs/jsonl/search_contents_2026-09-23.jsonl").decode("utf-8", "ignore")
notes = [json.loads(l) for l in raw.splitlines() if l.strip()]
text_rows = []
for r in notes:
    text = (r.get("title") or "") + "\n" + (r.get("desc") or "")
    ev = text_tool.build_evidence(text, source_id=r["note_id"])
    e0 = ev["evidence"][0]
    text_rows.append({
        "note_id": r["note_id"],
        "title": r.get("title"),
        "claim_count": e0.get("claim_violation_count"),
        "claim_high": e0.get("claim_high_count"),
        "claim_medium": e0.get("claim_medium_count"),
        "claim_risk": e0.get("claim_risk_level"),
        "top_claim": (e0["claim_violations"][0]["word"] if e0.get("claim_violations") else None),
    })

print(f"\n=== TEXT SIDE ({len(text_rows)} lipstick ad copies) ===")
print(f"  notes with any 违禁宣称命中: {sum(1 for t in text_rows if t['claim_count'] > 0)}")
print(f"  notes with HIGH-risk 宣称:   {sum(1 for t in text_rows if t['claim_high'])}")
for t in text_rows:
    if t["claim_count"] > 0:
        print(f"    {t['note_id']} [{str(t['title'])[:18]}] risk={t['claim_risk']} "
              f"hits={t['claim_count']} top={t['top_claim']}")

out = REPO / "results" / "xhs_beauty_test.json"
out.write_text(json.dumps({"images": img_rows, "text": text_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwritten {out}")

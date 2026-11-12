"""Import the repository's source-checked brand excerpts into the local catalog."""
import json
from pathlib import Path

from claim_review import add_verified_reference, verified_catalog


def main():
    source = Path(__file__).resolve().parents[1] / "data" / "cn_brand_claims.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if any(row.get("source_type") != "brand_claim" for row in rows):
        raise ValueError("品牌资料导入只接受品牌宣称，不能导入功效试验")
    existing = {(row["source"], row["product"], row["excerpt"]) for row in verified_catalog().values()}
    added = 0
    for row in rows:
        key = (row["source"], row["product"], row["excerpt"])
        if key not in existing:
            add_verified_reference({**row, "reviewer_confirmed": True})
            existing.add(key)
            added += 1
    print(f"新增 {added} 条品牌资料；品牌宣称不会自动成为功效试验证据。")


if __name__ == "__main__":
    main()

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_doubao_split import prepare_split


def test_source_groups_and_copies_stay_fixed_on_retry(tmp_path):
    images = tmp_path/"source"
    images.mkdir()
    rows = []
    reals = []
    for group in range(5):
        name = f"product-{group}"
        reals.append({"file": f"real-{group}.jpg", "group": name})
        for version in range(2):
            path = images/f"{group}-{version}.jpg"
            data = f"verified original bytes {group} {version}".encode()
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            rows.append({"path": str(path), "sha256": digest, "pixel_sha256": digest,
                         "group": name, "batch": f"batch-{version}"})
    reviewed = tmp_path/"reviewed.json"
    reviewed.write_text(json.dumps(rows), encoding="utf8")
    membership = tmp_path/"real-groups.json"
    membership.write_text(json.dumps({"files": reals}), encoding="utf8")
    output = tmp_path/"split"

    first = prepare_split(reviewed, membership, images, output)
    locked = (output/"manifest.json").read_bytes()
    second = prepare_split(reviewed, membership, images, output)
    assert first == second
    assert (output/"manifest.json").read_bytes() == locked
    assert first["ai_images"] == {"train": 8, "validation": 2}
    assert first["real_controls"] == {"train": 4, "validation": 1}
    manifest = json.loads(locked)
    roles = {}
    for row in manifest:
        roles.setdefault(row["group"], set()).add(row["split"])
        destination = output/row["split"]/"ai"/row["file"]
        assert destination.read_bytes() == Path(row["source_path"]).read_bytes()
    assert all(len(splits) == 1 for splits in roles.values())

    Path(rows[0]["path"]).write_bytes(b"changed after review")
    with pytest.raises(ValueError, match="内容发生变化"):
        prepare_split(reviewed, membership, images, output)
    assert (output/"manifest.json").read_bytes() == locked

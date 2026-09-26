# -*- coding: utf-8 -*-
"""BeautyProof 工具层单测 —— 覆盖本次重构的共享函数（不依赖深度学习模型，秒跑）。

运行：
    python -m pytest tests/test_utils.py -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import _imageutil    # noqa: E402
import rule_engine  # noqa: E402
import planner       # noqa: E402
import roundtable    # noqa: E402


class TestImageUtil(unittest.TestCase):
    def test_path_returns_data_uri(self):
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        try:
            Image.new("RGB", (20, 10), (255, 0, 0)).save(tmp.name)
            uri = _imageutil.image_to_base64(tmp.name, max_side=100, quality=80)
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        finally:
            os.unlink(tmp.name)

    def test_nonexistent_returns_none(self):
        self.assertIsNone(_imageutil.image_to_base64("no_such_file.png"))

    def test_pil_image_accepted(self):
        from PIL import Image
        img = Image.new("RGB", (40, 30), (0, 255, 0))
        uri = _imageutil.image_to_base64(img, max_side=100, quality=80)
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))


class TestThresholdCentralization(unittest.TestCase):
    """planner / roundtable 必须引用规则引擎的同一套阈值，不能自己写魔法数字。"""
    def test_planner_imports_thresholds(self):
        self.assertIn("aigc_high_risk", rule_engine.THRESHOLDS)
        self.assertIs(planner.THRESHOLDS, rule_engine.THRESHOLDS)

    def test_roundtable_imports_thresholds(self):
        self.assertIs(roundtable.THRESHOLDS, rule_engine.THRESHOLDS)

    def test_first_helper_shared(self):
        self.assertIs(planner._first, rule_engine._first)
        self.assertIs(roundtable._first, rule_engine._first)


if __name__ == "__main__":
    unittest.main()

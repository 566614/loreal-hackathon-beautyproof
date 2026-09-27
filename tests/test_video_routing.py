# -*- coding: utf-8 -*-
"""输入类型路由单测 —— 视频成为一等公民，同时图像路径的行为一点都不能变。

只测「路由判断 + 统一证据字段」，不跑 cv2、不跑模型，秒级。

运行：
    python -m pytest tests/test_video_routing.py -v
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import pipeline   # noqa: E402
import video_tool  # noqa: E402


class TestVideoRouting(unittest.TestCase):
    """pipeline.main() 的输入类型路由：视频走视频流水线，图片走原路径。"""

    def test_video_extensions_recognized(self):
        for ext in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
            self.assertTrue(pipeline.is_video_input(f"demo/x{ext}"),
                            f"{ext} 应该被认成视频")
            # 大小写不敏感：评委在 Windows 上拿到 .MP4 也得认
            self.assertTrue(pipeline.is_video_input(f"demo/x{ext.upper()}"))

    def test_image_is_not_video(self):
        # 红线：图片绝不能被路由到视频流水线
        for name in ("clean_01.png", "a.jpg", "b.jpeg", "c.webp", "no_ext"):
            self.assertFalse(pipeline.is_video_input(f"data/clean/{name}"))

    def test_directory_with_video_counts_as_video(self):
        d = REPO / "demo"
        # demo/ 里有 BeautyProof_演示视频.mp4（真实存在，不造假数据）
        self.assertTrue(any(f.suffix.lower() in pipeline.VIDEO_EXTS
                            for f in d.iterdir()))
        self.assertTrue(pipeline.is_video_input(str(d)))
        vids = pipeline.expand_video_inputs(str(d))
        self.assertTrue(vids)
        self.assertTrue(all(v.suffix.lower() in pipeline.VIDEO_EXTS for v in vids))

    def test_directory_without_video_is_not_video(self):
        self.assertFalse(pipeline.is_video_input(str(REPO / "tools")))

    def test_value_flags_not_treated_as_paths(self):
        """--max-frames 6 里的 6、--text 后面的文案都不能被当成输入路径。"""
        self.assertIn("--max-frames", pipeline.VALUE_FLAGS)
        self.assertIn("--fps-sample", pipeline.VALUE_FLAGS)
        self.assertIn("--text", pipeline.VALUE_FLAGS)
        self.assertIn("--text-file", pipeline.VALUE_FLAGS)

    def test_analyze_signature_unchanged_for_image_path(self):
        """红线自检：图像主入口 analyze() 的签名没被这次改动碰过。"""
        import inspect
        params = list(inspect.signature(pipeline.analyze).parameters)
        for p in ("image_path", "fast", "skip", "verbose", "timeout",
                  "on_progress", "text", "llm", "roundtable"):
            self.assertIn(p, params)


class TestVideoUnifiedEvidence(unittest.TestCase):
    """视频级结果必须沿用图片侧那套 {source_asset_id, cannot_prove, ...}。"""

    def test_cannot_prove_mentions_coverage(self):
        self.assertIn("抽帧覆盖率", video_tool.VIDEO_CANNOT_PROVE_TPL)
        self.assertIn("不能证明", video_tool.VIDEO_CANNOT_PROVE_TPL)

    def test_coverage_ratio_computed_not_invented(self):
        cov = video_tool.build_coverage({"total_frames": 4958}, 6)
        self.assertEqual(cov["sampled"], 6)
        self.assertEqual(cov["total_frames"], 4958)
        self.assertEqual(cov["uncovered_frames"], 4952)
        self.assertAlmostEqual(cov["coverage_ratio"], 6 / 4958, places=6)

    def test_coverage_unknown_when_total_missing(self):
        # 读不到总帧数就给 None，绝不编一个好看的覆盖率
        cov = video_tool.build_coverage({"total_frames": 0}, 6)
        self.assertIsNone(cov["coverage_ratio"])
        self.assertIsNone(cov["uncovered_frames"])

    def test_analyze_video_supports_passthrough_switches(self):
        import inspect
        params = list(inspect.signature(video_tool.analyze_video).parameters)
        for p in ("fast", "max_frames", "fps_sample", "llm", "roundtable"):
            self.assertIn(p, params)


class TestVideoAggregation(unittest.TestCase):
    """聚合规则：任一帧 high_risk → 视频 high_risk（这条写在 README/报告里，必须成立）。"""

    def _fr(self, level, fid="frame_0000"):
        return {"frame_id": fid, "timestamp": 0.0, "risk_level": level,
                "reasons": [], "tools_used": []}

    def test_high_risk_frame_wins(self):
        # 1 帧高风险 + 5 帧无法判定 → 视频必须是高风险，不能是"无法判定"
        frames = [self._fr("high_risk")] + [self._fr("inconclusive")] * 5
        self.assertEqual(video_tool.aggregate_risk_level(frames), "high_risk")

    def test_suspicious_higher_than_credible(self):
        frames = [self._fr("credible"), self._fr("suspicious"), self._fr("credible")]
        self.assertEqual(video_tool.aggregate_risk_level(frames), "suspicious")

    def test_credible_beats_inconclusive(self):
        frames = [self._fr("inconclusive"), self._fr("credible")]
        self.assertEqual(video_tool.aggregate_risk_level(frames), "credible")

    def test_error_frames_do_not_downgrade(self):
        # 分析失败的帧不能代表"没问题"，也不该参与定级
        frames = [self._fr("error"), self._fr("inconclusive")]
        self.assertEqual(video_tool.aggregate_risk_level(frames), "inconclusive")

    def test_all_error_is_inconclusive(self):
        self.assertEqual(video_tool.aggregate_risk_level([self._fr("error")]),
                         "inconclusive")

    def test_empty_is_inconclusive(self):
        self.assertEqual(video_tool.aggregate_risk_level([]), "inconclusive")


if __name__ == "__main__":
    unittest.main(verbosity=2)

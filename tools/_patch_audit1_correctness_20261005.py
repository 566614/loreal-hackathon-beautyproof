# -*- coding: utf-8 -*-
"""2026-10-05 · 全面复检 · 第一批：correctness / 健壮性

F821 真 bug
  tools/threshold_sweep.py:91  build_sweep() 引用了从未定义的 BASELINE_SUSP
  → 任何调用该函数的地方都会 NameError（main() 第 137 行就在调）→ 脚本必崩。
  修法：按规则引擎真实语义补齐 —— rule_engine.py 第 40/65 行写明
  「AIGC 分数 ∈ [aigc_high_risk(0.9), aigc_hard_high(0.99)) → suspicious（灰带）」，
  即**灰带入口就是 aigc_high_risk 本身，并不存在独立的 aigc_suspicious 常量**。
  故用 .get 兼容将来新增该键，注释写清依据，不臆造新阈值。

B905 zip() 缺 strict（静默丢证据）
  tools/ai_label_tool.py:91、tools/ocr_tool.py:42 都是
      zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
  PaddleOCR 三个数组本应等长；一旦某版本/某图返回不等长，zip 会**静默截断到最短**，
  导致 OCR 行被悄悄丢弃 → ai_label 读不到图上「AI生成」标识 → 漏检 → 判错。
  这与本项目「证据链可信、每条工具写明不能证明什么」的立身之本直接冲突：
  丢证据必须响亮报错，不能静默。改用 strict=True。
  tools/make_defense_pptx.py:344 的 zip(labels, caps) 是定长 5→5，一并加 strict 固化契约。

幂等：每处先判 new 是否已在，断言 old 命中恰好 1 次。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATCHES = [
    # ---------- F821：threshold_sweep.py ----------
    ("tools/threshold_sweep.py",
     '''try:
    import rule_engine
    BASELINE_HIGH = rule_engine.THRESHOLDS["aigc_high_risk"]
    BASELINE_HARD = rule_engine.THRESHOLDS["aigc_hard_high"]
except Exception:
    BASELINE_HIGH, BASELINE_HARD = 0.9, 0.99''',
     '''try:
    import rule_engine
    BASELINE_HIGH = rule_engine.THRESHOLDS["aigc_high_risk"]
    BASELINE_HARD = rule_engine.THRESHOLDS["aigc_hard_high"]
    # 灰带下沿（AIGC 侧）。依据 rule_engine.py 第 40 / 65 行：AIGC 分数
    # ∈ [aigc_high_risk=0.9, aigc_hard_high=0.99) 即判 suspicious（灰带，转人工不自动下架）——
    # 也就是说灰带入口就是 aigc_high_risk 本身，规则引擎里**没有**独立的 aigc_suspicious 键。
    # 这里用 .get 兼容将来新增该键的情形，不臆造当前不存在的阈值。
    BASELINE_SUSP = rule_engine.THRESHOLDS.get("aigc_suspicious", BASELINE_HIGH)
except Exception:
    BASELINE_HIGH, BASELINE_HARD = 0.9, 0.99
    BASELINE_SUSP = BASELINE_HIGH'''),

    # ---------- B905：OCR 证据行，缺 strict 会静默丢证据 ----------
    ("tools/ai_label_tool.py",
     'for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):',
     'for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True):'),

    ("tools/ocr_tool.py",
     'for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):',
     'for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True):'),

    # ---------- B905：PPT 架构图，定长数组固化契约 ----------
    ("tools/make_defense_pptx.py",
     "for i, (lb, cp) in enumerate(zip(labels, caps)):",
     "for i, (lb, cp) in enumerate(zip(labels, caps, strict=True)):"),
]


def main():
    by_file = {}
    for rel, old, new in PATCHES:
        by_file.setdefault(rel, []).append((old, new))

    for rel, pairs in by_file.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        for old, new in pairs:
            if new in text and old not in text:
                print(f"[skip] {rel} 已应用")
                continue
            n = text.count(old)
            assert n == 1, f"{rel} 命中 {n} 次（应为 1）：{old[:50]}"
            text = text.replace(old, new)
            print(f"[ok]   {rel}  ← {old[:50]}")
        p.write_text(text, encoding="utf-8")

    # 复核：BASELINE_SUSP 已定义且 build_sweep 能真跑通（不抛 NameError）
    import subprocess
    code = (
        "import sys; sys.path.insert(0,'tools');"
        "import threshold_sweep as t;"
        "print('BASELINE_SUSP=', t.BASELINE_SUSP, 'HIGH=', t.BASELINE_HIGH, 'HARD=', t.BASELINE_HARD);"
        "r=t.build_sweep([],[ ]); print('build_sweep OK, 点数 =', len(r))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("\n[验证] threshold_sweep:", (out.stdout or out.stderr).strip()[:300])
    assert out.returncode == 0, "threshold_sweep 仍不可用"
    print("[done] 第一批 correctness 修复完成")


if __name__ == "__main__":
    import sys
    main()

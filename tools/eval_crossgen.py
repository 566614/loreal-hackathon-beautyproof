# -*- coding: utf-8 -*-
"""跨生成器零样本压力测试（生产模型 v4 vs 基线 v3 同图对比）。

对 data/ai_cross_native/ 下 12 张 ImageGen 平台直出的图逐张跑 AIGC 鉴伪。
该组图仅作 held-out 测试，绝不用于训练（合规：与训练份同生成器但零样本、不同提示词/风格）。

本脚本一次性测两个模型，方便答辩时诚实给出「v3=58.3% → v4=?」的对比：
- v3：import 后 pin MODEL_PRIORITY=['beautyproof_aigen_v3']
- v4：pin MODEL_PRIORITY=['beautyproof_aigen_v4']（生产置顶模型）

两个模型都显式 assert model_path() 解析结果，避免误用。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import aigc_tool  # noqa: E402

IMG_DIR = REPO / "data" / "ai_cross_native"
STYLES = {
    "crossgen_01.png": "电商产品静物图（白底精华液）",
    "crossgen_02.png": "带妆容人像（红唇棚拍模特）",
    "crossgen_03.png": "社交媒体滤镜自拍（贴纸+磨皮滤镜）",
    "crossgen_04.png": "美妆品牌海报（渐变背景悬浮口红）",
    "crossgen_05.png": "磨皮精修人像（无毛孔玻璃肌特写）",
    "crossgen_06.png": "复古胶片感（90年代港风颗粒色调）",
    "crossgen_07.png": "赛博朋克霓虹妆容",
    "crossgen_08.png": "水彩插画美妆广告",
    "crossgen_09.png": "3D 渲染化妆品悬浮场景",
    "crossgen_10.png": "电商直播截图风（简陋灯光+直播美颜）",
    "crossgen_11.png": "时尚杂志封面风（硬光金属眼妆）",
    "crossgen_12.png": "生活感护肤场景（浴室窗边逆光）",
}


def run_for(model_name):
    aigc_tool.MODEL_PRIORITY = [model_name]
    resolved = aigc_tool.model_path()
    assert resolved is not None and model_name in resolved, \
        f"pin {model_name} 失败：实际解析到 {resolved}"
    per = []
    for f in sorted(IMG_DIR.glob("crossgen_*.png")):
        prob, label, available = aigc_tool.run_aigc(f)
        per.append({
            "file": f.name,
            "style_hint": STYLES.get(f.name, ""),
            "ai_prob": (round(prob, 4) if prob is not None else None),
            "label": label,
            "available": available,
            "verdict_at_0.6": ("ai_generated" if (available and prob is not None and prob >= 0.6) else "not_ai"),
            "verdict_at_0.5": ("ai_generated" if (available and prob is not None and prob >= 0.5) else "not_ai"),
        })
    n = sum(1 for r in per if r["available"])
    recall_06 = round(sum(1 for r in per if r["verdict_at_0.6"] == "ai_generated") / n, 4) if n else None
    recall_05 = round(sum(1 for r in per if r["verdict_at_0.5"] == "ai_generated") / n, 4) if n else None
    return {"model": model_name, "resolved_dir": Path(resolved).name,
            "per_image": per, "recall_at_0.6": recall_06, "recall_at_0.5": recall_05}


def main():
    out = {}
    out["generated_at"] = datetime.now().isoformat(timespec="seconds")
    out["provenance"] = ("本组图全部由平台 ImageGen 于 2026-09-26 生成，仅作 held-out 测试，"
                         "绝不用于训练；与训练份同生成器但零样本、不同提示词/风格。")
    out["n_images"] = len(list(IMG_DIR.glob("crossgen_*.png")))
    out["v3"] = run_for("beautyproof_aigen_v3")
    out["v4"] = run_for("beautyproof_aigen_v4")
    out["conclusion"] = (
        f"v3 零样本召回 {out['v3']['recall_at_0.6']} → v4 {out['v4']['recall_at_0.6']}；"
        "注意：该组仍是 ImageGen 同生成器族，不等于 MJ/SD/Flux 等真·第三方生成器，"
        "换内核仍需重新校准（最高 ROI 路径是团队自拍真实美妆图 → 伪标注 → v4 再训练）。"
    )
    report_path = REPO / "results" / "crossgen_eval_v4.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

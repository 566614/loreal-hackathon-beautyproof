# -*- coding: utf-8 -*-
"""临时脚本：跨生成器零样本压力测试（维度B·短板补齐）。

对 data/ai_cross_native/ 下 12 张 ImageGen 平台直出的图逐张跑 AIGC 鉴伪。
关键点：显式 pin v3 —— import 后立刻覆盖 MODEL_PRIORITY，确保另一个 agent
正在训练的 beautyproof_aigen_v4 中途出现也不会被用到。
用完即删。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import aigc_tool  # noqa: E402

# ---- 显式 pin v3（必须在任何 run_aigc 调用之前）----
aigc_tool.MODEL_PRIORITY = ["beautyproof_aigen_v3"]

# 自检：确认 model_path() 解析到的目录确实是 v3
resolved = aigc_tool.model_path()
assert resolved is not None and "beautyproof_aigen_v3" in resolved, \
    f"pin 失败：实际解析到 {resolved}"
resolved_name = Path(resolved).name

# 检查 v4 是否已在 models 下出现（仅作记录，评测不使用它）
v4_dir = REPO / "models" / "beautyproof_aigen_v4"
v4_present = v4_dir.exists()

IMG_DIR = REPO / "data" / "ai_cross_native"
styles = {
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

per_image = []
for f in sorted(IMG_DIR.glob("crossgen_*.png")):
    prob, label, available = aigc_tool.run_aigc(f)
    per_image.append({
        "file": f.name,
        "style_hint": styles.get(f.name, ""),
        "ai_prob": prob,
        "label": label,
        "available": available,
        "verdict_at_0.6": ("ai_generated" if (available and prob is not None and prob >= 0.6) else "not_ai"),
        "verdict_at_0.5": ("ai_generated" if (available and prob is not None and prob >= 0.5) else "not_ai"),
    })

n = sum(1 for r in per_image if r["available"])
recall_06 = round(sum(1 for r in per_image if r["verdict_at_0.6"] == "ai_generated") / n, 4) if n else None
recall_05 = round(sum(1 for r in per_image if r["verdict_at_0.5"] == "ai_generated") / n, 4) if n else None

report = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "provenance": "本组图全部由平台 ImageGen 于 2026-09-26 生成，仅作 held-out 测试，绝不用于训练",
    "evaluated_model": resolved_name,
    "model_pin_confirmed": f"import aigc_tool 后立即设 aigc_tool.MODEL_PRIORITY=['beautyproof_aigen_v3']，"
                           f"并在脚本内 assert model_path() 解析结果为 {resolved_name}",
    "v4_observed_in_models_dir": v4_present,
    "per_image": per_image,
    "recall_at_0.6": recall_06,
    "recall_at_0.5": recall_05,
    "conclusion": "",
}

# 结论占位，等看到分数后手工定稿
out = REPO / "results" / "crossgen_eval.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))

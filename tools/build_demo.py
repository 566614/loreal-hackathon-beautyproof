# -*- coding: utf-8 -*-
"""
把 Web 演示页打包成「纯静态、双击就能看」的离线 Demo

用法：
    ./run.sh tools/build_demo.py

它做三件事：
    1. 读 outputs/ 下所有 analysis_*.json（每张图鉴定一次的完整结果）
    2. 读 web/templates/index.html（网页长什么样只有这一份，不会改出两个版本）
    3. 把结果数据以内嵌 <script> 注入，另存为 demo/index.html —— 图片全部 base64 内嵌，
       单个 HTML 文件，可以发群里、挂静态托管，不依赖任何后端
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import pipeline  # noqa: E402  —— 只为拿到 TOOL_META（每个工具的人话说明）
OUT_DIR = REPO / "outputs"
TEMPLATE = REPO / "web" / "templates" / "index.html"
DEST = REPO / "demo" / "index.html"


def main():
    # 只收录 manifest 登记过的正式样本 —— 不然 Web 调试时上传的图也会混进来
    manifest = json.loads((REPO / "data" / "manifest.json").read_text(encoding="utf-8"))
    keep = {Path(s["path"]).stem: s for s in manifest.get("samples", [])}

    samples = []
    for f in sorted(OUT_DIR.glob("analysis_*.json")):
        if f.stem[len("analysis_"):] not in keep:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        v = data["verdict"]
        # 兜底：早期生成的分析结果里没有工具说明，补上，免得页面上六张卡片描述空白
        data.setdefault("tool_meta", pipeline.TOOL_META)
        samples.append({
            "stem": data["image"]["stem"],
            "name": data["image"]["name"],
            "risk_level": v["risk_level"],
            "risk_zh": {
                "high_risk": "高风险", "suspicious": "可疑",
                "credible": "可信", "inconclusive": "无法判定",
            }.get(v["risk_level"], v["risk_level"]),
            "analysis": data,
        })
    if not samples:
        print("outputs/ 下没有 analysis_*.json —— 先跑 tools/pipeline.py 生成")
        return 1

    html = TEMPLATE.read_text(encoding="utf-8")
    # 页面标题改成离线版，避免误导（静态页没有上传功能）
    html = html.replace(
        "<title>美妆图取证台 · BeautyProof</title>",
        "<title>美妆图取证台 · BeautyProof（离线演示版）</title>")
    inject = ("<script>window.__DEMO_DATA__ = "
              + json.dumps({"samples": samples}, ensure_ascii=False)
              + ";</script>\n</body>")
    html = html.replace("</body>", inject, 1)

    DEST.parent.mkdir(exist_ok=True)
    DEST.write_text(html, encoding="utf-8")
    print(f"离线 Demo 已生成: {DEST}（{DEST.stat().st_size / 1024 / 1024:.1f} MB，"
          f"含 {len(samples)} 个样本）")
    print("双击用浏览器打开即可，无需任何服务。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

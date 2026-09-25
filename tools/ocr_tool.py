# -*- coding: utf-8 -*-
# 来源：第三方调用 —— PaddleOCR (PP-OCRv6, Apache-2.0) 为被调用能力；统一证据封装为 BeautyProof 团队原创
"""
OCR 检测工具 —— 把图上的文字抄下来，输出「统一格式」的证据 JSON

用法（在仓库目录下）：
    python tools/ocr_tool.py <图片路径>

比如：
    python tools/ocr_tool.py ../my-beautyproof-practice/assets/post.jpg

会做什么：
    1. 调已装好的 PaddleOCR，把图上的文字一行一行抄下来；
    2. 每一行记下：文字内容、在图中的位置框(bbox)、抄写的把握(confidence)；
    3. 打包成「统一证据格式」的 JSON —— 后面哈希/C2PA/TruFor/AIGC 全都照这个格式来，
       这样规则引擎才能用同一套代码读所有工具的结果。

铁律：
    OCR 是抄写员，不是事实核查员。confidence 只说明「抄得像不像」，
    不代表这句话是真的，也不代表图被改过。
"""
import json
import sys
from pathlib import Path


def run_ocr(image_path):
    """调 PaddleOCR 抄图上的字。返回一堆行，每行含 text / bbox / confidence"""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="paddle",
    )

    result = ocr.predict(str(image_path))

    lines = []
    for res in result:
        for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            lines.append({
                "text": str(text),
                "bbox": [round(min(xs), 1), round(min(ys), 1),
                         round(max(xs), 1), round(max(ys), 1)],
                "confidence": round(float(score), 4),
            })
    return lines


def build_evidence(image_path, lines):
    """把 OCR 结果打包成统一证据格式。后面所有工具都照抄这个结构"""
    return {
        "tool": "ocr",
        "source_asset_id": Path(image_path).name,
        "observed": f"在图上一共读出 {len(lines)} 行文字",
        "cannot_prove": "OCR 只是抄写员，不能证明文字内容为真，也不能证明图片被篡改",
        "evidence": lines,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/ocr_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    lines = run_ocr(image_path)
    report = build_evidence(image_path, lines)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"ocr_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

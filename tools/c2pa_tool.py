# -*- coding: utf-8 -*-
"""
内容凭证工具 —— 查图片里的两种「身份凭证」，输出统一格式的证据 JSON

用法：
    python tools/c2pa_tool.py <图片路径>

大白话：
    查两种凭证，逻辑都是「有就照着读，没有不下结论」：

    1. C2PA（国际标准）：相机或修图软件留下的「出生证明」，
       记录这张图是谁拍的、用什么软件改过。
    2. TC260 AIGC 标识（中国强制标准）：《人工智能生成合成内容标识办法》
       （网信办等四部门 2025-03-07 发布、2025-09-01 施行）要求
       AI 生成内容必须打显式标识。合规的生成器会在文件元数据里写入
       TC260:AIGC 字段（Label=1 表示 AI 生成，含生产者编码与生产 ID）。

    2026-09-23 实测：即梦导出的 PNG 全部带有 TC260 标识 —— 读元数据即可确认，
    不需要任何模型推理。这是全套工具里证据强度最高的一种。

能证明什么：
    有 TC260 AIGC 标识 → 生成器自己声明了「这是 AI 生成的」，法律依据明确。
    有 C2PA 凭证 → 编辑历史可查。

不能证明什么（铁律）：
    没有凭证 ≠ 是假图。目前网上绝大多数图两种凭证都没有。
    ⚠️ 标识可以被剥离：重新保存、截图、裁剪、转格式都可能丢掉元数据。
       所以「有标识」是强证据，「没标识」什么都不是。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

C2PA_EXE = r"C:\Users\Lenovo\AppData\Local\Programs\c2patool\c2patool\c2patool.exe"

NO_CLAIM_HINTS = [
    "no claim", "no manifest", "no embedded", "not found",
    "noclaim", "error", "unable", "no c2pa",
]


def run_c2pa(image_path):
    """调已装好的 c2patool，读出图片里的凭证信息"""
    try:
        result = subprocess.run(
            [C2PA_EXE, str(image_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return "", "找不到 c2patool，请检查脚本里的 C2PA_EXE 路径", 127
    except OSError as e:
        return "", f"运行 c2patool 失败：{e}", 127

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return stdout, stderr, result.returncode


# ------------------------------------------------------------- TC260 AIGC 标识
def read_tc260_aigc(image_path):
    """读《人工智能生成合成内容标识办法》要求的 TC260:AIGC 显式标识

    PNG：XMP 文本块里的 <TC260:AIGC>{...}</TC260:AIGC>
    JPEG：XMP 段（先按文本粗搜，读不出就当没有 —— 不硬猜）
    """
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            xmp = ""
            if hasattr(im, "text") and im.text:
                xmp = im.text.get("XML:com.adobe.xmp", "") or ""
            if not xmp:
                # JPEG 的 XMP 藏在原始字节里，粗搜一段足够覆盖本场景
                with open(image_path, "rb") as f:
                    raw = f.read(4 * 1024 * 1024)
                i = raw.find(b"<TC260:AIGC>")
                if i < 0:
                    return None
                j = raw.find(b"</TC260:AIGC>", i)
                xmp = raw[i:j + 13].decode("utf-8", errors="ignore") if j > 0 else ""
    except Exception:  # noqa: BLE001
        return None

    m = re.search(r"<TC260:AIGC>(.*?)</TC260:AIGC>", xmp, re.S)
    if not m:
        return None
    payload = m.group(1).replace("&quot;", '"').strip()
    try:
        data = json.loads(payload)
    except Exception:  # noqa: BLE001
        data = {"raw": payload[:200]}

    is_ai = str(data.get("Label")) == "1"
    return {
        "present": True,
        "is_ai_generated": is_ai,
        "label": data.get("Label"),
        "content_producer": (data.get("ContentProducer") or "")[:40],
        "produce_id": (data.get("ProduceID") or "")[:60],
        "content_propagator": (data.get("ContentPropagator") or "")[:40],
        "basis": "《人工智能生成合成内容标识办法》（2025-09-01 施行）+ GB/T TC260 AIGC 元数据规范",
    }


def judge(stdout, stderr, returncode):
    """判断有没有凭证。返回 (状态, 人类可读的一句话)"""
    if "找不到 c2patool" in stderr or "运行 c2patool" in stderr:
        return "error", stderr

    combined = (stdout + stderr).lower()

    if returncode != 0:
        return "missing", "c2patool 没能读出凭证（绝大多数网图都没有凭证，属正常）"

    if not stdout:
        return "missing", "这张图没有 C2PA 内容凭证（绝大多数网图都没有，属正常）"

    for hint in NO_CLAIM_HINTS:
        if hint in combined:
            return "missing", "这张图没有 C2PA 内容凭证（绝大多数网图都没有，属正常）"

    return "present", "这张图带有 C2PA 内容凭证，需人工查看凭证内容确认编辑历史"


def build_evidence(image_path, status, note, raw_output):
    tc = read_tc260_aigc(image_path)

    if tc and tc.get("is_ai_generated"):
        observed = (f"文件元数据带 TC260 AIGC 强制标识（Label={tc.get('label')}），"
                    f"生成器自声明这是 AI 生成内容（生产者编码 {tc.get('content_producer')}…）。"
                    f"依据 {tc.get('basis')}。"
                    f"这是全部工具里证据强度最高的一种：不需要模型推理，直接读文件即可复核。")
    elif tc:
        observed = ("文件带 TC260 AIGC 元数据字段，但 Label 未标记为 AI 生成，"
                    "需人工查看字段内容。")
    elif status == "present":
        observed = note + " 未检出 TC260 AIGC 标识。"
    else:
        observed = (note + " 未检出 TC260 AIGC 标识"
                    "（注意：标识可能被截图/转格式/重保存剥离，无标识不代表不是 AI 生成）。")

    return {
        "tool": "c2pa",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "cannot_prove": ("没有凭证/标识不等于图片是假的；有凭证也不等于没被恶意编辑过；"
                         "TC260 标识可能被剥离，缺失时不能反推结论"),
        "evidence": [
            {
                "c2pa_status": status,
                "detail": note,
                "tc260_aigc": tc,
                "tc260_present": bool(tc),
                "tc260_is_ai_generated": bool(tc and tc.get("is_ai_generated")),
            },
        ],
        "raw_output": raw_output[:2000],
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/c2pa_tool.py <图片路径>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"找不到这张图: {image_path}")
        return 1

    stdout, stderr, returncode = run_c2pa(image_path)
    status, note = judge(stdout, stderr, returncode)
    # 有 TC260 强制标识时，凭证状态视为 present（这是更强的凭证）
    if read_tc260_aigc(image_path):
        status = "present"
    report = build_evidence(image_path, status, note, stdout or stderr)

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"c2pa_{image_path.stem}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存到: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

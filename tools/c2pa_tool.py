# -*- coding: utf-8 -*-
# 来源：第三方调用 + 原创解析 —— c2patool 为被调用外部工具；TC260 标识解析与统一证据封装为 BeautyProof 团队原创
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
import os
import re
import subprocess
import sys
from pathlib import Path

# ⚠️ 不再写死开发者机器的 Windows 路径（reviewer P1：干净环境复现要求）。
# 改为「环境变量 C2PA_EXE → PATH 里的 c2patool → 常见安装位置」三级解析，
# 找不到时返回 error 而非崩溃，由规则引擎降级为「待核验」。
def find_c2patool():
    """跨平台定位 c2patool 可执行文件，返回路径或 None"""
    import shutil
    env = (Path(os.environ["C2PA_EXE"]) if os.environ.get("C2PA_EXE")
           else None)
    if env and env.exists():
        return str(env)
    on_path = shutil.which("c2patool")
    if on_path:
        return on_path
    for cand in (
        r"C:\Users\Lenovo\AppData\Local\Programs\c2patool\c2patool\c2patool.exe",
        r"C:\Program Files\c2patool\c2patool.exe",
        "/usr/local/bin/c2patool",
        "/opt/homebrew/bin/c2patool",
    ):
        if Path(cand).exists():
            return cand
    return None


NO_CLAIM_HINTS = [
    "no claim", "no manifest", "no embedded", "not found",
    "noclaim", "error", "unable", "no c2pa",
]


def run_c2pa(image_path):
    """调已装好的 c2patool，读出图片里的凭证信息"""
    exe = find_c2patool()
    if not exe:
        return "", "找不到 c2patool：请设置环境变量 C2PA_EXE 或将其加入 PATH", 127
    try:
        result = subprocess.run(
            [exe, str(image_path)],
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


def parse_c2pa_claim(stdout):
    """尽量从 c2patool 输出里解析出「有无可验证的 manifest」「是否声明 AI 生成」。

    返回 dict：
        has_manifest  —— 输出里是否解析到 C2PA active_manifest（没有则不是真凭证）
        verified      —— 凭证签名/声明是否通过验证（c2patool 自己给出 validation_status=valid）
        declares_ai   —— manifest/断言里是否标记该内容为 AI 生成
    注意：离线环境无法做完整密码学校验，verified 仅当 c2patool 明确给出 valid 才置 True；
    其余情况一律 present_unverified，由规则引擎降级为「待核验」，绝不自动判可信。
    """
    out = {"has_manifest": False, "verified": False, "declares_ai": False}
    if not stdout:
        return out
    try:
        data = json.loads(stdout)
    except Exception:
        # 非空但不可解析 —— 不能当成有凭证
        return out
    if isinstance(data, dict):
        manifest = data.get("active_manifest") or data.get("manifest")
    else:
        manifest = None
    if not isinstance(manifest, dict):
        return out
    out["has_manifest"] = True
    vs = str(data.get("validation_status", "")).lower()
    # 仅当 c2patool 自身判定 valid 才算验证通过；仅存在 signature 结构不视为已验证
    out["verified"] = (vs == "valid")
    blob = json.dumps(manifest).lower()
    out["declares_ai"] = any(
        k in blob for k in ("ai-generated", "ai_generated", "is_ai", "\"ai\"")
    )
    return out


def judge(stdout, stderr, returncode):
    """判断凭证状态。返回 (状态, 一句话, 解析标记 dict)

    状态词汇（reviewer P0 要求拆分「声明存在/验证通过/声明为AI生成」）：
        error              —— 工具本身不可用/运行失败
        missing            —— 无凭证 / 输出不可解析为凭证（绝大多数网图属此）
        present_unverified —— 有 manifest，但本环境无法验证签名/声明 → 待核验
        present_verified   —— 有 manifest 且 c2patool 验证通过 → 才考虑提升可信度
    关键修复：旧逻辑只要「非空且不含错误关键词」就返回 present；
    现在必须解析到真实 manifest，否则按 missing 处理——杜绝「非空字符串即可信」。
    """
    claim_marker = {"verified": False, "declares_ai": False}
    if "找不到 c2patool" in stderr or "运行 c2patool" in stderr:
        return "error", stderr, claim_marker

    combined = (stdout + stderr).lower()

    if returncode != 0:
        return "missing", "c2patool 没能读出凭证（绝大多数网图都没有凭证，属正常）", claim_marker

    if not stdout:
        return "missing", "这张图没有 C2PA 内容凭证（绝大多数网图都没有，属正常）", claim_marker

    # 原 NO_CLAIM_HINTS 仍作为「明确报错/无声明」的快速拦截
    for hint in NO_CLAIM_HINTS:
        if hint in combined:
            return "missing", "这张图没有 C2PA 内容凭证（绝大多数网图都没有，属正常）", claim_marker

    claim = parse_c2pa_claim(stdout)
    if not claim["has_manifest"]:
        # 非空输出但解析不到真实 manifest —— 不能当 present，按无凭证处理
        return "missing", "c2patool 输出非空但无可解析的 C2PA 凭证，按无凭证处理", claim_marker

    claim_marker = {"verified": claim["verified"], "declares_ai": claim["declares_ai"]}
    if claim["declares_ai"]:
        status = "present_verified" if claim["verified"] else "present_unverified"
        note = ("凭证/标识显示该内容由 AI 生成（生成器自声明），"
                "属 AI 生成证据而非来源可信证明，需人工确认")
        return status, note, claim_marker
    if claim["verified"]:
        return "present_verified", "该图带有已验证的 C2PA 内容凭证，编辑历史可查", claim_marker
    return "present_unverified", "该图带有 C2PA 内容凭证，但本环境无法验证签名/声明，保留为待核验", claim_marker


def build_evidence(image_path, status, note, raw_output, verified=False, declares_ai=False):
    tc = read_tc260_aigc(image_path)
    tc_ai = bool(tc and tc.get("is_ai_generated"))
    # TC260 强制标识是比 C2PA 更硬的证据：Label=1 直接声明 AI 生成
    declares_ai = bool(declares_ai) or tc_ai

    if tc_ai:
        observed = (f"文件元数据带 TC260 AIGC 强制标识（Label={tc.get('label')}），"
                    f"生成器自声明这是 AI 生成内容（生产者编码 {tc.get('content_producer')}…）。"
                    f"依据 {tc.get('basis')}。"
                    f"这是全部工具里证据强度最高的一种：不需要模型推理，直接读文件即可复核。"
                    f"但这只证明「生成器自认是 AI 生成」，不构成「来源可信」，"
                    f"规则引擎据此标为可疑而非可信。")
    elif tc:
        observed = ("文件带 TC260 AIGC 元数据字段，但 Label 未标记为 AI 生成，"
                    "需人工查看字段内容。")
    elif status in ("present_verified", "present_unverified"):
        observed = (note + " 未检出 TC260 AIGC 强制标识。"
                    "注意：C2PA 凭证存在不等于内容真实，仍需结合图像侧检测；"
                    "本环境未验证签名时，凭证不得自动提升为可信。")
    else:
        observed = (note + " 未检出 TC260 AIGC 标识"
                    "（注意：标识可能被截图/转格式/重保存剥离，无标识不代表不是 AI 生成）。")

    return {
        "tool": "c2pa",
        "source_asset_id": Path(image_path).name,
        "observed": observed,
        "cannot_prove": ("没有凭证/标识不等于图片是假的；有凭证也不等于没被恶意编辑过；"
                         "TC260 标识可能被剥离，缺失时不能反推结论；"
                         "本环境未做密码学验证的 C2PA 凭证不得直接判为可信"),
        "evidence": [
            {
                "c2pa_status": status,
                "c2pa_verified": bool(verified),
                "c2pa_declares_ai_generated": declares_ai,
                "detail": note,
                "tc260_aigc": tc,
                "tc260_present": bool(tc),
                "tc260_is_ai_generated": tc_ai,
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
    status, note, claim = judge(stdout, stderr, returncode)
    verified = claim.get("verified", False)
    # 有 TC260 强制标识（AI 生成）时：凭证状态至少记为 present_unverified，
    # 且 declares_ai 一定为真（main 内 build_evidence 还会再确认一次 tc_ai）。
    tc = read_tc260_aigc(image_path)
    if tc and tc.get("is_ai_generated") and status == "missing":
        status = "present_unverified"
    report = build_evidence(image_path, status, note, stdout or stderr,
                            verified=verified, declares_ai=claim.get("declares_ai", False))

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

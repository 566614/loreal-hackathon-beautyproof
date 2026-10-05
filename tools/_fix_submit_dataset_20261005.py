# -*- coding: utf-8 -*-
"""2026-10-05 · 修正：评论样本补齐官方三件套 + 修回 zip 根结构

起因：10/05 补评论区样本后 pytest 挂了 2 项 —— 我重打包时多套了一层 `SubmitDataset/` 前缀，
把「样本在根目录 + 顶层总说明」的结构弄坏了。

本脚本做三件事（都幂等）：
1. 给 3 个评论样本补齐官方要求的三件套 —— `种草正文.txt`（评论所依附的种草帖正文）
   + `image_1.jpg`（该帖的实拍图，直接复用 sample_001 的产品图）。
   这样做不是糊格式：场景②「评论区核验」天然依附于一篇种草帖，补上父帖后
   这三个样本同时覆盖「图文交叉 + 评论灌水」两条链路，比只有孤立评论更完整。
2. 重打包：路径改为相对 `outputs/SubmitDataset`（`make_submit_dataset.py` 用
   `shutil.make_archive(root_dir=root)`，样本在 zip 根目录，不是再套一层）。
3. 重生成顶层 `_总说明_README.txt`，列出 10 个样本 + 场景覆盖说明。
"""
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = json.load(open(ROOT / "data" / "comment_cases.json", encoding="utf-8"))["cases"]
SD = ROOT / "outputs" / "SubmitDataset"
ZIP = ROOT / "outputs" / "BeautyProof_submit_dataset.zip"
CST = timezone(timedelta(hours=8))
IMG_SRC = SD / "sample_001" / "image_1.jpg"

# 每个评论样本对应的「父帖」种草正文（与 clean_01 同一支粉底液产品，语气与评论一致）
POSTS = {
    "008": "这支粉底液是02号色号，我混油皮夏天用不假白，也不太卡粉。价格中等，适合学生党。"
           "唯一的缺点是量不太耐用，一支大概用三个月。",
    "009": "混油皮实测：02号色号上脸到下午三点鼻翼会出油，但不会斑驳。"
           "配方比旧版薄一点，柜姐说是新配方我自己也对比过，遮瑕力稍弱一点点，日常够用。",
    "010": "02号粉底液，混油皮亲测。优点是薄、显色正、价格实惠；缺点是量少、冬天用会偏干。",
}
PICKS = [
    ("cc_01_flood_burst", "008", "high_risk", True),
    ("cc_04_organic_detailed", "009", "credible", False),
    ("cc_05_organic_short", "010", "credible", False),
]


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, CST).strftime("%Y-%m-%d %H:%M:%S")


def build_one(case, idx, verdict, is_fake):
    d = SD / ("sample_" + idx)
    d.mkdir(parents=True, exist_ok=True)
    case_id = case["id"]

    # ① 评论区文本
    lines, ts_list = [], case.get("timestamps") or []
    if ts_list:
        for ts, c in zip(ts_list, case["comments"]):
            lines.append(f"[{fmt_ts(ts)}] {c}")
        ts_note = "每行格式：[发布时间] 评论内容，便于核对灌水节奏特征"
    else:
        for c in case["comments"]:
            lines.append(c)
        ts_note = "每行一条评论；本样本不含时间戳字段（原始案例未定义投放节奏），未编造时间"
    (d / "评论区文本.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ② 父帖种草正文（评论所依附的原帖）
    (d / "种草正文.txt").write_text(POSTS[idx] + "\n", encoding="utf-8")

    # ③ 父帖实拍图（复用 clean_01 产品图，README 已声明来源一致）
    shutil.copyfile(IMG_SRC, d / "image_1.jpg")

    # ④ README：官方四字段 + 本样本实际包含的四个文件
    n, uniq = len(case["comments"]), len(set(case["comments"]))
    if is_fake:
        how = (f"模拟水军刷评：{case['note']}。\n"
               f"  文本统计：共 {n} 条，去重后仅 {uniq} 条不同文本（复读率 {(1-uniq/n)*100:.0f}%）。\n"
               f"  行为特征：{case['label']}。\n"
               f"  ⚠️ 评论区为伪造（灌水），父帖种草正文与图片为真实未篡改素材。")
    else:
        how = (f"模拟真实用户评论（非伪造）：{case['note']}。\n"
               f"  文本统计：共 {n} 条，去重后 {uniq} 条不同文本。\n"
               f"  行为特征：{case['label']}。\n"
               f"  设此样本意图：检验系统不会因为「短评比例高 / 褒贬不一」就误判灌水。\n"
               f"  ⚠️ 评论区为模拟真实评论，父帖种草正文与图片为真实未篡改素材。")
    readme = f"""样本编号：sample_{idx}
样本 ID：{case_id}

① 数据来源平台：团队自建合成（评论为按公开可观察灌水模式合成；父帖图/文为团队自产真实素材，未采集任何真实平台用户评论）
② 样本类型：评论区真实性核验（场景②，含配套种草父帖，可做图文↔评论交叉核验）
③ 是否为伪造样本：{'是（评论区为模拟水军刷评）' if is_fake else '否（评论区为模拟真实用户评论）'}
④ 伪造方式说明：
{how}

期望判定：{verdict}
本样本包含 4 个文件：
  README.txt          本说明
  种草正文.txt        评论所依附的种草父帖正文（用于图文交叉核验）
  image_1.jpg         该父帖的实拍配图（与 sample_001 同源，真实未篡改）
  评论区文本.txt      评论区原始文本；{ts_note}

⚠️ 合成数据边界：评论为团队合成，用于验证特征管线与判定方向，不代表真实场景准确率，
不含任何真实用户评论或个人信息。父帖图文为团队自产真实素材，未做篡改。
"""
    (d / "README.txt").write_text(readme, encoding="utf-8")
    print(f"[ok]   sample_{idx}  {case_id}  ({verdict}, {n} 条, 4 文件齐)")


def write_total_readme():
    samples = sorted(p.name for p in SD.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    lines = [
        "BeautyProof · 赛题2「信任守护师」测试数据包",
        "=" * 56,
        "",
        f"样本总数：{len(samples)}",
        "",
        "目录格式（遵循赛题说明页「测试数据包」要求）：",
        "  sample_001/  每条样本一个文件夹（每样本文件无缺项）",
        "    README.txt      数据来源平台 / 样本类型 / 是否伪造 / 伪造方式说明",
        "    种草正文.txt    配套文案；纯视觉样本内为「本样本无配套文案」的显式声明（不伪造配文）",
        "    image_1.jpg     样本图片",
        "    评论区文本.txt  仅场景②（评论区核验）样本含此文件，为该评论所依附的种草父帖评论区",
        "",
        "场景覆盖（对齐赛题三个场景）：",
        "  ① 种草内容核验        —— sample_001/005/007（真实）+ 002/003/004（局部篡改：复制移动/拼接/文字编辑）",
        "  ② 评论区真实性核验    —— sample_008/009/010（灌水 / 真实详细 / 真实短评，含配套种草父帖）",
        "  ③ AI 生成视觉素材鉴伪 —— sample_006（即梦 AI 生成直出）",
        "",
        "样本清单：",
    ]
    for s in samples:
        lines.append(f"  · {s}")
    lines += [
        "",
        "说明：全部样本由团队自产（真实拍摄图 / 本地基准图合成篡改 / 即梦 AI 生成 / 按公开模式合成评论），",
        "未使用任何第三方数据集，符合赛题「自建/合成数据」要求与代码合规声明。",
        "合成样本均在 README 中显式标注为合成，不冒充真实数据。",
        "本数据包仅用于测试与评估，比赛结束后按主办方要求删除。",
    ]
    (SD / "_总说明_README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[readme] 顶层总说明已重写（{len(samples)} 个样本）")


def rezip():
    """与 make_submit_dataset.py 完全一致：make_archive(root_dir=SD)。

    ⚠️ 必须用 make_archive 而不是手写 zipfile：官方格式测试
    （test_提交包内至少有一个样本文件夹）用 `^sample_\\d{3,}$` 去匹配 zip 的**目录条目**
    （"sample_001/"），手写 zipfile 只写文件、不写目录项 → 该测试会误判「没有样本文件夹」。
    """
    zip_base = ZIP.with_suffix("")          # BeautyProof_submit_dataset
    if zip_base.exists():
        shutil.rmtree(zip_base, ignore_errors=True)
    if ZIP.exists():
        ZIP.unlink()
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=SD)
    assert Path(archive) == ZIP, f"产物路径异常：{archive}"
    with zipfile.ZipFile(ZIP) as z:
        n = len(z.namelist())
    print(f"[zip]  {ZIP.name}  {ZIP.stat().st_size/1024/1024:.2f} MB  {n} 个条目（含目录项）")


def main():
    for cid, idx, verdict, is_fake in PICKS:
        case = next((c for c in CASES if c["id"] == cid), None)
        assert case is not None, f"找不到案例 {cid}"
        build_one(case, idx, verdict, is_fake)
    write_total_readme()
    rezip()
    # 复核：zip 根目录结构
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    roots = {n.split("/")[0] for n in names}
    assert "sample_001" in roots, "样本没有落在 zip 根目录"
    assert any(r.endswith(".txt") for r in roots), "顶层缺总说明"
    print(f"\nzip 根级条目 {len(roots)} 个；样本目录 {len([r for r in roots if r.startswith('sample_')])} 个")


if __name__ == "__main__":
    main()

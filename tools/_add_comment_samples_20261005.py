# -*- coding: utf-8 -*-
"""2026-10-05 · 收尾升级：测试数据包补评论区样本（覆盖官方场景②）

问题：原 7 个 sample 全是「AI 视觉素材鉴伪」场景（种草正文只是配套文案），
而官方明文三个场景 = 种草核验 / **评论区真实性核验** / AI 视觉素材鉴伪。
初赛评分点之一是「数据是否涵盖边缘场景」→ 补评论区样本是真实加分，不是灌水。

动作：从 data/comment_cases.json 取 3 个判定方向互补的案例
  high_risk 灌水（复读+节奏爆发） / credible 真实（细节丰富） / credible 真实（都很短，防误报）
按官方 sample_XXX 格式生成，然后重打包 zip。

诚实口径（与项目「不粉饰」一致）：README 明确写「团队自建合成，未采集真实平台评论」，
不把合成样本包装成真实数据 —— 这正是官方要「伪造方式说明」的原因。
幂等：目录已存在则跳过。
"""
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = json.load(open(ROOT / "data" / "comment_cases.json", encoding="utf-8"))["cases"]
SD = ROOT / "outputs" / "SubmitDataset"
ZIP = ROOT / "outputs" / "BeautyProof_submit_dataset.zip"
CST = timezone(timedelta(hours=8))

PICKS = [
    ("cc_01_flood_burst", "008", "high_risk", True),
    ("cc_04_organic_detailed", "009", "credible", False),
    ("cc_05_organic_short", "010", "credible", False),
]


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, CST).strftime("%Y-%m-%d %H:%M:%S")


def build_one(case, idx, verdict, is_fake):
    d = SD / ("sample_" + idx)
    # 幂等判据要看内容文件，不能只看目录在不在：
    # 上一轮若在 mkdir 之后崩溃会留下空目录，只判 d.exists() 会把空样本误当已完成。
    if (d / "README.txt").exists() and (d / "评论区文本.txt").exists():
        print(f"[skip] sample_{idx} 已完整生成")
        return False
    d.mkdir(parents=True, exist_ok=True)

    lines = []
    ts_list = case.get("timestamps") or []
    if ts_list:
        for ts, c in zip(ts_list, case["comments"]):
            lines.append(f"[{fmt_ts(ts)}] {c}")
        ts_note = "每行格式：[发布时间] 评论内容，便于核对灌水节奏特征"
    else:
        # 部分案例不带时间戳（如「都很短」这类只看文本形态的），照实不编造时间
        for c in case["comments"]:
            lines.append(c)
        ts_note = "每行一条评论；本样本不含时间戳字段（原始案例未定义投放节奏），未编造时间"
    (d / "评论区文本.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n = len(case["comments"])
    uniq = len(set(case["comments"]))
    if is_fake:
        how = (f"模拟水军刷评：{case['note']}。\n"
               f"  文本统计：共 {n} 条，去重后仅 {uniq} 条不同文本（复读率 {(1-uniq/n)*100:.0f}%）。\n"
               f"  行为特征：{case['label']}。")
    else:
        how = (f"模拟真实用户评论（非伪造）：{case['note']}。\n"
               f"  文本统计：共 {n} 条，去重后 {uniq} 条不同文本。\n"
               f"  行为特征：{case['label']}。\n"
               f"  设此样本意图：{case['id']} 对应「真实但很短」的边缘形态，"
               f"用于检验系统不会因为「短评比例高」就误判灌水。")
    readme = f"""样本编号：sample_{idx}
样本 ID：{case['id']}

① 数据来源平台：团队自建合成（未采集任何真实平台评论，不含真实用户数据）
② 样本类型：评论区真实性核验
③ 是否为伪造样本：{'是（模拟水军刷评）' if is_fake else '否（模拟真实用户评论）'}
④ 伪造方式说明：
{how}

期望判定：{verdict}
本样本配套文件：评论区文本.txt（{ts_note}）
未附图片：本样本为纯文本场景，官方格式允许文案类内容单独成 .txt。

⚠️ 合成数据边界：本样本为团队按公开可观察的灌水模式合成，用于验证特征管线与判定方向，
不代表真实场景准确率，不含任何真实用户评论或个人信息。
"""
    (d / "README.txt").write_text(readme, encoding="utf-8")
    print(f"[ok]   sample_{idx}  {case['id']}  ({verdict}, {n} 条)")
    return True


def rezip():
    if ZIP.exists():
        ZIP.unlink()
    files = sorted(p for p in SD.rglob("*") if p.is_file())
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(SD.parent))
    print(f"[zip]  {ZIP.name}  {ZIP.stat().st_size/1024/1024:.2f} MB  {len(files)} 个文件")


def main():
    made = False
    for cid, idx, verdict, is_fake in PICKS:
        case = next((c for c in CASES if c["id"] == cid), None)
        assert case is not None, f"找不到案例 {cid}"
        made |= build_one(case, idx, verdict, is_fake)
    rezip()
    samples = sorted(p.name for p in SD.iterdir() if p.is_dir())
    print(f"\n数据包现有 {len(samples)} 个样本：{', '.join(samples)}")


if __name__ == "__main__":
    main()

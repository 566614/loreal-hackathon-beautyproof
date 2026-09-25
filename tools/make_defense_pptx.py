# -*- coding: utf-8 -*-
"""生成 BeautyProof 答辩 PPT（.pptx），对齐 Brandstorm 2026 官方 5+5 评审准则。
输出：docs/BeautyProof_答辩PPT.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

CJK = "Microsoft YaHei"
DARK  = RGBColor(0x12, 0x12, 0x16)
LIGHT = RGBColor(0xFF, 0xFF, 0xFF)
INK   = RGBColor(0x1A, 0x1A, 0x1A)
MUTE  = RGBColor(0x6B, 0x72, 0x80)
GOLD  = RGBColor(0xC8, 0xA2, 0x4B)
ROSE  = RGBColor(0xE8, 0x9A, 0xA8)
PANEL = RGBColor(0xF4, 0xF1, 0xEC)
PANEL2= RGBColor(0x1E, 0x1E, 0x24)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

def set_cjk(run, font=CJK):
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)

def bg(slide, color):
    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = color

def textbox(slide, l, t, w, h, text, size=18, color=INK, bold=False,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.12):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05); tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    first = True
    for line in text.split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align; p.line_spacing = spacing
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = CJK
        set_cjk(r)
    return tb

def bullets(slide, l, t, w, h, items, size=16, color=INK, gap=7, spacing=1.08):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        txt, lvl = it if isinstance(it, tuple) else (it, 0)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = spacing; p.space_after = Pt(gap)
        b = ("      – ") if lvl > 0 else "● "
        r = p.add_run(); r.text = b + txt
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = CJK
        set_cjk(r)
    return tb

def rect(slide, l, t, w, h, fill, line=None, line_w=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sp = slide.shapes.add_shape(shape, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = line_w or Pt(1)
    sp.shadow.inherit = False
    return sp

def callout(slide, l, t, w, h, text, fill=PANEL, barcolor=GOLD, size=15, color=INK):
    rect(slide, l, t, w, h, fill)
    rect(slide, l, t, Inches(0.09), h, barcolor)
    textbox(slide, l + Inches(0.22), t + Inches(0.08), w - Inches(0.34), h - Inches(0.16),
            text, size=size, color=color, anchor=MSO_ANCHOR.MIDDLE)
    return

def header(slide, title, kicker=None):
    rect(slide, 0, 0, Inches(0.22), SH, GOLD)
    textbox(slide, Inches(0.55), Inches(0.42), Inches(12.2), Inches(0.85),
            title, size=29, bold=True, color=INK)
    if kicker:
        textbox(slide, Inches(0.57), Inches(1.22), Inches(12.0), Inches(0.4),
                kicker, size=14, color=MUTE)

def add_table(slide, data, l, t, w, h, header_fill=DARK, header_color=LIGHT,
              body_size=13, head_size=13):
    rows, cols = len(data), len(data[0])
    gtbl = slide.shapes.add_table(rows, cols, l, t, w, h)
    tbl = gtbl.table
    # 关掉默认样式条带
    tbl.first_row = False; tbl.horz_banding = False
    colw = w // cols
    for c in range(cols):
        tbl.columns[c].width = colw
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if ri == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT if ri % 2 else PANEL
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            r.font.size = Pt(head_size if ri == 0 else body_size)
            r.font.bold = (ri == 0)
            r.font.color.rgb = header_color if ri == 0 else INK
            r.font.name = CJK; set_cjk(r)
    return tbl

def new(): return prs.slides.add_slide(BLANK)

# ============ Slide 1: 封面 ============
s = new(); bg(s, DARK)
rect(s, 0, Inches(2.55), SW, Inches(0.06), GOLD)
textbox(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.1),
        "BeautyProof", size=52, bold=True, color=LIGHT)
textbox(s, Inches(0.92), Inches(2.75), Inches(11.5), Inches(0.7),
        "美妆内容信任守护师 · 多模态 AI 取证台", size=24, color=ROSE)
textbox(s, Inches(0.92), Inches(3.75), Inches(11.5), Inches(0.5),
        "欧莱雅第二届美妆科技黑客松 · 赛题 2「信任守护师」", size=16, color=LIGHT)
textbox(s, Inches(0.92), Inches(5.7), Inches(11.5), Inches(0.5),
        "团队：BeautyProof Team（阮 / liying856 / AI 工程）  ·  2026-09", size=14, color=MUTE)

# ============ Slide 2: 痛点 ============
s = new()
header(s, "我们解决的问题", "美妆，是 AI 伪造 / 篡改的重灾区")
bullets(s, Inches(0.6), Inches(1.95), Inches(7.4), Inches(4.6), [
    "种草图、评论区、品牌素材里混着 PS 拼接、局部篡改、整图 AI 生成。",
    "普通消费者看不出「这张口红试色，到底是拍的还是 AI 画的」。",
    "品牌信任资产因此被悄悄侵蚀——赛题要我们「守护信任」。",
    "通用检测模型在中文美妆域集体失效：AIRealNet 基本判反、capcheck 全判 human≈0.99。",
])
callout(s, Inches(8.25), Inches(2.2), Inches(4.5), Inches(2.4),
        "痛点一句话：\n美妆内容真假难辨，而且「通用检测」在我们这个领域根本不准。",
        fill=PANEL2, barcolor=ROSE, color=LIGHT, size=17)

# ============ Slide 3: 我们的答案 ============
s = new()
header(s, "BeautyProof 是什么", "一句话定位")
textbox(s, Inches(0.6), Inches(2.0), Inches(12.1), Inches(1.2),
        "面向美妆垂域的「多模态取证台」——一张图（或图文）进去，一份普通人看得懂的信任结论出来。",
        size=20, bold=True, color=INK)
bullets(s, Inches(0.6), Inches(3.4), Inches(7.6), Inches(3.4), [
    "不是黑箱打分：8 件工具各出证据 → 规则引擎定四档 → 大模型翻译成人话。",
    "纯 CPU 可跑、开源、可复现，已上线在线 Demo。",
    "覆盖「被局部改过」与「整张 AI 画的」两类风险，互补不打架。",
])
callout(s, Inches(8.4), Inches(3.5), Inches(4.35), Inches(1.7),
        "定位 = 美妆界的「内容体检中心」\n每张图都能拿到一份带证据的体检报告。",
        fill=PANEL, barcolor=GOLD, size=16)

# ============ Slide 4: 赛题四件事 ============
s = new()
header(s, "赛题要的四件事，我们逐条交卷", "任务验收口径 vs 实现")
add_table(s, [
    ["赛题硬性要求", "BeautyProof 实现"],
    ["多模态检测（图 + 文并列）", "8 工具证据链：看图 6 件 + 读字 2 件"],
    ["可解释判定", "四段式人话报告 + LLM 解释层（过护栏）"],
    ["智能体 Agent", "planner 波次调度，每条决策留人话理由"],
    ["数据集展示闭环", "batch 评测：受控 100% / 真实误报 1.4% 实证"],
], Inches(0.6), Inches(2.0), Inches(12.1), Inches(4.3), body_size=15, head_size=15)

# ============ Slide 5: 架构流 ============
s = new()
header(s, "系统架构：证据链 → 规则 → 人话", "一张图看懂全流程")
labels = ["输入\n图 / 文", "8 取证工具", "统一证据\n格式", "规则引擎\n四档判定", "人话报告\n(+LLM/圆桌)"]
caps   = ["图片或图文帖", "hash/c2pa/ela/\nocr/aigc/trufor", "{tool,observed,\ncannot_prove,ev}", "high/suspicious/\ncredible/inconclusive", "普通人看得懂\n的信任结论"]
bw, bh, top = Inches(2.18), Inches(1.25), Inches(3.1)
left0 = Inches(0.55); gap = Inches(0.28)
for i, (lb, cp) in enumerate(zip(labels, caps)):
    l = left0 + i * (bw + gap)
    rect(s, l, top, bw, bh, PANEL2 if i % 2 == 0 else RGBColor(0x2A,0x2A,0x33), line=GOLD, line_w=Pt(1.25))
    textbox(s, l, top + Inches(0.12), bw, Inches(0.7), lb, size=15, bold=True,
            color=LIGHT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, l, top + bh + Inches(0.08), bw, Inches(0.8), cp, size=11,
            color=MUTE, align=PP_ALIGN.CENTER)
    if i < 4:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, l + bw + Inches(0.01),
                                top + Inches(0.42), gap - Inches(0.02), Inches(0.4))
        ar.fill.solid(); ar.fill.fore_color.rgb = GOLD; ar.line.fill.background(); ar.shadow.inherit = False
callout(s, Inches(0.55), Inches(5.5), Inches(12.2), Inches(1.2),
        "关键设计：所有工具共享同一套证据格式 → 可组合、可审计；规则引擎是「法官」，LLM 只是「翻译官」（绝不下结论）。",
        fill=PANEL, barcolor=GOLD, size=15)

# ============ Slide 6: 八工具 ============
s = new()
header(s, "八工具证据链（统一证据格式）", "看图 6 件 + 读字 2 件")
bullets(s, Inches(0.6), Inches(2.0), Inches(6.2), Inches(4.4), [
    "hash — 图片指纹（是否被复制搬运）",
    "c2pa — 出生证（是否带可信出处元数据）",
    "ela — 篡改噪点（局部是否被PS）",
    "ocr — 抄字（图上文字提取）",
    "aigc — 整图 AI·本域微调（是否 AI 生成）",
    "trufor — 局部篡改定位热力图",
], size=15)
bullets(s, Inches(7.0), Inches(2.0), Inches(5.7), Inches(2.4), [
    "text — 文案分析（宣称是否夸张/违规）",
    "crossmodal — 图文是否对得上（跨模态交叉）",
    "统一格式 {tool, observed, cannot_prove, evidence[]}",
], size=15)
callout(s, Inches(7.0), Inches(4.5), Inches(5.7), Inches(1.8),
        "互补口径：TruFor 管「被局部改过」，AIGC 管「整张 AI 画的」——\n两者互补，不是替代（R5 规则专门处理）。",
        fill=PANEL2, barcolor=ROSE, color=LIGHT, size=15)

# ============ Slide 7: 可解释 Agent ============
s = new()
header(s, "可解释 Agent：planner 是灵魂", "直接命中赛题「可解释 + Agent」")
bullets(s, Inches(0.6), Inches(2.0), Inches(7.5), Inches(4.4), [
    "波次调度：快检(哈希/c2pa) → 中检(ela/ocr) → 深检(aigc/trufor)。",
    "R1–R5 跳过规则：如 PNG 跳过 ELA、已知原图跳过重复查——每条跳过都写「人话理由」。",
    "A1–A3 后置动作：命中后自动补查，不留死角。",
    "决策可审计：评委能看清「下一步查什么、为什么查」。",
])
callout(s, Inches(8.3), Inches(2.2), Inches(4.45), Inches(2.6),
        "Agent 不是装饰：\n它决定「下一步查什么」，并解释「为什么这么查」——这正是赛题「可解释」的题眼。",
        fill=PANEL, barcolor=GOLD, size=16)

# ============ Slide 8: 本域微调翻车故事 ============
s = new()
header(s, "本域微调：把「不准」变成「准」", "翻车 → 修复（评委最爱看的对比）")
add_table(s, [
    ["阶段", "发生了什么", "结果"],
    ["起", "通用 AIGC 检测器在中文美妆域失效", "真实精修图 66% 误判为 AI"],
    ["承", "74 张真实精修图 + 20 张即梦图本域微调 MobileNetV3", "训练集内饱和到 0/1"],
    ["转", "重训 v2 + 置信度校准 + 弃权阈值", "真实误报 66.2% → 1.4%"],
    ["合", "AI 召回保持 100%（即梦域），v1 留作兜底", "「针对美妆定制」的硬证据"],
], Inches(0.6), Inches(2.0), Inches(12.1), Inches(3.6), body_size=14, head_size=14)
callout(s, Inches(0.6), Inches(5.85), Inches(12.1), Inches(0.95),
        "一句话：别人用通用模型，我们为美妆重训了一把——这就是「场景定制」的差异化杀招。",
        fill=PANEL, barcolor=GOLD, size=15)

# ============ Slide 9: 双保险 ============
s = new()
header(s, "双保险：多 Agent 圆桌 + LLM 解释层", "方案 B + 方案 C 都已落地")
bullets(s, Inches(0.6), Inches(2.0), Inches(7.6), Inches(4.2), [
    "方案 C · LLM 解释层：本地 Ollama + Qwen3-VL-4B 生成「人话解读」，接入 pipeline --llm。",
    "方案 B · 多 Agent 圆桌：ImageAgent / TextAgent / SourceAgent 结构化交叉复核 + JudgeAgent 共识。",
    "护栏 validator 六道关：解释层只翻译、绝不自己下结论；越界措辞一律拦截。",
    "两者均可选接入，不拖累核心流水线，失败优雅降级。",
])
callout(s, Inches(8.4), Inches(2.2), Inches(4.35), Inches(2.2),
        "大模型 = 翻译官\n规则引擎 = 法官\n分数非法律结论，高风险必人工复核。",
        fill=PANEL2, barcolor=ROSE, color=LIGHT, size=16)

# ============ Slide 10: 数据集闭环 ============
s = new()
header(s, "数据集闭环 & 实测结果（诚实版）", "敢把短板写进报告，才是可信度来源")
add_table(s, [
    ["评测指标", "实测数值"],
    ["受控评测集（35 张）", "100% 正确"],
    ["真实美妆图误报", "66.2% → 1.4%（v2 重训后）"],
    ["AI 召回（即梦域）", "100%"],
    ["跨生成器召回（MJ/SD/Flux）", "25%（诚实标注，v3 在补）"],
    ["弃权率（置信度校准）", "1.8%"],
], Inches(0.6), Inches(2.0), Inches(12.1), Inches(4.0), body_size=15, head_size=15)

# ============ Slide 11: 合规开源 ============
s = new()
header(s, "合规与开源（资格前提已消除）", "原创性 / IP / 数据红线")
bullets(s, Inches(0.6), Inches(2.0), Inches(12.0), Inches(4.6), [
    "仓库已转 public —— 评委 / 举办方可见，满足「被评审 / 被开源」前提。",
    "Apache-2.0 LICENSE + NOTICE：声明 TruFor(CVPR2023) / PyTorch / timm / PaddleOCR / Qwen3-VL 等第三方许可与归属。",
    "数据红线：训练 / 评测 = 自产（ImageGen）+ 用户压缩包，未抓任何第三方平台图；GenImage 仅离线参考、不入仓。",
    "模型权重：TruFor / AIGC 权重走现场脚本下载，不塞仓库（GitHub 单文件 100MB 上限）。",
])

# ============ Slide 12: 5+5 映射 ============
s = new()
header(s, "对齐 Brandstorm 官方 5+5 评审准则", "L'Oréal 评委最终打分口径（共 50 分）")
add_table(s, [
    ["维度", "含义", "BeautyProof 对应", "自评"],
    ["项目·INNOVATIVE", "前人未有的方案", "首个美妆垂域取证台 + b+c 混合架构", "4/5"],
    ["项目·SUSTAINABLE", "长期责任", "全开源、可复现、数据集闭环可持续", "5/5"],
    ["项目·INCLUSIVE", "不排斥群体", "保护易被误导的消费者；多形态零门槛演示", "4/5"],
    ["项目·FEASIBLE", "现实可落地", "纯 CPU 可跑、线上 Demo HTTP 200、三形态交付", "5/5"],
    ["项目·SCALABLE", "大规模可行", "API/Web/闭环骨架具备；跨生成器待 v3 校准", "3–4/5"],
    ["团队·JUDGMENT", "复杂决策", "planner R1–R5 + 误报危机果断重训", "4/5"],
    ["团队·RESILIENCE", "克服困难", "66%→1.4%；SSH 失效改 HTTPS 绕通；补 LICENSE", "5/5"],
    ["团队·AMBITION", "愿景", "美妆信任基础设施，而非一次性 detector", "4/5"],
    ["团队·EMPATHY", "团队支持", "跨角色协作（owner/AI/队友）；待口述补实", "3–4/5"],
    ["团队·LEARNING AGILITY", "学陌生领域", "零技术背景 → 主导完整取证系统；快速试错", "5/5"],
    ["合计", "", "预计", "≈43–47/50"],
], Inches(0.45), Inches(1.95), Inches(12.45), Inches(4.9), body_size=12.5, head_size=12.5)
textbox(s, Inches(0.45), Inches(6.95), Inches(12.4), Inches(0.4),
        "注：天池技术赛道「多模态/可解释/Agent/数据集闭环」四项全中；5+5 是 L'Oréal 评委最终打分口径，答辩须双轨覆盖。",
        size=12, color=MUTE)

# ============ Slide 13: 团队故事 ============
s = new()
header(s, "团队：从零基础到完整系统", "5+5 里最该讲透的「团队维度」")
bullets(s, Inches(0.6), Inches(2.0), Inches(12.0), Inches(4.6), [
    "LEARNING AGILITY：owner 为准大一零技术背景，从 CV/ML/Agent 零基础主导整套取证系统；快速试错（通用模型失效 → 本域微调）。",
    "RESILIENCE：误报危机果断重训；SSH 推送通道被代理劫持 → 改 HTTPS+token 绕通；TruFor 许可证灰区 → 补 LICENSE/NOTICE。",
    "JUDGMENT：planner 的 R1–R5 是「复杂局面下怎么查」的工程化判断，每条留人话理由。",
    "协作：owner 定业务方向，AI 工程执行，队友 liying856 协助——角色互补。",
])
callout(s, Inches(0.6), Inches(6.0), Inches(12.1), Inches(0.95),
        "我们不讳言起点低——但 8 周把「看不懂」变成了「能拿去答辩」。",
        fill=PANEL2, barcolor=ROSE, color=LIGHT, size=16)

# ============ Slide 14: 边界 & 下一步 ============
s = new()
header(s, "我们诚实声明的边界 & 下一步", "不回避短板，才有可信度")
bullets(s, Inches(0.6), Inches(2.0), Inches(7.6), Inches(4.4), [
    "边界：跨生成器（非即梦）召回仅 25%；AIGC 整图判定在重度滤镜真图上需谨慎。",
    "分数非法律结论，高风险必人工复核。",
    "下一步：① v3 跨生成器重训拉起召回；② 置信度校准 + 弃权阈值完善；③ 数据集扩到 200+ 张。",
])
callout(s, Inches(8.3), Inches(2.2), Inches(4.45), Inches(2.6),
        "答辩主张：\nTruFor 篡改定位（鲁棒）+ 图文交叉验证 是主防线；\n整图 AI 判定主动说「校准中 + 高风险人工复核」。",
        fill=PANEL, barcolor=GOLD, size=15)

# ============ Slide 15: 结尾 ============
s = new(); bg(s, DARK)
rect(s, 0, Inches(3.4), SW, Inches(0.06), GOLD)
textbox(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(1.1),
        "让每一张美妆图，都经得起追问", size=34, bold=True, color=LIGHT)
textbox(s, Inches(0.92), Inches(3.7), Inches(11.5), Inches(0.6),
        "BeautyProof — 美妆内容信任守护师", size=20, color=ROSE)
textbox(s, Inches(0.92), Inches(5.4), Inches(11.5), Inches(0.5),
        "谢谢 · 欢迎评委实测我们的在线 Demo（https://beautyproof-demo.app.workbuddy.host/）", size=15, color=MUTE)

import os
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "BeautyProof_答辩PPT.pptx")
prs.save(out)
print("SAVED", out, "slides:", len(prs.slides._sldIdLst))

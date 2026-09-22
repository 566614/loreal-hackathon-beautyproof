#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BeautyProof 鉴定报告导出工具：把 Markdown 报告渲染成带中文排版的可存档 PDF。

用法：
    python tools/export_pdf.py tampered_02_splice     # 单份：reports/report_<stem>.md -> .pdf
    python tools/export_pdf.py --all                  # 批量：reports/ 下所有 report_*.md

只新增本文件，不改动任何已有文件。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PREFIX = "report_"

# 每条对外输出都必须带的边界声明
DISCLAIMER = "本报告由算法自动生成，不构成法律意义上的鉴定结论。"


class ExportError(Exception):
    """可直接打印给用户的中文错误。"""


# reportlab 属于可选依赖，延迟到使用时再校验，这样 --help 仍然可用
try:
    from reportlab.lib import colors as _rl_colors  # noqa: F401
    _REPORTLAB_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    _REPORTLAB_IMPORT_ERROR = exc


def _check_reportlab():
    if _REPORTLAB_IMPORT_ERROR is not None:
        raise ExportError(
            "缺少 PDF 依赖库 reportlab，无法导出。\n"
            "请先安装：python -m pip install reportlab\n"
            f"（原始错误：{_REPORTLAB_IMPORT_ERROR}）"
        )


# --------------------------------------------------------------------------
# 字体
# --------------------------------------------------------------------------

# 依次尝试，第一个能注册成功的字体族胜出。
# role -> (路径, subfontIndex)；subfontIndex 为 None 表示普通单字形字体。
FONT_CANDIDATES = [
    ("MicrosoftYaHei", [
        ("normal", "C:/Windows/Fonts/msyh.ttc", 0),
        ("bold", "C:/Windows/Fonts/msyhbd.ttc", 0),
    ]),
    ("DengXian", [
        ("normal", "C:/Windows/Fonts/Deng.ttf", None),
        ("bold", "C:/Windows/Fonts/Dengb.ttf", None),
    ]),
    ("SimHei", [
        ("normal", "C:/Windows/Fonts/simhei.ttf", None),
        ("bold", "C:/Windows/Fonts/simhei.ttf", None),  # 黑体只有常规字重
    ]),
    ("SimSun", [
        ("normal", "C:/Windows/Fonts/simsun.ttc", 0),
    ]),
    ("KaiTi", [
        ("normal", "C:/Windows/Fonts/simkai.ttf", None),
    ]),
]

# 等宽字体只用于纯 ASCII 的代码片段（路径、指纹等）
MONO_CANDIDATES = [
    ("C:/Windows/Fonts/consola.ttf", None),
    ("C:/Windows/Fonts/cour.ttf", None),
]


def _load_ttf(ttf_cls, internal_name, path, subfont_index):
    kwargs = {} if subfont_index is None else {"subfontIndex": subfont_index}
    return ttf_cls(internal_name, str(path), **kwargs)


def _register_family(name, variants):
    """尝试注册一个字体族，返回 (是否成功, 失败原因)。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    missing = [p for _, p, _ in variants if not Path(p).is_file()]
    if missing:
        return False, "{}：字体文件不存在（{}）".format(name, "、".join(missing))

    registered = {}
    for role, path, index in variants:
        # 关键：reportlab 要求「字体族名」本身是一个已注册字体的名字，
        # 否则 Paragraph 用该名字排版时会报 “Can't map determine family/bold/italic”。
        internal = name if role == "normal" else "{}-{}".format(name, role)
        try:
            pdfmetrics.registerFont(_load_ttf(TTFont, internal, Path(path), index))
        except Exception as exc:  # noqa: BLE001 - 字体损坏/不受支持时要给出可读原因
            return False, "{}：加载失败（{}: {}）".format(name, exc.__class__.__name__, exc)
        registered[role] = internal

    normal = registered["normal"]
    bold = registered.get("bold", normal)
    italic = registered.get("italic", normal)
    bold_italic = registered.get("boldItalic", bold)
    try:
        pdfmetrics.registerFontFamily(
            name, normal=normal, bold=bold, italic=italic, boldItalic=bold_italic
        )
    except Exception as exc:  # noqa: BLE001
        return False, "{}：字体族注册失败（{}: {}）".format(name, exc.__class__.__name__, exc)
    return True, None


def _register_mono():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path, index in MONO_CANDIDATES:
        if not Path(path).is_file():
            continue
        try:
            pdfmetrics.registerFont(_load_ttf(TTFont, "BeautyProofMono", Path(path), index))
            return "BeautyProofMono"
        except Exception:  # noqa: BLE001
            continue
    return "Courier"  # reportlab 内置字体，仅含 ASCII，够用


def register_fonts():
    """返回 (中文字体族名, 等宽字体名, 未采用的前序字体及其原因)。"""
    failures = []
    for name, variants in FONT_CANDIDATES:
        ok, reason = _register_family(name, variants)
        if ok:
            return name, _register_mono(), failures
        failures.append(reason)

    raise ExportError(
        "中文字体全部注册失败，无法生成 PDF（缺字体的话 PDF 里会是一片空白方块）。\n"
        "已尝试的字体及失败原因：\n  - " + "\n  - ".join(failures) + "\n"
        "解决办法：确认机器上有 C:/Windows/Fonts/msyh.ttc 或 simhei.ttf；"
        "若在 Linux / macOS 上运行，请自备一款中文 TTF 并加到 "
        "tools/export_pdf.py 的 FONT_CANDIDATES 中。"
    )


# --------------------------------------------------------------------------
# Markdown 解析
# --------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")


def _is_cjk(ch):
    code = ord(ch)
    return (
        0x2E80 <= code <= 0x9FFF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def _join_soft(lines):
    """Markdown 软换行：中文之间直接拼接，其余情况补一个空格。"""
    out = ""
    for piece in lines:
        piece = piece.strip()
        if not piece:
            continue
        if not out or not _is_cjk(out[-1]) or not _is_cjk(piece[0]):
            out = piece if not out else out + " " + piece
        else:
            out = out + piece
    return out


def parse_blocks(text):
    """把 Markdown 文本切成区块：('h', level, text) / ('p', text) /
    ('li', indent, text) / ('quote', [段落]) / ('hr',) 。"""
    blocks = []
    para, quote, quote_cur = [], [], []

    def flush_para():
        if para:
            blocks.append(("p", _join_soft(para)))
            para.clear()

    def flush_quote():
        if quote_cur:
            quote.append(_join_soft(quote_cur))
            quote_cur.clear()
        if quote:
            blocks.append(("quote", list(quote)))
            quote.clear()

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            flush_para()
            flush_quote()
            continue

        matched = _HEADING_RE.match(line)
        if matched:
            flush_para()
            flush_quote()
            blocks.append(("h", len(matched.group(1)), matched.group(2).strip()))
            continue

        if _HR_RE.match(line):
            flush_para()
            flush_quote()
            blocks.append(("hr",))
            continue

        matched = _QUOTE_RE.match(line)
        if matched:
            flush_para()
            body = matched.group(1).strip()
            if body:
                quote_cur.append(body)
            elif quote_cur:
                quote.append(_join_soft(quote_cur))
                quote_cur.clear()
            continue

        matched = _BULLET_RE.match(line)
        if matched:
            flush_para()
            flush_quote()
            indent_level = min(len(matched.group(1).replace("\t", "  ")) // 2, 2)
            blocks.append(("li", indent_level, matched.group(2).strip()))
            continue

        flush_quote()
        para.append(line.strip())

    flush_para()
    flush_quote()
    return blocks


# --------------------------------------------------------------------------
# 行内标记转换（-> reportlab 支持的段落标记）
# --------------------------------------------------------------------------

def _escape_xml(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _code_span(code, mono_font, body_font):
    # 含中文的代码片段不能用内置 Courier（会显示成方块），回退到中文字体
    face = mono_font if all(ord(ch) < 128 for ch in code) else body_font
    return '<font face="{}" size="9.5" color="#B45309">{}</font>'.format(face, code)


def make_inline_renderer(mono_font, body_font):
    def render(text):
        text = _escape_xml(text)
        text = re.sub(
            r"`([^`\n]+)`",
            lambda m: _code_span(m.group(1), mono_font, body_font),
            text,
        )
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*([^*\n]+?)\*", r"<i>\1</i>", text)
        return text

    return render


# --------------------------------------------------------------------------
# 样式
# --------------------------------------------------------------------------

def make_styles(font):
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle

    ink = "#1F2937"
    styles = {
        "h1": ParagraphStyle(
            "H1", fontName=font, fontSize=19, leading=26, alignment=TA_CENTER,
            textColor="#111827", wordWrap="CJK", spaceBefore=0, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2", fontName=font, fontSize=14, leading=20, textColor="#1D4ED8",
            wordWrap="CJK", spaceBefore=14, spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "H3", fontName=font, fontSize=11.5, leading=17, textColor="#374151",
            wordWrap="CJK", spaceBefore=9, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body", fontName=font, fontSize=10.5, leading=17.5, textColor=ink,
            wordWrap="CJK", spaceBefore=1, spaceAfter=6,
        ),
        "quote": ParagraphStyle(
            "Quote", fontName=font, fontSize=10.5, leading=17, textColor="#374151",
            wordWrap="CJK", spaceBefore=0, spaceAfter=0,
        ),
    }
    bullet_texts = ("\u2022", "\u25E6", "\u25AA")  # • ◦ ▪
    for level in range(3):
        styles["bullet%d" % level] = ParagraphStyle(
            "Bullet%d" % level, fontName=font, fontSize=10.5, leading=17,
            textColor=ink, wordWrap="CJK",
            leftIndent=16 + level * 18, bulletIndent=2 + level * 18,
            bulletFontName=font, bulletFontSize=10.5, bulletText=bullet_texts[level],
            spaceBefore=1, spaceAfter=3,
        )
    return styles


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------

def build_story(blocks, styles, avail_width, render_inline):
    from reportlab.platypus import HRFlowable, Paragraph, Spacer, Table, TableStyle

    story = []
    for index, block in enumerate(blocks):
        kind = block[0]

        if kind == "h":
            level = block[1]
            style_key = {1: "h1", 2: "h2"}.get(level, "h3")
            story.append(Paragraph(render_inline(block[2]), styles[style_key]))
            if index == 0 and level == 1:  # 封面标题下加一道分隔线
                story.append(HRFlowable(
                    width="100%", thickness=1.2, spaceBefore=4, spaceAfter=14,
                    color="#2563EB", hAlign="CENTER",
                ))
            continue

        if kind == "hr":
            story.append(Spacer(1, 8))
            story.append(HRFlowable(
                width="100%", thickness=0.6, spaceBefore=0, spaceAfter=8,
                color="#D1D5DB", hAlign="CENTER",
            ))
            continue

        if kind == "p":
            story.append(Paragraph(render_inline(block[1]), styles["body"]))
            continue

        if kind == "li":
            level = block[1]
            style = styles["bullet%d" % level]
            story.append(Paragraph(render_inline(block[2]), style, bulletText=style.bulletText))
            continue

        if kind == "quote":
            cell = []
            for para_index, text in enumerate(block[1]):
                if para_index:
                    cell.append(Spacer(1, 5))
                cell.append(Paragraph(render_inline(text), styles["quote"]))
            # 注意必须包两层：Table([[cell]]) 才是「一行一列、单元格内多个 flowable」；
            # 写成 Table([cell]) 会被当成一行多列，导致引用块溢出页面。
            table = Table([[cell]], colWidths=[avail_width])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), "#F3F4F6"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LINEBEFORE", (0, 0), (0, -1), 3, "#9CA3AF"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(Spacer(1, 4))
            story.append(table)
            story.append(Spacer(1, 8))

    return story


def make_footer(font):
    from reportlab.lib import colors

    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#9CA3AF"))
        canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
        canvas.setLineWidth(0.5)
        y = 34
        canvas.line(doc.leftMargin, y + 11, doc.pagesize[0] - doc.rightMargin, y + 11)
        canvas.drawString(doc.leftMargin, y, DISCLAIMER)
        canvas.drawRightString(
            doc.pagesize[0] - doc.rightMargin, y, "第 {} 页".format(canvas.getPageNumber())
        )
        canvas.restoreState()

    return draw


def read_report(md_path):
    try:
        raw = md_path.read_bytes()
    except OSError as exc:
        raise ExportError("读取失败：{}（{}）".format(md_path.name, exc.strerror or exc))

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExportError("无法解码 {}：已尝试 UTF-8 / GB18030 均失败".format(md_path.name))


def export_one(md_path, pdf_path, body_font, mono_font):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate

    text = read_report(md_path)
    if not text.strip():
        raise ExportError("{} 内容为空，没有可导出的正文".format(md_path.name))

    blocks = parse_blocks(text)
    if not blocks:
        raise ExportError("{} 里没有解析出任何内容块".format(md_path.name))

    title = blocks[0][2] if blocks and blocks[0][0] == "h" else md_path.stem
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=55, rightMargin=55, topMargin=52, bottomMargin=58,
        title=title,
        author="BeautyProof",
        subject="BeautyProof 图片鉴定报告",
    )
    styles = make_styles(body_font)
    story = build_story(blocks, styles, doc.width, make_inline_renderer(mono_font, body_font))
    try:
        doc.build(story, onFirstPage=make_footer(body_font), onLaterPages=make_footer(body_font))
    except Exception as exc:  # noqa: BLE001
        raise ExportError(
            "生成 PDF 失败：{}（{}: {}）".format(md_path.name, exc.__class__.__name__, exc)
        )
    return pdf_path


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------

def list_stems():
    if not REPORTS_DIR.is_dir():
        return []
    stems = []
    for path in REPORTS_DIR.glob(REPORT_PREFIX + "*.md"):
        stems.append(path.name[len(REPORT_PREFIX):-len(".md")])
    return sorted(stems)


def normalize_stem(raw):
    """允许传入 tampered_02_splice / report_tampered_02_splice / 带 .md 后缀等写法。"""
    stem = Path(raw.strip()).name
    if stem.endswith(".md"):
        stem = stem[:-len(".md")]
    if stem.startswith(REPORT_PREFIX):
        stem = stem[len(REPORT_PREFIX):]
    return stem


def format_size(num_bytes):
    return "{:.1f} KB".format(num_bytes / 1024.0)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="把 reports/ 下的 Markdown 鉴定报告导出为中文 PDF",
        epilog="示例：python tools/export_pdf.py tampered_02_splice | python tools/export_pdf.py --all",
    )
    parser.add_argument("stem", nargs="?", help="报告标识，对应 reports/report_<stem>.md")
    parser.add_argument("--all", action="store_true", help="导出 reports/ 下所有 report_*.md")
    args = parser.parse_args(argv)

    if not REPORTS_DIR.is_dir():
        print("[失败] 找不到报告目录：{}".format(REPORTS_DIR))
        return 1

    if not args.stem and not args.all:
        stems = list_stems()
        print("请指定报告标识，或用 --all 全部导出。")
        if stems:
            print("当前可用的报告：")
            for stem in stems:
                print("  - {}".format(stem))
        else:
            print("目录里没有 report_*.md：{}".format(REPORTS_DIR))
        return 1

    if args.all:
        stems = list_stems()
        if not stems:
            print("[失败] 没有找到任何 report_*.md：{}".format(REPORTS_DIR))
            return 1
    else:
        stem = normalize_stem(args.stem)
        md_path = REPORTS_DIR / (REPORT_PREFIX + stem + ".md")
        if not md_path.is_file():
            print("[失败] 找不到报告文件：{}".format(md_path))
            candidates = list_stems()
            if candidates:
                print("可用的报告标识：{}".format("、".join(candidates)))
            return 1
        stems = [stem]

    try:
        _check_reportlab()
        body_font, mono_font, font_notes = register_fonts()
    except ExportError as exc:
        print("[失败] {}".format(exc))
        return 1
    for note in font_notes:
        print("[提示] 字体不可用，已跳过：{}".format(note))

    ok, failed = [], []
    for stem in stems:
        md_path = REPORTS_DIR / (REPORT_PREFIX + stem + ".md")
        pdf_path = REPORTS_DIR / (REPORT_PREFIX + stem + ".pdf")
        try:
            export_one(md_path, pdf_path, body_font, mono_font)
        except ExportError as exc:
            print("[失败] {}".format(exc))
            failed.append(stem)
            continue
        except Exception as exc:  # noqa: BLE001 - 兜底，避免裸 traceback
            print("[失败] {}：未预期的错误（{}: {}）".format(stem, exc.__class__.__name__, exc))
            failed.append(stem)
            continue
        size = pdf_path.stat().st_size if pdf_path.is_file() else 0
        print("[完成] report_{}.md -> report_{}.pdf ({})".format(
            stem, stem, format_size(size)))
        ok.append(stem)

    print("---- 导出结束：成功 {} 份，失败 {} 份 ----".format(len(ok), len(failed)))
    if failed:
        print("失败清单：{}".format("、".join(failed)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

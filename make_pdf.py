"""
make_pdf.py — Convert PROJECT_REPORT.md to a styled PDF using reportlab.
Usage: python make_pdf.py
Output: PROJECT_REPORT.pdf
"""

import re
import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Preformatted, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
except ImportError:
    print("Installing reportlab...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "-q"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Preformatted, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ── Colours ──────────────────────────────────────────────────────────────────
BLUE_DARK   = colors.HexColor("#1a237e")
BLUE_MID    = colors.HexColor("#283593")
BLUE_LIGHT  = colors.HexColor("#3949ab")
GREY_BG     = colors.HexColor("#f5f5f5")
GREY_BORDER = colors.HexColor("#cccccc")
CODE_BG     = colors.HexColor("#f0f0f0")
WHITE       = colors.white
BLACK       = colors.HexColor("#212121")
GREEN       = colors.HexColor("#2e7d32")

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def make_styles():
    return {
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=22,
                             textColor=WHITE, backColor=BLUE_DARK,
                             spaceAfter=6, spaceBefore=0,
                             leftIndent=-18, rightIndent=-18,
                             borderPadding=(8, 18, 8, 18),
                             leading=28),

        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=16,
                             textColor=WHITE, backColor=BLUE_MID,
                             spaceAfter=4, spaceBefore=14,
                             leftIndent=-18, rightIndent=-18,
                             borderPadding=(6, 18, 6, 18),
                             leading=20),

        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=13,
                             textColor=BLUE_DARK,
                             spaceAfter=3, spaceBefore=10,
                             leading=16),

        "h4": ParagraphStyle("h4", fontName="Helvetica-BoldOblique", fontSize=11,
                             textColor=BLUE_LIGHT,
                             spaceAfter=2, spaceBefore=6,
                             leading=14),

        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                               textColor=BLACK, spaceAfter=4,
                               leading=14),

        "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=10,
                                 textColor=BLACK, spaceAfter=2,
                                 leftIndent=16, bulletIndent=4,
                                 leading=13),

        "bullet2": ParagraphStyle("bullet2", fontName="Helvetica", fontSize=10,
                                  textColor=BLACK, spaceAfter=2,
                                  leftIndent=32, bulletIndent=20,
                                  leading=13),

        "code_inline": ParagraphStyle("code_inline", fontName="Courier",
                                      fontSize=9, textColor=BLACK,
                                      backColor=CODE_BG, leading=12),

        "toc_title": ParagraphStyle("toc_title", fontName="Helvetica-Bold",
                                    fontSize=14, textColor=BLUE_DARK,
                                    spaceAfter=6, spaceBefore=8),

        "toc_item": ParagraphStyle("toc_item", fontName="Helvetica", fontSize=10,
                                   textColor=BLACK, spaceAfter=2, leftIndent=10),
    }


def clean(text):
    """Remove/replace markdown syntax and fix chars for ReportLab."""
    # Replace unicode arrows and special chars
    text = text.replace("→", "->").replace("►", "->").replace("•", "-")
    text = text.replace("─", "-").replace("━", "-").replace("│", "|")
    text = text.replace("—", "--").replace("–", "-")
    text = text.replace("≥", ">=").replace("≤", "<=").replace("×", "x")
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="9">\1</font>', text)
    # Links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Escaped chars
    text = text.replace("\\*", "*").replace("\\_", "_")
    return text


def parse_table(lines):
    """Parse markdown table lines into a list of row lists."""
    rows = []
    for line in lines:
        if re.match(r'\s*\|[-: |]+\|\s*$', line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build_table_flowable(rows, styles):
    if not rows:
        return None
    col_count = max(len(r) for r in rows)
    # Pad rows to same length
    padded = [r + [""] * (col_count - len(r)) for r in rows]

    # Convert cells to Paragraphs
    def cell(text, bold=False):
        text = clean(text)
        fn = "Helvetica-Bold" if bold else "Helvetica"
        style = ParagraphStyle("cell", fontName=fn, fontSize=9,
                               textColor=BLACK, leading=12)
        return Paragraph(text, style)

    table_data = []
    for i, row in enumerate(padded):
        table_data.append([cell(c, bold=(i == 0)) for c in row])

    page_width = A4[0] - 3.6 * cm
    col_width = page_width / col_count

    t = Table(table_data, colWidths=[col_width] * col_count,
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("BACKGROUND",   (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def md_to_flowables(md_text, styles):
    flowables = []
    lines = md_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        raw = line.rstrip()

        # ── H1 ───────────────────────────────────────────────────────────────
        if raw.startswith("# ") and not raw.startswith("## "):
            text = clean(raw[2:])
            flowables.append(Spacer(1, 0.3 * cm))
            flowables.append(Paragraph(text, styles["h1"]))
            flowables.append(Spacer(1, 0.2 * cm))
            i += 1

        # ── H2 ───────────────────────────────────────────────────────────────
        elif raw.startswith("## "):
            text = clean(raw[3:])
            flowables.append(Spacer(1, 0.3 * cm))
            flowables.append(Paragraph(text, styles["h2"]))
            flowables.append(Spacer(1, 0.1 * cm))
            i += 1

        # ── H3 ───────────────────────────────────────────────────────────────
        elif raw.startswith("### "):
            text = clean(raw[4:])
            flowables.append(Spacer(1, 0.2 * cm))
            flowables.append(Paragraph(text, styles["h3"]))
            i += 1

        # ── H4 ───────────────────────────────────────────────────────────────
        elif raw.startswith("#### "):
            text = clean(raw[5:])
            flowables.append(Paragraph(text, styles["h4"]))
            i += 1

        # ── Horizontal rule ──────────────────────────────────────────────────
        elif re.match(r'^-{3,}$', raw) or re.match(r'^\*{3,}$', raw):
            flowables.append(HRFlowable(width="100%", thickness=1,
                                        color=BLUE_LIGHT, spaceAfter=4))
            i += 1

        # ── Code block (fenced) ──────────────────────────────────────────────
        elif raw.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].rstrip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = "\n".join(code_lines)
            # Replace special chars in code
            code_text = (code_text.replace("→", "->").replace("►", "->")
                         .replace("─", "-").replace("━", "-")
                         .replace("│", "|").replace("•", "-")
                         .replace("≥", ">=").replace("≤", "<=")
                         .replace("×", "x").replace("—", "--")
                         .replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;"))
            style = ParagraphStyle("code_block", fontName="Courier",
                                   fontSize=8, textColor=BLACK,
                                   backColor=CODE_BG, leading=11,
                                   leftIndent=8, rightIndent=8,
                                   spaceBefore=4, spaceAfter=6,
                                   borderColor=GREY_BORDER,
                                   borderWidth=0.5, borderPadding=6)
            flowables.append(Paragraph(code_text.replace("\n", "<br/>"), style))

        # ── Table ─────────────────────────────────────────────────────────────
        elif raw.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].rstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            tbl = build_table_flowable(rows, styles)
            if tbl:
                flowables.append(Spacer(1, 0.15 * cm))
                flowables.append(tbl)
                flowables.append(Spacer(1, 0.15 * cm))

        # ── Bullet list ──────────────────────────────────────────────────────
        elif re.match(r'^(\s*)-\s+', raw) or re.match(r'^(\s*)\*\s+', raw):
            indent = len(raw) - len(raw.lstrip())
            text = re.sub(r'^\s*[-*]\s+', '', raw)
            text = clean(text)
            style = styles["bullet2"] if indent >= 2 else styles["bullet"]
            flowables.append(Paragraph(f"&bull; {text}", style))
            i += 1

        # ── Numbered list ────────────────────────────────────────────────────
        elif re.match(r'^\d+\.\s+', raw):
            m = re.match(r'^(\d+)\.\s+(.*)', raw)
            if m:
                text = clean(m.group(2))
                flowables.append(
                    Paragraph(f"{m.group(1)}. {text}", styles["bullet"]))
            i += 1

        # ── Blockquote ───────────────────────────────────────────────────────
        elif raw.startswith("> "):
            text = clean(raw[2:])
            style = ParagraphStyle("quote", fontName="Helvetica-Oblique",
                                   fontSize=10, textColor=colors.HexColor("#555"),
                                   backColor=GREY_BG, leftIndent=12,
                                   borderColor=BLUE_LIGHT, borderWidth=3,
                                   borderPadding=(4, 8, 4, 8), leading=13)
            flowables.append(Paragraph(text, style))
            i += 1

        # ── Empty line ───────────────────────────────────────────────────────
        elif raw.strip() == "":
            flowables.append(Spacer(1, 0.1 * cm))
            i += 1

        # ── Regular paragraph ────────────────────────────────────────────────
        else:
            text = clean(raw)
            if text.strip():
                flowables.append(Paragraph(text, styles["body"]))
            i += 1

    return flowables


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.2 * cm,
                           f"Page {doc.page}")
    canvas.drawString(1.8 * cm, 1.2 * cm,
                      "Multi-Agent CAE Simulation & Compliance Platform")
    canvas.setStrokeColor(GREY_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(1.8 * cm, 1.5 * cm, A4[0] - 1.8 * cm, 1.5 * cm)
    canvas.restoreState()


def main():
    src = Path("PROJECT_REPORT.md")
    out = Path("PROJECT_REPORT.pdf")

    if not src.exists():
        print(f"ERROR: {src} not found. Run from the project root directory.")
        sys.exit(1)

    print(f"Reading {src} ...")
    md_text = src.read_text(encoding="utf-8")

    print("Building PDF ...")
    styles = make_styles()

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Multi-Agent CAE Simulation & Compliance Platform",
        author="Swapnil Kadlag",
        subject="Project Report",
    )

    flowables = md_to_flowables(md_text, styles)
    doc.build(flowables, onFirstPage=add_page_number,
              onLaterPages=add_page_number)

    print(f"\nDone! Saved to: {out.resolve()}")
    print(f"Pages: open the PDF to check page count")


if __name__ == "__main__":
    main()

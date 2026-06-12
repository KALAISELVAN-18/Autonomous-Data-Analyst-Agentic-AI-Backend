"""
Report Generation Agent
────────────────────────
Responsibilities:
  • Assemble all artifacts (stats, charts, insights, cleaning log) into a
    professional, multi-section PDF report using ReportLab.
  • Save the PDF to data/reports/<dataset_name>_report.pdf.
  • Store the final path in state["report_path"].

PDF Structure:
  Cover Page
  ├── 1. Executive Summary
  ├── 2. Dataset Overview & Data Quality
  ├── 3. Cleaning Report
  ├── 4. Statistical Analysis
  ├── 5. Visualizations
  ├── 6. Key Insights & Recommendations
  └── 7. Appendix — Column Metadata
"""

import os
import io
import base64
import textwrap
from datetime import datetime
from typing import Dict, Any, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from src.core.state import AnalystState
from src.core.config import settings

# ─── Colours ─────────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#1B2A4A")
BLUE      = colors.HexColor("#3A6FD8")
TEAL      = colors.HexColor("#2E9E8E")
LIGHT_BG  = colors.HexColor("#F4F6FB")
TEXT      = colors.HexColor("#2D2D2D")
MUTED     = colors.HexColor("#6B7280")
WHITE     = colors.white

# ─── Styles ───────────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title", parent=base["Title"],
        fontSize=28, textColor=WHITE, spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"],
        fontSize=13, textColor=colors.HexColor("#CBD5E1"),
        alignment=TA_CENTER, fontName="Helvetica",
    )
    styles["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"],
        fontSize=16, textColor=NAVY, spaceAfter=6, spaceBefore=14,
        fontName="Helvetica-Bold", borderPad=4,
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"],
        fontSize=12, textColor=BLUE, spaceAfter=4, spaceBefore=10,
        fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=10, textColor=TEXT, spaceAfter=4,
        fontName="Helvetica", leading=15, alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=base["Normal"],
        fontSize=10, textColor=TEXT, spaceAfter=3, leftIndent=16,
        fontName="Helvetica", bulletIndent=6,
    )
    styles["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"],
        fontSize=8, textColor=MUTED, spaceAfter=2,
        fontName="Helvetica-Oblique", alignment=TA_CENTER,
    )
    styles["code"] = ParagraphStyle(
        "code", parent=base["Code"],
        fontSize=8, textColor=TEXT, fontName="Courier",
        backColor=LIGHT_BG, borderPad=4,
    )
    return styles


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_image(b64_string: str, width: float = 14 * cm) -> Image | None:
    """Decode a Base64 PNG and return a ReportLab Image flowable."""
    try:
        data = base64.b64decode(b64_string)
        img_buf = io.BytesIO(data)
        img = Image(img_buf)
        aspect = img.imageHeight / float(img.imageWidth)
        img.drawWidth = width
        img.drawHeight = width * aspect
        return img
    except Exception as exc:
        print(f"[REPORTER] Failed to decode chart image: {exc}")
        return None


def _hr(color=BLUE) -> HRFlowable:
    return HRFlowable(width="100%", thickness=1, color=color, spaceAfter=6)


def _kv_table(data: List[List[str]], styles_obj) -> Table:
    """Render a two-column key-value table."""
    table = Table(data, colWidths=[5.5 * cm, 11 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), LIGHT_BG),
        ("TEXTCOLOR",   (0, 0), (-1, -1), TEXT),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


# ─── Section builders ────────────────────────────────────────────────────────

def _cover_page(elements: list, styles: dict, state: AnalystState):
    """Full-bleed cover page with title and metadata."""
    file_name = state.get("file_name", "Dataset")
    date_str  = datetime.now().strftime("%B %d, %Y")

    # We simulate a dark banner with a coloured table row
    cover_data = [[
        Paragraph(
            f"<br/><br/><br/>"
            f"<font size=30><b>Autonomous Data Analyst</b></font><br/><br/>"
            f"<font size=16 color='#CBD5E1'>Analysis Report</font><br/><br/>"
            f"<font size=12 color='#94A3B8'>Dataset: {file_name}</font><br/>"
            f"<font size=10 color='#94A3B8'>Generated: {date_str}</font><br/><br/><br/>",
            ParagraphStyle("cover_inner", alignment=TA_CENTER,
                           textColor=WHITE, fontName="Helvetica"),
        )
    ]]
    cover_table = Table(cover_data, colWidths=[17 * cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(cover_table)
    elements.append(PageBreak())


def _section_overview(elements: list, styles: dict, state: AnalystState, inv_mapping: dict):
    elements.append(Paragraph("1. Dataset Overview & Data Quality", styles["h1"]))
    elements.append(_hr())

    meta       = state.get("metadata", {})
    statistics = state.get("statistics", {})
    shape      = statistics.get("shape", meta.get("shape", {}))

    # Map columns back to original names for overview presentation
    num_cols = [inv_mapping.get(c, c) for c in meta.get("numeric_columns", [])]
    cat_cols = [inv_mapping.get(c, c) for c in meta.get("categorical_columns", [])]
    dt_cols  = [inv_mapping.get(c, c) for c in meta.get("datetime_columns", [])]

    rows = [
        ["Dataset Name",     str(meta.get("file_name", "N/A"))],
        ["Rows",             f"{shape.get('rows', 'N/A'):,}"],
        ["Columns",          str(shape.get('columns', 'N/A'))],
        ["Numeric Columns",  ", ".join(num_cols) or "None"],
        ["Categorical Cols", ", ".join(cat_cols) or "None"],
        ["Datetime Cols",    ", ".join(dt_cols) or "None"],
    ]

    # Missing value summary
    null_pcts = meta.get("null_percentages", {})
    if null_pcts:
        worst = sorted(null_pcts.items(), key=lambda x: x[1], reverse=True)[:5]
        worst_str = "; ".join([f"{inv_mapping.get(c, c)}: {v:.1f}%" for c, v in worst if v > 0]) or "No missing values"
        rows.append(["Top Missing Cols", worst_str])

    elements.append(_kv_table(rows, styles))
    elements.append(Spacer(1, 0.3 * cm))


def _section_cleaning(elements: list, styles: dict, state: AnalystState):
    elements.append(Paragraph("2. Data Cleaning Report", styles["h1"]))
    elements.append(_hr())
    elements.append(Paragraph(
        "The following cleaning operations were automatically applied to the raw dataset:",
        styles["body"]
    ))
    elements.append(Spacer(1, 0.2 * cm))

    log = state.get("cleaning_log", [])
    if not log:
        elements.append(Paragraph("No cleaning steps were recorded.", styles["body"]))
        return

    for i, entry in enumerate(log, 1):
        elements.append(Paragraph(f"• {entry}", styles["bullet"]))
    elements.append(Spacer(1, 0.3 * cm))


def _section_statistics(elements: list, styles: dict, state: AnalystState, inv_mapping: dict):
    elements.append(Paragraph("3. Statistical Analysis", styles["h1"]))
    elements.append(_hr())

    statistics = state.get("statistics", {})
    descriptive = statistics.get("descriptive", {})

    if not descriptive:
        elements.append(Paragraph("No numeric columns found for statistical analysis.", styles["body"]))
        return

    elements.append(Paragraph("Descriptive Statistics (Numeric Columns)", styles["h2"]))

    # Build statistics table
    headers = ["Column", "Count", "Mean", "Median", "Std Dev", "Min", "Max", "Skewness"]
    table_data = [headers]
    for col, s in descriptive.items():
        table_data.append([
            inv_mapping.get(col, col),
            f"{s.get('count', 0):,}",
            f"{s.get('mean', 0):.3f}",
            f"{s.get('median', 0):.3f}",
            f"{s.get('std', 0):.3f}",
            f"{s.get('min', 0):.3f}",
            f"{s.get('max', 0):.3f}",
            f"{s.get('skewness', 0):.2f}",
        ])

    col_widths = [3.5*cm, 1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 1.5*cm]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4 * cm))

    # Top correlations
    top_corrs = statistics.get("top_correlations", [])
    if top_corrs:
        elements.append(Paragraph("Top Feature Correlations", styles["h2"]))
        corr_data = [["Column A", "Column B", "Pearson r", "Strength"]]
        for pair in top_corrs[:8]:
            r = pair["correlation"]
            if abs(r) >= 0.7:
                strength = "Strong"
            elif abs(r) >= 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
            col_a = inv_mapping.get(pair["col_a"], pair["col_a"])
            col_b = inv_mapping.get(pair["col_b"], pair["col_b"])
            corr_data.append([col_a, col_b, f"{r:.4f}", strength])

        t2 = Table(corr_data, colWidths=[4*cm, 4*cm, 3*cm, 3*cm], repeatRows=1)
        t2.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
            ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.3 * cm))


def _section_visualizations(elements: list, styles: dict, state: AnalystState, inv_mapping: dict):
    elements.append(Paragraph("4. Visualizations", styles["h1"]))
    elements.append(_hr())

    charts = state.get("visualizations", {})
    chart_titles = {
        "missing_heatmap":        "Missing Value Rate by Column",
        "numeric_distributions":  "Numeric Column Distributions",
        "correlation_heatmap":    "Pearson Correlation Heatmap",
        "boxplots":               "Box Plots — Spread & Outliers",
        "top_correlations_bar":   "Top Feature Correlations",
    }

    if not charts:
        elements.append(Paragraph("No visualizations were generated.", styles["body"]))
        return

    for key, b64 in charts.items():
        if not b64:
            continue
        title = chart_titles.get(key)
        if not title:
            if key.startswith("bar_"):
                col_name = key[4:]
                orig_col = inv_mapping.get(col_name, col_name.replace('_', ' ').title())
                title = f"Value Distribution — {orig_col}"
            else:
                title = key.replace("_", " ").title()

        elements.append(Paragraph(title, styles["h2"]))
        img = _b64_to_image(b64, width=15 * cm)
        if img:
            elements.append(img)
        elements.append(Paragraph(f"Figure: {title}", styles["caption"]))
        elements.append(Spacer(1, 0.4 * cm))


def _section_insights(elements: list, styles: dict, state: AnalystState):
    elements.append(Paragraph("5. Key Insights & Recommendations", styles["h1"]))
    elements.append(_hr())

    insights_md = state.get("insights", "")
    recommendations = state.get("recommendations", [])

    if not insights_md:
        elements.append(Paragraph("No insights were generated.", styles["body"]))
        return

    # Render insights markdown line-by-line
    for line in insights_md.splitlines():
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 0.15 * cm))
        elif line.startswith("## "):
            elements.append(Paragraph(line[3:], styles["h2"]))
        elif line.startswith("# "):
            elements.append(Paragraph(line[2:], styles["h1"]))
        elif line.startswith("- ") or line.startswith("* "):
            elements.append(Paragraph(f"• {line[2:]}", styles["bullet"]))
        elif line.startswith("**") and line.endswith("**"):
            elements.append(Paragraph(f"<b>{line[2:-2]}</b>", styles["body"]))
        else:
            # Escape special chars for ReportLab
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            elements.append(Paragraph(safe, styles["body"]))

    elements.append(Spacer(1, 0.3 * cm))

    if recommendations:
        elements.append(Paragraph("Quick Reference — Recommendations", styles["h2"]))
        for rec in recommendations:
            elements.append(Paragraph(f"✔  {rec}", styles["bullet"]))


def _section_appendix(elements: list, styles: dict, state: AnalystState, inv_mapping: dict):
    elements.append(PageBreak())
    elements.append(Paragraph("Appendix — Column Metadata", styles["h1"]))
    elements.append(_hr())

    meta = state.get("metadata", {})
    dtypes    = meta.get("dtypes", {})
    nulls     = meta.get("null_percentages", {})
    uniques   = meta.get("unique_counts", {})
    samples   = meta.get("sample_values", {})

    cols = list(dtypes.keys())
    if not cols:
        elements.append(Paragraph("No column metadata available.", styles["body"]))
        return

    headers = ["Column", "Data Type", "Missing %", "Unique Values", "Sample Values"]
    data = [headers]
    for col in cols:
        sample_vals = samples.get(col, [])
        sample_str = ", ".join([str(v) for v in sample_vals[:3]])
        data.append([
            inv_mapping.get(col, col),
            str(dtypes.get(col, "N/A")),
            f"{nulls.get(col, 0):.1f}%",
            str(uniques.get(col, "N/A")),
            sample_str[:40],
        ])

    col_widths = [4*cm, 2.5*cm, 2.5*cm, 3*cm, 5.5*cm]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP",      (0, 0), (-1, -1), True),
    ]))
    elements.append(t)


# ─── Main node ────────────────────────────────────────────────────────────────

def reporter_node(state: AnalystState) -> Dict[str, Any]:
    """
    Report Generation Node.

    Builds a structured, professional PDF from the full AnalystState
    and saves it to disk. Stores the output path in state["report_path"].
    """
    print("\n========== REPORTER AGENT ==========")

    os.makedirs(settings.report_dir, exist_ok=True)

    file_name = state.get("file_name", "dataset")
    clean_name = os.path.splitext(file_name)[0].replace(" ", "_")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        settings.report_dir, f"{clean_name}_report_{timestamp}.pdf"
    )

    # Build flowables
    styles   = _build_styles()
    elements = []

    meta = state.get("metadata", {})
    mapping = meta.get("column_mapping", {})
    inv_mapping = {v: k for k, v in mapping.items()} if mapping else {}

    _cover_page(elements, styles, state)
    _section_overview(elements, styles, state, inv_mapping)
    elements.append(PageBreak())
    _section_cleaning(elements, styles, state)
    elements.append(PageBreak())
    _section_statistics(elements, styles, state, inv_mapping)
    elements.append(PageBreak())
    _section_visualizations(elements, styles, state, inv_mapping)
    elements.append(PageBreak())
    _section_insights(elements, styles, state)
    _section_appendix(elements, styles, state, inv_mapping)

    # Page number footer
    def _add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(
            A4[0] - 1.5 * cm, 1 * cm,
            f"Page {doc.page}  |  Autonomous Data Analyst  |  {datetime.now().strftime('%Y-%m-%d')}"
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2 * cm,
    )

    doc.build(elements, onFirstPage=_add_footer, onLaterPages=_add_footer)

    print(f"[REPORTER] PDF saved -> {output_path}")
    return {"report_path": output_path, "error_count": 0}

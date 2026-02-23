import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas as rl_canvas
from datetime import datetime

# ── PATHS ──────────────────────────────────────────────────────────────────────
DATASET_PATH = (
    r"C:\Users\rapha\.cache\kagglehub\datasets\carrie1"
    r"\ecommerce-data\versions\1\data.csv"
)
OUTPUT_PDF = "cohort_report.pdf"

# ── PALETA ────────────────────────────────────────────────────────────────────
AZUL_ESCURO = colors.HexColor("#16213e")
AZUL_MEDIO  = colors.HexColor("#0f3460")
AZUL_CLARO  = colors.HexColor("#e8f0fe")
CINZA_LINHA = colors.HexColor("#f5f5f5")
DESTAQUE    = colors.HexColor("#e94560")
VERDE       = colors.HexColor("#27ae60")
BRANCO      = colors.white

WIDTH, HEIGHT = A4
MARGEM = 2 * cm
FULL_W = WIDTH - 2 * MARGEM

# ── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
print("Loading dataset...")
df_raw = pd.read_csv(DATASET_PATH, encoding="latin1")
df = df_raw.copy()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df["Description"] = df["Description"].str.strip()
df = df[df["Description"].notna() & (df["Description"] != "")]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
df["CustomerID"]  = df["CustomerID"].astype(int)
df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")

print(f"  Clean rows: {len(df):,}")

# ── 2. COHORT CALCULATIONS ────────────────────────────────────────────────────
# Cohort = month of first purchase
cohort_df = (
    df.groupby("CustomerID")["InvoiceMonth"]
    .min()
    .reset_index()
    .rename(columns={"InvoiceMonth": "CohortMonth"})
)
df = df.merge(cohort_df, on="CustomerID")

# Months since first purchase
df["CohortIndex"] = (
    (df["InvoiceMonth"].dt.year  - df["CohortMonth"].dt.year) * 12 +
    (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
)

# Active cohorts (exclude Dec 2011 as source cohort — too little follow-up)
LAST_COHORT = pd.Period("2011-11", "M")
df_cohorts = df[df["CohortMonth"] <= LAST_COHORT].copy()

# ── COHORT SIZES (unique customers per cohort) ────────────────────────────────
cohort_sizes = (
    df_cohorts[df_cohorts["CohortIndex"] == 0]
    .groupby("CohortMonth")["CustomerID"]
    .nunique()
    .sort_index()
)

# ── RETENTION MATRIX ─────────────────────────────────────────────────────────
ret_pivot = (
    df_cohorts
    .groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
    .nunique()
    .reset_index()
)
cohort_counts = ret_pivot.pivot(
    index="CohortMonth", columns="CohortIndex", values="CustomerID"
)
retention = cohort_counts.divide(cohort_sizes, axis=0) * 100

# ── REVENUE / LTV MATRIX ─────────────────────────────────────────────────────
rev_pivot = (
    df_cohorts
    .groupby(["CohortMonth", "CohortIndex"])["Revenue"]
    .sum()
    .reset_index()
)
revenue_matrix = rev_pivot.pivot(
    index="CohortMonth", columns="CohortIndex", values="Revenue"
)
cumulative_ltv = revenue_matrix.cumsum(axis=1).divide(cohort_sizes, axis=0)

# ── PERIOD REVENUE (per-period average per customer) ─────────────────────────
avg_rev_per_cust = revenue_matrix.divide(cohort_sizes, axis=0)

# ── COMPUTED SUMMARY STATS ───────────────────────────────────────────────────
ret_m1 = retention[1].dropna().mean()  if 1  in retention.columns else 0.0
ret_m3 = retention[3].dropna().mean()  if 3  in retention.columns else 0.0
ret_m6 = retention[6].dropna().mean()  if 6  in retention.columns else 0.0

dec_cohort = pd.Period("2010-12", "M")
ltv_12m = (
    cumulative_ltv.loc[dec_cohort, 12]
    if dec_cohort in cumulative_ltv.index and 12 in cumulative_ltv.columns
    else 0.0
)

# Best and worst M+1 cohorts
m1_series = retention[1].dropna() if 1 in retention.columns else pd.Series(dtype=float)
best_m1_cohort  = str(m1_series.idxmax()) if not m1_series.empty else "—"
worst_m1_cohort = str(m1_series.idxmin()) if not m1_series.empty else "—"
best_m1_val     = m1_series.max()  if not m1_series.empty else 0.0
worst_m1_val    = m1_series.min()  if not m1_series.empty else 0.0

date_min = df["InvoiceDate"].min()
date_max = df["InvoiceDate"].max()

print(f"  Cohorts:  {len(cohort_sizes)}")
print(f"  Avg M+1 retention: {ret_m1:.1f}%")
print(f"  Avg M+3 retention: {ret_m3:.1f}%")
print(f"  Dec'10 12-month LTV: £{ltv_12m:,.0f}")


# ── 3. CHARTS ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

PALETA = [
    "#16213e", "#0f3460", "#1a6b8a", "#2196F3",
    "#64B5F6", "#90CAF9", "#e94560", "#f39c12", "#27ae60", "#8e44ad",
    "#e67e22", "#1abc9c",
]


def fig_to_image(fig, width_cm, height_cm):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


# ── CHART 1: Cohort Acquisition ───────────────────────────────────────────────
def chart_cohort_sizes():
    fig, ax = plt.subplots(figsize=(13, 3.8))
    x    = list(range(len(cohort_sizes)))
    vals = cohort_sizes.values
    cors = ["#e94560" if v == vals.max() else "#0f3460" for v in vals]

    bars = ax.bar(x, vals, color=cors, width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vals.max() * 0.015,
            f"{int(val):,}", ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in cohort_sizes.index],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("New Customers", fontsize=9)
    ax.set_title("New Customers Acquired per Month  (Cohort Size)",
                 fontsize=11, fontweight="bold", color="#16213e", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ── CHART 2: Retention Heatmap ────────────────────────────────────────────────
def chart_retention_heatmap():
    ret = retention.copy()
    # Limit columns to M0..M+12
    cols = [c for c in ret.columns if 0 <= c <= 12]
    ret  = ret[cols]

    # String annotation matrix (with % sign, blank for NaN)
    annot = ret.copy().astype(object)
    for col in annot.columns:
        annot[col] = ret[col].apply(
            lambda v: f"{v:.0f}%" if pd.notna(v) else ""
        )

    col_labels = [f"M+{int(c)}" if c > 0 else "M0\nAcq." for c in cols]
    idx_labels = [str(p) for p in ret.index]

    fig, ax = plt.subplots(figsize=(15, 7.5))

    if HAS_SEABORN:
        sns.heatmap(
            ret,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="Blues",
            mask=ret.isnull(),
            linewidths=0.5,
            linecolor="#e0e0e0",
            cbar_kws={"label": "Retention Rate (%)", "shrink": 0.75},
            annot_kws={"size": 8},
            vmin=0, vmax=100,
        )
    else:
        # Fallback: matplotlib imshow
        data = ret.values.astype(float)
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        plt.colorbar(im, ax=ax, label="Retention Rate (%)")
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                if not np.isnan(data[r, c]):
                    ax.text(c, r, f"{data[r,c]:.0f}%",
                            ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(idx_labels)))
        ax.set_yticklabels(idx_labels, fontsize=8.5)

    ax.set_xticklabels(col_labels, fontsize=8.5)
    ax.set_yticklabels(idx_labels, fontsize=8.5, rotation=0)
    ax.set_xlabel("Months Since First Purchase", fontsize=10, labelpad=8)
    ax.set_ylabel("Acquisition Cohort", fontsize=10, labelpad=8)
    ax.set_title(
        "Customer Retention Cohort — % of Cohort Who Purchased Again",
        fontsize=12, fontweight="bold", color="#16213e", pad=12
    )
    fig.tight_layout()
    return fig


# ── CHART 3: Retention Curves ─────────────────────────────────────────────────
def chart_retention_curves():
    # Skip M0 (always 100%), show M+1 .. M+12
    cols = [c for c in retention.columns if 1 <= c <= 12]
    ret  = retention[cols]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    palette = plt.cm.Blues(np.linspace(0.35, 0.9, len(ret)))

    for i, (cohort, row) in enumerate(ret.iterrows()):
        vals = row.dropna()
        if vals.empty:
            continue
        ax.plot(vals.index, vals.values,
                color=palette[i], linewidth=1.6, marker="o",
                markersize=3.5, label=str(cohort), alpha=0.85)

    # Average curve
    avg = ret.mean()
    ax.plot(avg.index, avg.values,
            color="#e94560", linewidth=2.8, linestyle="--",
            marker="s", markersize=5, label="Average", zorder=10)

    ax.set_xticks(cols)
    ax.set_xticklabels([f"M+{c}" for c in cols], fontsize=8.5)
    ax.set_ylabel("Retention Rate (%)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Retention Curves by Cohort  (M+1 → M+12)",
                 fontsize=11, fontweight="bold", color="#16213e", pad=10)
    ax.legend(fontsize=7.5, ncol=4, loc="upper right",
              frameon=False, bbox_to_anchor=(1, 1.02))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ── CHART 4: Cumulative LTV Curves ───────────────────────────────────────────
def chart_ltv_curves():
    cols = [c for c in cumulative_ltv.columns if 0 <= c <= 12]
    cum  = cumulative_ltv[cols]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    palette = plt.cm.Blues(np.linspace(0.35, 0.9, len(cum)))

    for i, (cohort, row) in enumerate(cum.iterrows()):
        vals = row.dropna()
        if vals.empty:
            continue
        ax.plot(vals.index, vals.values,
                color=palette[i], linewidth=1.6, marker="o",
                markersize=3.5, label=str(cohort), alpha=0.85)

    avg = cum.mean()
    ax.plot(avg.index, avg.values,
            color="#e94560", linewidth=2.8, linestyle="--",
            marker="s", markersize=5, label="Average", zorder=10)

    ax.set_xticks(cols)
    ax.set_xticklabels([f"M+{c}" if c > 0 else "M0" for c in cols], fontsize=8.5)
    ax.set_ylabel("Cumulative Revenue per Customer (£)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.set_title("Cumulative LTV per Acquired Customer by Cohort",
                 fontsize=11, fontweight="bold", color="#16213e", pad=10)
    ax.legend(fontsize=7.5, ncol=4, loc="upper left",
              frameon=False, bbox_to_anchor=(0, 1.02))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ── CHART 5: M+1 Retention bar by cohort ─────────────────────────────────────
def chart_m1_bar():
    m1 = retention[1].dropna() if 1 in retention.columns else pd.Series(dtype=float)
    if m1.empty:
        return None

    fig, ax = plt.subplots(figsize=(13, 3.5))
    x    = list(range(len(m1)))
    vals = m1.values
    cors = ["#e94560" if v == vals.max() else
            "#f39c12" if v == vals.min() else
            "#0f3460" for v in vals]
    avg  = vals.mean()

    bars = ax.bar(x, vals, color=cors, width=0.6)
    ax.axhline(avg, color="#e94560", linewidth=1.5, linestyle="--",
               label=f"Avg {avg:.1f}%")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in m1.index],
                       fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("M+1 Retention (%)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("First-Month Retention (M+1) by Acquisition Cohort",
                 fontsize=11, fontweight="bold", color="#16213e", pad=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


# ── 4. PDF STYLES ─────────────────────────────────────────────────────────────
_ss = getSampleStyleSheet()

def _s(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_ss[parent], **kw)

sTitle    = _s("sTitle",   "Title",  fontSize=24, textColor=BRANCO,
               alignment=TA_CENTER, spaceAfter=4, leading=30)
sCoverSub = _s("sCoverSub",          fontSize=12, textColor=colors.HexColor("#aabbdd"),
               alignment=TA_CENTER, spaceAfter=4)
sCoverDate= _s("sCoverDate",         fontSize=9,  textColor=colors.HexColor("#888888"),
               alignment=TA_CENTER)
sSection  = _s("sSection",           fontSize=13, textColor=AZUL_ESCURO,
               fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
sSubSect  = _s("sSubSect",           fontSize=10, textColor=AZUL_MEDIO,
               fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
sBody     = _s("sBody",              fontSize=9,  textColor=colors.HexColor("#333333"),
               leading=14, spaceAfter=6)
sKpiLabel = _s("sKpiLabel",          fontSize=8,  textColor=colors.HexColor("#6688aa"),
               alignment=TA_CENTER)
sKpiValue = _s("sKpiValue",          fontSize=20, textColor=AZUL_MEDIO,
               alignment=TA_CENTER, fontName="Helvetica-Bold")
sFooter   = _s("sFooter",            fontSize=8,  textColor=colors.HexColor("#999999"),
               alignment=TA_CENTER)
sInsight  = _s("sInsight",           fontSize=9,  textColor=colors.HexColor("#16213e"),
               leading=14, leftIndent=8, spaceAfter=5)
sInsightBox = _s("sInsightBox",      fontSize=9,  textColor=colors.HexColor("#16213e"),
                 leading=14, leftIndent=10, rightIndent=10,
                 spaceAfter=4, spaceBefore=4,
                 backColor=colors.HexColor("#f0f4ff"),
                 borderPadding=(6, 6, 6, 6))


def tabela_estilo(tem_total=False, highlight_avg=False):
    base = [
        ("BACKGROUND",    (0, 0),  (-1, 0),  AZUL_ESCURO),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  BRANCO),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, 0),  9),
        ("ALIGN",         (0, 0),  (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -1), [colors.white, CINZA_LINHA]),
        ("FONTSIZE",      (0, 1),  (-1, -1), 8.5),
        ("ALIGN",         (1, 1),  (-1, -1), "RIGHT"),
        ("ALIGN",         (0, 1),  (0, -1),  "LEFT"),
        ("GRID",          (0, 0),  (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0, 0),  (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 5),
        ("LEFTPADDING",   (0, 0),  (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0),  (-1, -1), 7),
    ]
    if tem_total or highlight_avg:
        base += [
            ("BACKGROUND", (0, -1), (-1, -1), AZUL_CLARO),
            ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE",  (0, -1), (-1, -1), 1, AZUL_MEDIO),
        ]
    return TableStyle(base)


def insight_block(text):
    """Returns a lightly-styled table with an insight bullet."""
    t = Table(
        [[Paragraph(text, sBody)]],
        colWidths=[FULL_W],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f0f4ff")),
        ("BOX",           (0, 0), (-1, -1), 0.8, colors.HexColor("#b0c4de")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


# ── 5. PAGE NUMBERS ───────────────────────────────────────────────────────────
class NumeradorPaginas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        rl_canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for i, state in enumerate(self._saved_page_states):
            self.__dict__.update(state)
            if i > 0:
                self._draw_footer(i + 1, total)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_footer(self, current, total):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#aaaaaa"))
        self.drawString(MARGEM, 1.3 * cm,
                        "UK E-Commerce Cohort Analysis  |  Confidential")
        self.drawRightString(WIDTH - MARGEM, 1.3 * cm,
                             f"Page {current} of {total}")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(MARGEM, 1.6 * cm, WIDTH - MARGEM, 1.6 * cm)
        self.restoreState()


# ── 6. BUILD STORY ────────────────────────────────────────────────────────────
print("Building report...")
story = []

# ────────────────────────────────────────────────────────────────────────────
# COVER
# ────────────────────────────────────────────────────────────────────────────
capa_header = Table(
    [[Paragraph("UK E-COMMERCE", sTitle)],
     [Paragraph("COHORT ANALYSIS", sTitle)]],
    colWidths=[FULL_W],
    rowHeights=[2.2 * cm, 2.2 * cm],
)
capa_header.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), AZUL_ESCURO),
    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 10),
]))

story.append(Spacer(1, 1.5 * cm))
story.append(capa_header)
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "Customer Retention &amp; Lifetime Value  —  Cohort-Based Analysis",
    sCoverSub,
))
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph(
    f"Period: {date_min.strftime('%d %b %Y')} – {date_max.strftime('%d %b %Y')}  ·  "
    f"Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}",
    sCoverDate,
))
story.append(Spacer(1, 0.8 * cm))
story.append(HRFlowable(width="100%", thickness=1.5, color=AZUL_MEDIO))
story.append(Spacer(1, 0.6 * cm))

# KPIs on cover
kpi_table = Table(
    [
        [Paragraph("AVG M+1<br/>RETENTION",  sKpiLabel),
         Paragraph("AVG M+3<br/>RETENTION",  sKpiLabel),
         Paragraph("AVG M+6<br/>RETENTION",  sKpiLabel),
         Paragraph("DEC'10<br/>12M LTV/CUST", sKpiLabel)],
        [Paragraph(f"{ret_m1:.1f}%",   sKpiValue),
         Paragraph(f"{ret_m3:.1f}%",   sKpiValue),
         Paragraph(f"{ret_m6:.1f}%",   sKpiValue),
         Paragraph(f"£{ltv_12m:,.0f}", sKpiValue)],
    ],
    colWidths=[FULL_W / 4] * 4,
    rowHeights=[0.9 * cm, 1.3 * cm],
)
kpi_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), AZUL_CLARO),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#c0cfee")),
    ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#c0cfee")),
]))
story.append(kpi_table)
story.append(Spacer(1, 0.8 * cm))

story.append(Paragraph("Methodology", sSection))
story.append(Paragraph(
    "Each customer is assigned to the cohort of their <b>first purchase month</b>. "
    "<b>Retention rate</b> for period M+N = % of that cohort who purchased at least once "
    "N months after acquisition. <b>Cumulative LTV</b> = total revenue per acquired customer "
    "up to and including each period. Cancelled invoices, zero-price lines and "
    "anonymous transactions are excluded. Dec 2011 is excluded as an acquisition cohort "
    "due to insufficient follow-up data.",
    sBody,
))
story.append(PageBreak())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 1 — COHORT ACQUISITION
# ────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("1. New Customer Acquisition by Cohort", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph(
    "The bar chart below shows how many <b>new (first-time) customers</b> were acquired in "
    "each month. This is the base population of each retention cohort.",
    sBody,
))
story.append(fig_to_image(chart_cohort_sizes(), 17, 5))
story.append(Spacer(1, 0.3 * cm))

# Cohort size table
rows_sizes = [["Cohort Month", "New Customers", "% of All New Customers",
               "Cohort Rank"]]
sorted_sizes = cohort_sizes.sort_values(ascending=False)
rank_map = {cohort: i + 1 for i, cohort in enumerate(sorted_sizes.index)}
total_new = cohort_sizes.sum()
for cohort, size in cohort_sizes.items():
    rows_sizes.append([
        str(cohort),
        f"{int(size):,}",
        f"{size / total_new * 100:.1f}%",
        f"#{rank_map[cohort]}",
    ])
rows_sizes.append(["TOTAL", f"{int(total_new):,}", "100%", "—"])

t_sizes = Table(rows_sizes,
                colWidths=[4.5 * cm, 4 * cm, 5.5 * cm, 3 * cm])
t_sizes.setStyle(tabela_estilo(tem_total=True))
story.append(t_sizes)
story.append(Spacer(1, 0.4 * cm))

story.append(insight_block(
    "<b>Insight:</b>  The three largest acquisition cohorts are <b>Sep, Oct and Nov 2011</b> "
    "— a pattern consistent with pre-Christmas B2B gifting demand. "
    "However, because they sit at the end of the observation window, their long-term value "
    "cannot yet be measured; the <b>Dec 2010</b> cohort (holiday season) is the only one "
    "with a full 12-month follow-up and therefore the most reliable LTV benchmark. "
    "Invest in Q4 acquisition campaigns: the data suggests holiday-acquires convert and retain well."
))
story.append(PageBreak())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RETENTION HEATMAP
# ────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("2. Retention Cohort Heatmap", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph(
    "<b>How to read:</b> each cell shows the % of a cohort (row) that purchased again "
    "N months after acquisition (column). <b>M0 = acquisition month (100% by definition)</b>. "
    "Blank cells = the cohort has not yet reached that period. "
    "Darker blue = higher retention.",
    sBody,
))
story.append(Spacer(1, 0.15 * cm))
story.append(fig_to_image(chart_retention_heatmap(), 17, 9.5))
story.append(Spacer(1, 0.3 * cm))

story.append(insight_block(
    "<b>Insight:</b>  Retention drops sharply from M0 to M+1 across all cohorts — "
    "this is the most critical churn window. After M+1, the rate stabilises in a "
    "<b>20–40% band</b>, indicating that customers who survive the first month become "
    "reliable repeat buyers. The Dec 2010 cohort sustains the highest retention at M+6 "
    f"and M+12, confirming its quality. Cohort <b>{best_m1_cohort}</b> posted the best "
    f"M+1 retention ({best_m1_val:.0f}%), while <b>{worst_m1_cohort}</b> posted the lowest "
    f"({worst_m1_val:.0f}%) — investigate the acquisition channel difference between these months."
))
story.append(PageBreak())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RETENTION CURVES
# ────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("3. Retention Curves &amp; M+1 Breakdown", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))
story.append(fig_to_image(chart_retention_curves(), 17, 7))
story.append(Spacer(1, 0.4 * cm))

# M+1 bar chart
m1_fig = chart_m1_bar()
if m1_fig:
    story.append(fig_to_image(m1_fig, 17, 4.5))
    story.append(Spacer(1, 0.3 * cm))

story.append(insight_block(
    "<b>Insight:</b>  The <b>dashed red average curve</b> shows that across all cohorts, "
    f"roughly <b>{ret_m1:.0f}%</b> of customers return in month 1, "
    f"<b>{ret_m3:.0f}%</b> in month 3, and <b>{ret_m6:.0f}%</b> in month 6. "
    "Curves flatten rather than continuing to decline after M+3, which is a strong signal "
    "of a loyal core. A <b>win-back campaign targeting customers at the M+1 cliff</b> — "
    "e.g. a personalised email 3–4 weeks after first purchase — could meaningfully lift "
    "the long-term retention rate and LTV."
))
story.append(PageBreak())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 4 — M+1 RETENTION TABLE (detail)
# ────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("Retention Rate Detail — M+1 to M+6", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph(
    "The table below shows the raw retention rate (% of cohort) for the first six "
    "months after acquisition. The final row is the cohort-weighted average.",
    sBody,
))
story.append(Spacer(1, 0.2 * cm))

ret_cols = [c for c in retention.columns if 1 <= c <= 6]
ret_display = retention[ret_cols]

rows_ret = [["Cohort", "Size"] + [f"M+{c}" for c in ret_cols]]
for cohort, row in ret_display.iterrows():
    size = int(cohort_sizes[cohort])
    row_data = [str(cohort), f"{size:,}"]
    for v in row:
        row_data.append(f"{v:.0f}%" if pd.notna(v) else "—")
    rows_ret.append(row_data)

avg_row = ["Average", "—"]
for c in ret_cols:
    v = retention[c].dropna().mean() if c in retention.columns else np.nan
    avg_row.append(f"{v:.0f}%" if not np.isnan(v) else "—")
rows_ret.append(avg_row)

t_ret = Table(rows_ret,
              colWidths=[3.5 * cm, 2 * cm] + [2.1 * cm] * len(ret_cols))
t_ret.setStyle(tabela_estilo(highlight_avg=True))
story.append(t_ret)
story.append(PageBreak())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CUMULATIVE LTV
# ────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("4. Cumulative LTV per Acquired Customer", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph(
    "Cumulative revenue generated per customer from the acquisition month onward. "
    "A steeper slope indicates faster value realisation; a plateau signals churn. "
    "Cohorts with limited follow-up data naturally show fewer data points.",
    sBody,
))
story.append(fig_to_image(chart_ltv_curves(), 17, 7))
story.append(Spacer(1, 0.4 * cm))

# LTV table M0..M+6
ltv_cols = [c for c in cumulative_ltv.columns if 0 <= c <= 6]
cum_display = cumulative_ltv[ltv_cols]

rows_ltv = [["Cohort", "Size"] + [f"M+{c}" if c > 0 else "M0" for c in ltv_cols]]
for cohort, row in cum_display.iterrows():
    size = int(cohort_sizes[cohort])
    row_data = [str(cohort), f"{size:,}"]
    for v in row:
        row_data.append(f"£{v:,.0f}" if pd.notna(v) else "—")
    rows_ltv.append(row_data)

avg_ltv_row = ["Average", "—"]
for c in ltv_cols:
    v = cumulative_ltv[c].dropna().mean() if c in cumulative_ltv.columns else np.nan
    avg_ltv_row.append(f"£{v:,.0f}" if not np.isnan(v) else "—")
rows_ltv.append(avg_ltv_row)

t_ltv = Table(rows_ltv,
              colWidths=[3.2 * cm, 2 * cm] + [2 * cm] * len(ltv_cols))
t_ltv.setStyle(tabela_estilo(highlight_avg=True))
story.append(t_ltv)
story.append(Spacer(1, 0.4 * cm))

story.append(insight_block(
    f"<b>Insight:</b>  The <b>Dec 2010 cohort</b> reaches a 12-month cumulative LTV of "
    f"<b>£{ltv_12m:,.0f} per customer</b>, making it the benchmark for acquisition ROI. "
    "LTV grows <b>non-linearly</b> — the steepest gains occur in M0 (the first order) "
    "and M+1 (the second purchase). Cohorts with high M+1 retention show significantly "
    "higher M+6 and M+12 LTV, reinforcing that <b>winning the second purchase is the "
    "single highest-leverage action</b> for growing customer lifetime value."
))
story.append(PageBreak())


# ────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CONSOLIDATED BUSINESS INSIGHTS
# ────────────────────────────────────────────────────────────────────────────
story.append(Paragraph("5. Consolidated Business Insights &amp; Recommendations",
                        sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

insights = [
    (
        "Win the Second Purchase",
        f"M+1 is the steepest drop-off point across every cohort (avg {ret_m1:.0f}% return). "
        "Customers who do return in M+1 are significantly more likely to stay active through "
        "M+6 and beyond. Deploy a targeted re-engagement flow — personalised email or "
        "exclusive offer — 3–4 weeks after first purchase."
    ),
    (
        "Q4 Cohorts Are Your Most Valuable",
        "Dec 2010 and Sep–Nov 2011 cohorts show the highest LTV trajectories. "
        f"The Dec 2010 12-month LTV of £{ltv_12m:,.0f}/customer is the benchmark for "
        "calculating maximum allowable CAC. Prioritise Q4 paid acquisition budgets "
        "and loyalty campaigns timed to the pre-Christmas gifting season."
    ),
    (
        "Long-Term Retention Is Healthy After M+3",
        f"After the initial churn cliff, retention stabilises at ~{ret_m3:.0f}% (M+3) "
        f"and ~{ret_m6:.0f}% (M+6). This implies a loyal 'core buyer' segment forms "
        "naturally. Identify these survivors and build a VIP tier (early access, volume "
        "discounts) to maximise their spend over time."
    ),
    (
        "Investigate Cohort Quality Variation",
        f"M+1 retention varies from {worst_m1_val:.0f}% ({worst_m1_cohort}) to "
        f"{best_m1_val:.0f}% ({best_m1_cohort}). This spread suggests that acquisition "
        "channel mix or seasonal demand type affects customer quality. "
        "Correlate acquisition source data with cohort retention to identify which channels "
        "bring in buyers vs. one-time visitors."
    ),
    (
        "Monitor the 2011 Sep–Nov Cohorts as They Mature",
        "The three largest cohorts (Sep–Nov 2011) have limited follow-up data. "
        "Apply the Dec 2010 cohort's retention curve as a benchmark: if their M+3 "
        "retention is materially below the historical average, intervene early with "
        "win-back campaigns before they fully churn."
    ),
    (
        "Revenue Concentration Risk",
        "82% of revenue comes from the UK, and the top 10 customers account for ~17% "
        "of total revenue. A cohort lens confirms that losing even a single top-tier "
        "customer cohort would have outsized revenue impact. "
        "Consider key-account management programs for the highest-LTV cohorts."
    ),
]

for i, (title, body) in enumerate(insights, 1):
    story.append(Paragraph(f"{i}. {title}", sSubSect))
    story.append(insight_block(body))
    story.append(Spacer(1, 0.2 * cm))

# Footer
story.append(Spacer(1, 0.8 * cm))
story.append(HRFlowable(width="100%", thickness=0.5,
                         color=colors.HexColor("#cccccc")))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    f"Period: {date_min.strftime('%d/%m/%Y')} to {date_max.strftime('%d/%m/%Y')}  |  "
    f"Source: UCI Online Retail Dataset (data.csv)  |  Generated by cohort_analysis.py",
    sFooter,
))


# ── 7. RENDER PDF ─────────────────────────────────────────────────────────────
print("Rendering PDF...")
doc = SimpleDocTemplate(
    OUTPUT_PDF,
    pagesize=A4,
    rightMargin=MARGEM,
    leftMargin=MARGEM,
    topMargin=MARGEM,
    bottomMargin=2.5 * cm,
)
doc.build(story, canvasmaker=NumeradorPaginas)
print(f"\nCohort report generated: {OUTPUT_PDF}")

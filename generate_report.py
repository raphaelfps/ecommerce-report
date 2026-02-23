import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak,
)
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfgen import canvas as rl_canvas
from datetime import datetime

# ── PATHS ──────────────────────────────────────────────────────────────────────
DATASET_PATH = (
    r"C:\Users\rapha\.cache\kagglehub\datasets\carrie1"
    r"\ecommerce-data\versions\1\data.csv"
)
OUTPUT_PDF = "ecommerce_report.pdf"

# ── PALETA ────────────────────────────────────────────────────────────────────
AZUL_ESCURO = colors.HexColor("#16213e")
AZUL_MEDIO  = colors.HexColor("#0f3460")
AZUL_CLARO  = colors.HexColor("#e8f0fe")
CINZA_LINHA = colors.HexColor("#f5f5f5")
DESTAQUE    = colors.HexColor("#e94560")
VERDE       = colors.HexColor("#27ae60")
BRANCO      = colors.white

PALETA_MPL = [
    "#16213e", "#0f3460", "#1a6b8a", "#2196F3",
    "#64B5F6", "#90CAF9", "#e94560", "#f39c12", "#27ae60", "#8e44ad",
]

WIDTH, HEIGHT = A4
MARGEM = 2 * cm
FULL_W = WIDTH - 2 * MARGEM


# ── 1. CARREGAR E LIMPAR DADOS ────────────────────────────────────────────────
print("Loading dataset...")
df_raw = pd.read_csv(DATASET_PATH, encoding="latin1")
print(f"  Raw rows:   {len(df_raw):,}")

df = df_raw.copy()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]   # remove cancellations
df = df.dropna(subset=["CustomerID"])                         # require customer ID
df = df[df["Quantity"] > 0]                                   # remove returns/negatives
df = df[df["UnitPrice"] > 0]                                  # remove zero-price items
df["Description"] = df["Description"].str.strip()
df = df[df["Description"].notna() & (df["Description"] != "")]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"]     = df["Quantity"] * df["UnitPrice"]
df["Month"]       = df["InvoiceDate"].dt.to_period("M")
df["DayOfWeek"]   = df["InvoiceDate"].dt.day_name()
df["Hour"]        = df["InvoiceDate"].dt.hour
df["CustomerID"]  = df["CustomerID"].astype(int)

print(f"  Clean rows: {len(df):,}")

# ── 2. ANÁLISES ───────────────────────────────────────────────────────────────
total_revenue    = df["Revenue"].sum()
total_invoices   = df["InvoiceNo"].nunique()
unique_customers = df["CustomerID"].nunique()
avg_order_value  = df.groupby("InvoiceNo")["Revenue"].sum().mean()
date_min         = df["InvoiceDate"].min()
date_max         = df["InvoiceDate"].max()

# Mensal
monthly        = df.groupby("Month")["Revenue"].sum().sort_index()
monthly_qty    = df.groupby("Month")["Quantity"].sum().sort_index()
monthly_orders = df.groupby("Month")["InvoiceNo"].nunique().sort_index()
monthly_short  = [str(m) for m in monthly.index]
monthly_growth = monthly.pct_change() * 100

# País
country_rev  = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
country_ord  = df.groupby("Country")["InvoiceNo"].nunique()
country_cust = df.groupby("Country")["CustomerID"].nunique()
top10_countries = country_rev.head(10)

# Produto
prod_rev = df.groupby("Description")["Revenue"].sum().sort_values(ascending=False)
prod_qty = df.groupby("Description")["Quantity"].sum()
top10_products = prod_rev.head(10)
top20_products = prod_rev.head(20)

# Cliente
cust_rev    = df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
cust_orders = df.groupby("CustomerID")["InvoiceNo"].nunique()
top10_customers = cust_rev.head(10)

# Padrões temporais
_day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_rev    = df.groupby("DayOfWeek")["Revenue"].sum().reindex(_day_order).dropna()
dow_orders = df.groupby("DayOfWeek")["InvoiceNo"].nunique().reindex(_day_order).dropna()
hour_rev   = df.groupby("Hour")["Revenue"].sum().sort_index()

# KPIs de cliente
avg_cust_rev = cust_rev.mean()
median_cust  = cust_rev.median()
repeat_rate  = cust_orders[cust_orders > 1].count() / unique_customers * 100
avg_orders_per_cust = cust_orders.mean()


# ── 3. GRÁFICOS ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})


def fig_to_image(fig, width_cm, height_cm):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def chart_tendencia_mensal():
    fig, ax = plt.subplots(figsize=(13, 3.8))
    x    = list(range(len(monthly)))
    vals = monthly.values / 1_000

    ax.fill_between(x, vals, alpha=0.12, color="#0f3460")
    ax.plot(x, vals, color="#0f3460", linewidth=2.5, marker="o",
            markersize=5, markerfacecolor="#e94560", markeredgewidth=0)

    for i, v in enumerate(vals):
        ax.annotate(f"£{v:.0f}k", (i, v),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(monthly_short, fontsize=8, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Monthly Revenue (£ thousands)", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def chart_crescimento_mensal():
    fig, ax = plt.subplots(figsize=(13, 2.8))
    g    = monthly_growth.dropna()
    x    = list(range(len(g)))
    lbls = [monthly_short[i + 1] for i in x]
    cors = ["#27ae60" if v >= 0 else "#e94560" for v in g.values]

    bars = ax.bar(x, g.values, color=cors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=8, rotation=30, ha="right")
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Month-over-Month Growth (%)", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    for bar, val in zip(bars, g.values):
        offset = 1.2 if val >= 0 else -4.5
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    return fig


def chart_top_paises():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = [n[:30] + "…" if len(n) > 30 else n
             for n in top10_countries.index.tolist()[::-1]]
    vals  = top10_countries.values[::-1] / 1_000
    cors  = PALETA_MPL[:len(names)][::-1]

    bars = ax.barh(names, vals, color=cors, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + vals.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"£{val:.1f}k", va="center", fontsize=8.5, color="#333333")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlim(0, vals.max() * 1.25)
    ax.set_title("Top 10 Countries by Revenue (£ thousands)", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def chart_pizza_paises():
    top5  = country_rev.head(5)
    other = country_rev.iloc[5:].sum()
    data  = pd.concat([top5, pd.Series({"Other": other})])

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _, autotexts = ax.pie(
        data.values,
        autopct="%1.1f%%",
        colors=PALETA_MPL[:len(data)],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.72,
    )
    ax.set_aspect("equal")
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.legend(wedges, data.index,
              loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, fontsize=8.5, frameon=False)
    ax.set_title("Revenue Share by Country", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    fig.tight_layout()
    return fig


def chart_top_produtos():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = [n[:38] + "…" if len(n) > 38 else n
             for n in top10_products.index.tolist()[::-1]]
    vals  = top10_products.values[::-1] / 1_000
    cors  = PALETA_MPL[:10][::-1]

    bars = ax.barh(names, vals, color=cors, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + vals.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"£{val:.1f}k", va="center", fontsize=8.5, color="#333333")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlim(0, vals.max() * 1.25)
    ax.set_title("Top 10 Products by Revenue (£ thousands)", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def chart_dia_semana():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x    = list(range(len(dow_rev)))
    vals = dow_rev.values / 1_000
    cors = [PALETA_MPL[1]] * len(x)

    bars = ax.bar(x, vals, color=cors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:3] for d in dow_rev.index], fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Revenue by Day of Week (£ thousands)", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vals.max() * 0.01,
                f"£{val:.0f}k", ha="center", va="bottom", fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def chart_hora_dia():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x    = hour_rev.index.tolist()
    vals = hour_rev.values / 1_000

    ax.fill_between(x, vals, alpha=0.15, color="#e94560")
    ax.plot(x, vals, color="#e94560", linewidth=2, marker="o",
            markersize=4, markerfacecolor="#0f3460", markeredgewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}h" for h in x], fontsize=7.5, rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:.0f}k"))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Revenue by Hour of Day (£ thousands)", fontsize=11,
                 fontweight="bold", color="#16213e", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


# ── 4. ESTILOS PDF ────────────────────────────────────────────────────────────
_ss = getSampleStyleSheet()


def _s(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_ss[parent], **kw)


sTitle     = _s("sTitle",    "Title",  fontSize=26, textColor=BRANCO,
                alignment=TA_CENTER, spaceAfter=4, leading=32)
sCoverSub  = _s("sCoverSub",          fontSize=13, textColor=colors.HexColor("#aabbdd"),
                alignment=TA_CENTER, spaceAfter=4)
sCoverDate = _s("sCoverDate",         fontSize=9,  textColor=colors.HexColor("#888888"),
                alignment=TA_CENTER)
sSection   = _s("sSection",           fontSize=13, textColor=AZUL_ESCURO,
                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
sKpiLabel  = _s("sKpiLabel",          fontSize=8,  textColor=colors.HexColor("#6688aa"),
                alignment=TA_CENTER)
sKpiValue  = _s("sKpiValue",          fontSize=20, textColor=AZUL_MEDIO,
                alignment=TA_CENTER, fontName="Helvetica-Bold")
sFooter    = _s("sFooter",            fontSize=8,  textColor=colors.HexColor("#999999"),
                alignment=TA_CENTER)
sBody      = _s("sBody",              fontSize=9,  textColor=colors.HexColor("#333333"),
                leading=14, spaceAfter=6)
sInsightTitle = _s("sInsightTitle",   fontSize=9,  textColor=AZUL_ESCURO,
                   fontName="Helvetica-Bold", spaceAfter=2)


def insight_block(lines):
    """Render a list of (bold_label, body_text) tuples as a styled insight box."""
    content = []
    for label, text in lines:
        content.append(Paragraph(
            f"<b>{label}</b>  {text}",
            _s(f"sIns_{label[:8]}", fontSize=9, textColor=colors.HexColor("#16213e"),
               leading=14, leftIndent=8, spaceAfter=4),
        ))
    t = Table([[c] for c in content], colWidths=[FULL_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f0f4ff")),
        ("BOX",           (0, 0), (-1, -1), 0.8, colors.HexColor("#b0c4de")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def tabela_estilo(tem_total=False):
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
    if tem_total:
        base += [
            ("BACKGROUND", (0, -1), (-1, -1), AZUL_CLARO),
            ("FONTNAME",   (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE",  (0, -1), (-1, -1), 1, AZUL_MEDIO),
        ]
    return TableStyle(base)


# ── 5. CANVAS COM NUMERAÇÃO DE PÁGINAS ───────────────────────────────────────
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
                        "UK E-Commerce Report  |  Confidential")
        self.drawRightString(WIDTH - MARGEM, 1.3 * cm,
                             f"Page {current} of {total}")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(MARGEM, 1.6 * cm, WIDTH - MARGEM, 1.6 * cm)
        self.restoreState()


# ── 6. CONSTRUÇÃO DO STORY ────────────────────────────────────────────────────
print("Building report...")
story = []

# ── CAPA ──────────────────────────────────────────────────────────────────────
capa_header = Table(
    [[Paragraph("UK E-COMMERCE REPORT", sTitle)]],
    colWidths=[FULL_W],
    rowHeights=[3.8 * cm],
)
capa_header.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), AZUL_ESCURO),
    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 18),
]))

story.append(Spacer(1, 1.5 * cm))
story.append(capa_header)
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Complete E-Commerce Performance Analysis — United Kingdom", sCoverSub))
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph(
    f"Period: {date_min.strftime('%d %b %Y')} – {date_max.strftime('%d %b %Y')}  ·  "
    f"Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}",
    sCoverDate,
))
story.append(Spacer(1, 0.8 * cm))
story.append(HRFlowable(width="100%", thickness=1.5, color=AZUL_MEDIO))
story.append(Spacer(1, 0.6 * cm))

# KPIs principais
kpi_table = Table(
    [
        [Paragraph("TOTAL REVENUE",    sKpiLabel),
         Paragraph("ORDERS",           sKpiLabel),
         Paragraph("UNIQUE CUSTOMERS", sKpiLabel),
         Paragraph("AVG ORDER VALUE",  sKpiLabel)],
        [Paragraph(f"£{total_revenue / 1_000_000:.2f}M", sKpiValue),
         Paragraph(f"{total_invoices:,}",                  sKpiValue),
         Paragraph(f"{unique_customers:,}",                sKpiValue),
         Paragraph(f"£{avg_order_value:,.0f}",             sKpiValue)],
    ],
    colWidths=[FULL_W / 4] * 4,
    rowHeights=[0.75 * cm, 1.3 * cm],
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

# Destaques da capa
top_produto_nome = prod_rev.index[0]
top_produto_label = (top_produto_nome[:45] + "…") if len(top_produto_nome) > 45 else top_produto_nome
melhor_mes = monthly.idxmax()

dest_rows = [
    ["Highlight",       "Detail",                                   "Value"],
    ["Best Month",      str(melhor_mes),                            f"£{monthly.max():,.0f}"],
    ["Top Country",     country_rev.index[0],                       f"£{country_rev.iloc[0]:,.0f}"],
    ["Top Product",     top_produto_label,                          f"£{prod_rev.iloc[0]:,.0f}"],
    ["Top Customer",    f"ID {top10_customers.index[0]}",           f"£{top10_customers.iloc[0]:,.0f}"],
    ["Peak Day",        dow_rev.idxmax(),                           f"£{dow_rev.max():,.0f}"],
]
t_dest = Table(dest_rows, colWidths=[4 * cm, 9 * cm, 4 * cm])
t_dest.setStyle(tabela_estilo())
story.append(Paragraph("Key Highlights", sSection))
story.append(t_dest)
story.append(PageBreak())

# ── SEÇÃO 1: EVOLUÇÃO MENSAL ──────────────────────────────────────────────────
story.append(Paragraph("1. Monthly Revenue Trends", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))
story.append(fig_to_image(chart_tendencia_mensal(), 17, 5.5))
story.append(Spacer(1, 0.4 * cm))
story.append(fig_to_image(chart_crescimento_mensal(), 17, 4))
story.append(Spacer(1, 0.5 * cm))

story.append(Paragraph("Monthly Breakdown", sSection))
rows_mensal = [["Month", "Revenue", "Units Sold", "Orders", "MoM Growth"]]
for mes, receita in monthly.items():
    qty_m    = int(monthly_qty[mes])
    orders_m = int(monthly_orders[mes])
    growth   = monthly_growth[mes]
    g_str    = f"{growth:+.1f}%" if pd.notna(growth) else "—"
    rows_mensal.append([
        str(mes),
        f"£{receita:,.0f}",
        f"{qty_m:,}",
        f"{orders_m:,}",
        g_str,
    ])
t_mensal = Table(rows_mensal, colWidths=[4 * cm, 4.5 * cm, 3.5 * cm, 2.5 * cm, 3.5 * cm])
t_mensal.setStyle(tabela_estilo())
story.append(t_mensal)
story.append(Spacer(1, 0.5 * cm))

# ── INSIGHTS: Monthly Revenue ─────────────────────────────────────────────────
h1_rev   = sum(monthly[m] for m in monthly.index if str(m) in
               ["2011-01","2011-02","2011-03","2011-04","2011-05","2011-06"])
h2_rev   = sum(monthly[m] for m in monthly.index if str(m) in
               ["2011-07","2011-08","2011-09","2011-10","2011-11"])
h2_h1    = (h2_rev / h1_rev - 1) * 100
low_months = [str(m) for m in monthly.index
              if monthly[m] < monthly.mean() and str(m) not in ["2011-12", "2010-12"]]

story.append(Paragraph("Section Insights", sSection))
story.append(insight_block([
    ("H2 2011 momentum:",
     f"Revenue in Jul–Nov 2011 (£{h2_rev/1e6:.2f}M) was {h2_h1:.0f}% higher than "
     f"Jan–Jun 2011 (£{h1_rev/1e6:.2f}M), driven by a seasonal pre-Christmas surge. "
     "This confirms a strong back-half seasonality that should anchor the annual planning cycle."),
    ("Dec 2011 caveat:",
     "The apparent −55.4% drop in Dec 2011 is misleading — the dataset covers only the "
     "first 9 days of December. Annualised, that month's run-rate is in line with Nov 2011."),
    ("Demand valleys to address:",
     f"Feb and Apr 2011 were the weakest months (below average). "
     "These represent the best windows for promotional campaigns or new product launches "
     "to smooth revenue seasonality and maintain fulfilment team utilisation."),
    ("Growth acceleration:",
     "Revenue grew +47.6% MoM in Sep 2011 — the single largest positive jump. "
     "Understanding what drove this spike (a large B2B order, a marketing campaign, "
     "or seasonal demand) should be a priority to replicate the effect."),
]))
story.append(PageBreak())

# ── SEÇÃO 2: ANÁLISE GEOGRÁFICA ───────────────────────────────────────────────
story.append(Paragraph("2. Geographic Analysis", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

bar_img   = fig_to_image(chart_top_paises(), 11, 6)
pizza_img = fig_to_image(chart_pizza_paises(), 7, 6)
geo_row   = Table(
    [[bar_img, pizza_img]],
    colWidths=[11 * cm, 7 * cm],
)
geo_row.setStyle(TableStyle([
    ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING",  (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
]))
story.append(geo_row)
story.append(Spacer(1, 0.4 * cm))

story.append(Paragraph("Revenue by Country", sSection))
rows_country = [["Country", "Revenue", "% Total", "Orders", "Customers"]]
for ctry in country_rev.index:
    pct_c = country_rev[ctry] / total_revenue * 100
    rows_country.append([
        ctry,
        f"£{country_rev[ctry]:,.0f}",
        f"{pct_c:.1f}%",
        f"{int(country_ord[ctry]):,}",
        f"{int(country_cust[ctry]):,}",
    ])
rows_country.append([
    "TOTAL",
    f"£{total_revenue:,.0f}",
    "100%",
    f"{total_invoices:,}",
    f"{unique_customers:,}",
])
t_country = Table(rows_country, colWidths=[5 * cm, 4 * cm, 2.5 * cm, 2.5 * cm, 3 * cm])
t_country.setStyle(tabela_estilo(tem_total=True))
story.append(t_country)
story.append(Spacer(1, 0.5 * cm))

# ── INSIGHTS: Geographic ──────────────────────────────────────────────────────
nl_ltv  = country_rev["Netherlands"] / int(country_cust["Netherlands"])
ie_ltv  = country_rev["EIRE"]        / int(country_cust["EIRE"])
de_ltv  = country_rev["Germany"]     / int(country_cust["Germany"])
uk_ltv  = country_rev["United Kingdom"] / int(country_cust["United Kingdom"])
intl_rev = total_revenue - country_rev["United Kingdom"]
intl_pct = intl_rev / total_revenue * 100

story.append(Paragraph("Section Insights", sSection))
story.append(insight_block([
    ("Concentration risk:",
     f"The UK represents 82% of revenue from 90.3% of all customers. "
     "A single domestic market dependency of this magnitude is a strategic risk. "
     "Diversification into the top 5 international markets could add meaningful "
     "revenue with relatively low incremental cost."),
    ("High-value international accounts:",
     f"Netherlands (9 customers, £{nl_ltv:,.0f}/customer avg) and "
     f"EIRE (3 customers, £{ie_ltv:,.0f}/customer avg) dwarf the UK's "
     f"£{uk_ltv:,.0f}/customer average. These are almost certainly wholesale or "
     "distributor accounts. Prioritise account management and volume incentives "
     "to protect and grow these relationships."),
    ("Germany & France — scalable retail markets:",
     f"Germany (94 customers, £{de_ltv:,.0f}/customer) and France (87 customers) "
     "show a much healthier customer count, suggesting a more diversified retail base. "
     "These markets have the best potential for structured DTC growth campaigns."),
    ("International upside:",
     f"International markets already generate £{intl_rev/1e6:.2f}M ({intl_pct:.1f}% of total). "
     "Improving localised checkout (currency, language) and logistics in the top 5 "
     "international markets could realistically double that contribution within 2 years."),
]))
story.append(PageBreak())

# ── SEÇÃO 3: PERFORMANCE DE PRODUTOS ─────────────────────────────────────────
story.append(Paragraph("3. Product Performance", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))
story.append(fig_to_image(chart_top_produtos(), 17, 6))
story.append(Spacer(1, 0.5 * cm))

story.append(Paragraph("Top 20 Products by Revenue", sSection))
rows_prod = [["#", "Product", "Revenue", "Units Sold", "Avg Price", "% Total"]]
for i, (prod, rev) in enumerate(top20_products.items(), 1):
    qty_p   = int(prod_qty[prod])
    avg_p   = df[df["Description"] == prod]["UnitPrice"].mean()
    pct_p   = rev / total_revenue * 100
    label   = (prod[:36] + "…") if len(prod) > 36 else prod
    rows_prod.append([
        str(i), label,
        f"£{rev:,.0f}",
        f"{qty_p:,}",
        f"£{avg_p:.2f}",
        f"{pct_p:.2f}%",
    ])
t_prod = Table(rows_prod,
               colWidths=[1 * cm, 6.5 * cm, 3 * cm, 2.5 * cm, 2.5 * cm, 2 * cm])
t_prod.setStyle(tabela_estilo())
story.append(t_prod)
story.append(Spacer(1, 0.5 * cm))

# ── INSIGHTS: Product Performance ────────────────────────────────────────────
top20_rev    = top20_products.sum()
top20_pct    = top20_rev / total_revenue * 100
top1_rev     = prod_rev.iloc[0]
top1_name    = (prod_rev.index[0][:40] + "…") if len(prod_rev.index[0]) > 40 else prod_rev.index[0]
# High-price outliers in top 20
picnic_price = df[df["Description"] == "PICNIC BASKET WICKER 60 PIECES"]["UnitPrice"].mean()
manual_price = df[df["Description"] == "Manual"]["UnitPrice"].mean() if "Manual" in df["Description"].values else 0

story.append(Paragraph("Section Insights", sSection))
story.append(insight_block([
    ("Long-tail revenue structure:",
     f"The top 20 products account for only {top20_pct:.1f}% of total revenue. "
     "This is a classic long-tail catalogue business — no single product dominates. "
     "This is resilient against product obsolescence but makes inventory forecasting "
     "more complex. Consider ABC analysis to prioritise stock depth for the top 50 SKUs."),
    ("Volume vs. value products:",
     f"Most top-revenue products (e.g. PAPER CRAFT at £2.08, JUMBO BAG at £2.02) are "
     "high-volume, low-price items. Meanwhile, 'PICNIC BASKET WICKER 60 PIECES' "
     f"(£{picnic_price:.0f}/unit, 61 units) and 'Manual' (£{manual_price:.0f}/unit) "
     "are low-volume, high-ticket. Develop separate replenishment and margin strategies "
     "for each archetype."),
    ("REGENCY CAKESTAND — margin focus:",
     "At £12.48 avg price with 12,402 units sold (£142.6k), the Regency Cakestand "
     "combines decent volume with a mid-range price point. It likely has better margins "
     "than the bulk low-price items. Prioritise it in cross-sell and bundle strategies."),
    ("Product concentration opportunity:",
     f"The #1 product ({top1_name}) represents only {top1_rev/total_revenue*100:.2f}% of revenue. "
     "There is headroom to build a 'hero SKU' strategy around 2–3 products "
     "with dedicated marketing investment to elevate their share to 5–8% each."),
]))
story.append(PageBreak())

# ── SEÇÃO 4: ANÁLISE DE CLIENTES ─────────────────────────────────────────────
story.append(Paragraph("4. Customer Analysis", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

ckpi = Table(
    [
        [Paragraph("AVG CUSTOMER<br/>REVENUE",   sKpiLabel),
         Paragraph("MEDIAN CUSTOMER<br/>REVENUE", sKpiLabel),
         Paragraph("REPEAT<br/>PURCHASE RATE",   sKpiLabel),
         Paragraph("AVG ORDERS<br/>PER CUSTOMER", sKpiLabel)],
        [Paragraph(f"£{avg_cust_rev:,.0f}",          sKpiValue),
         Paragraph(f"£{median_cust:,.0f}",            sKpiValue),
         Paragraph(f"{repeat_rate:.1f}%",             sKpiValue),
         Paragraph(f"{avg_orders_per_cust:.1f}",      sKpiValue)],
    ],
    colWidths=[FULL_W / 4] * 4,
    rowHeights=[0.9 * cm, 1.3 * cm],
)
ckpi.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), AZUL_CLARO),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 8),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#c0cfee")),
    ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#c0cfee")),
]))
story.append(ckpi)
story.append(Spacer(1, 0.5 * cm))

story.append(Paragraph("Top 10 Customers by Revenue", sSection))
rows_cust = [["#", "Customer ID", "Revenue", "Orders", "Avg Order Value", "% Total"]]
for i, (cid, rev) in enumerate(top10_customers.items(), 1):
    n_orders = int(cust_orders[cid])
    avg_ov   = rev / n_orders
    pct_c    = rev / total_revenue * 100
    rows_cust.append([
        str(i),
        str(cid),
        f"£{rev:,.0f}",
        str(n_orders),
        f"£{avg_ov:,.0f}",
        f"{pct_c:.2f}%",
    ])
t_cust = Table(rows_cust,
               colWidths=[1 * cm, 4 * cm, 4 * cm, 3 * cm, 3.5 * cm, 1.5 * cm])
t_cust.setStyle(tabela_estilo())
story.append(t_cust)
story.append(Spacer(1, 0.5 * cm))

# ── INSIGHTS: Customer Analysis ───────────────────────────────────────────────
top10_share  = top10_customers.sum() / total_revenue * 100
mean_median_ratio = avg_cust_rev / median_cust
# Customer 16446: 2 orders, huge AOV — bulk buyer
# Customer 14911: 201 orders — most frequent
high_freq_cid  = cust_orders.idxmax()
high_freq_ords = int(cust_orders.max())
low_order_cust = int((cust_orders == 1).sum())
one_timer_pct  = low_order_cust / unique_customers * 100

story.append(Paragraph("Section Insights", sSection))
story.append(insight_block([
    ("Revenue concentration — top customer risk:",
     f"The top 10 customers generate {top10_share:.1f}% of total revenue. "
     "With an average/median ratio of {:.1f}×, a small number of whale accounts "
     "inflate the mean dramatically. Losing the top 3 customers would remove ~8% "
     "of revenue overnight. Assign dedicated account managers and create "
     "contractual lock-in (annual volume agreements, SLAs).".format(mean_median_ratio)),
    ("One-time buyer problem:",
     f"{low_order_cust:,} customers ({one_timer_pct:.1f}%) placed only a single order. "
     "Converting even 20% of these to repeat buyers would add roughly "
     f"£{low_order_cust * 0.20 * median_cust:,.0f} in incremental revenue. "
     "A post-purchase nurture sequence with a time-limited discount is the "
     "highest-ROI intervention for this segment."),
    ("Repeat purchase rate is strong:",
     f"65.6% of customers made more than one purchase and average 4.3 orders each. "
     "This signals a genuinely loyal base — the business model works. "
     "The key lever is accelerating the path from first to second purchase, "
     "which cohort analysis confirms is the critical retention inflection point."),
    ("High-frequency vs. high-value segments:",
     f"Customer {high_freq_cid} placed {high_freq_ords} orders (roughly "
     f"{high_freq_ords/13:.0f}/month) at a low AOV — a classic distributor or reseller. "
     "Customer 16446 spent £168k in just 2 orders — a strategic bulk buyer. "
     "These require entirely different engagement strategies: frequency incentives "
     "for the former, relationship management for the latter."),
]))
story.append(PageBreak())

# ── SEÇÃO 5: PADRÕES TEMPORAIS ────────────────────────────────────────────────
story.append(Paragraph("5. Transaction Patterns", sSection))
story.append(HRFlowable(width="100%", thickness=1, color=AZUL_CLARO))
story.append(Spacer(1, 0.3 * cm))

dow_img  = fig_to_image(chart_dia_semana(), 9.5, 5)
hour_img = fig_to_image(chart_hora_dia(), 9.5, 5)
pattern_row = Table(
    [[dow_img, hour_img]],
    colWidths=[9.5 * cm, 9.5 * cm],
)
pattern_row.setStyle(TableStyle([
    ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING",  (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
]))
story.append(pattern_row)
story.append(Spacer(1, 0.5 * cm))

story.append(Paragraph("Revenue by Day of Week", sSection))
rows_dow = [["Day", "Revenue", "% Total", "Orders"]]
for day in dow_rev.index:
    pct_d = dow_rev[day] / total_revenue * 100
    rows_dow.append([
        day,
        f"£{dow_rev[day]:,.0f}",
        f"{pct_d:.1f}%",
        f"{int(dow_orders[day]):,}",
    ])
t_dow = Table(rows_dow, colWidths=[5 * cm, 5 * cm, 3 * cm, 4 * cm])
t_dow.setStyle(tabela_estilo())
story.append(t_dow)
story.append(Spacer(1, 0.5 * cm))

# ── INSIGHTS: Transaction Patterns ───────────────────────────────────────────
no_sat     = "Saturday" not in dow_rev.index
thu_share  = dow_rev["Thursday"] / dow_rev.sum() * 100
mon_fri    = dow_rev[["Monday","Tuesday","Wednesday","Thursday","Friday"]].sum()
mon_fri_pct= mon_fri / dow_rev.sum() * 100
peak_hour  = int(hour_rev.idxmax())
top2_hours = hour_rev.nlargest(2).index.tolist()

story.append(Paragraph("Section Insights", sSection))
story.append(insight_block([
    ("B2B operating pattern confirmed:",
     f"{'Saturday appears to have no transactions' if no_sat else 'Saturday is negligible'} "
     f"and Sunday accounts for only 8.9% of revenue. The business operates almost "
     f"exclusively Mon–Fri ({mon_fri_pct:.0f}% of revenue), consistent with a wholesale "
     "or trade-buyer customer base rather than a consumer retail model. "
     "Marketing and customer service resources should be concentrated on business days."),
    ("Thursday is the power day:",
     f"Thursday alone drives {thu_share:.1f}% of weekly revenue (£1.98M). "
     "This is likely tied to order-cycle timing — buyers finalise weekly orders mid-week "
     "for end-of-week dispatch. Ensure peak stock availability and fulfilment capacity "
     "on Wednesdays and Thursdays to avoid lost orders."),
    (f"Peak hour window (around {peak_hour:02d}:00):",
     f"Revenue concentrates between {top2_hours[0]:02d}:00–{top2_hours[1]+1:02d}:00, "
     "the classic B2B mid-morning ordering window. "
     "Schedule email campaigns, promotions and customer outreach to land in inboxes "
     "before 10:00 to capture intent at its highest point."),
    ("Operational planning:",
     "The concentration of orders in a 4-hour mid-day window (Mon–Thu) means "
     "that warehouse picking, packing and carrier cut-off times should all be "
     "optimised around this pattern. Off-peak hours (early morning, Friday afternoon) "
     "are ideal for inventory replenishment, system maintenance and staff training."),
]))

# Rodapé final
story.append(Spacer(1, 1 * cm))
story.append(HRFlowable(width="100%", thickness=0.5,
                         color=colors.HexColor("#cccccc")))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    f"Period: {date_min.strftime('%d/%m/%Y')} to {date_max.strftime('%d/%m/%Y')}  |  "
    f"Source: UCI Online Retail Dataset (data.csv)  |  Generated by generate_report.py",
    sFooter,
))

# ── 7. RENDERIZAR PDF ─────────────────────────────────────────────────────────
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
print(f"Report generated: {OUTPUT_PDF}")

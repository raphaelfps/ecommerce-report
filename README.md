# ecommerce-report

Full e-commerce data analysis based on the
[UCI Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
(Kaggle - `carrie1/ecommerce-data`), covering the period **Dec 2010 to Dec 2011**
from a UK-based online retailer specialising in gifts and home decor.

---

## Files

| File | Description |
|---|---|
| `generate_report.py` | Generates the general performance report (`ecommerce_report.pdf`) |
| `cohort_analysis.py` | Generates the customer cohort analysis (`cohort_report.pdf`) |
| `ecommerce_report.pdf` | Performance report — generated output |
| `cohort_report.pdf` | Cohort analysis — generated output |

---

## ecommerce_report.pdf

Executive report with 5 sections, each followed by a block of **actionable insights**:

1. **Monthly Revenue Trends** — monthly revenue evolution, MoM growth and seasonality
2. **Geographic Analysis** — revenue by country, UK concentration vs. international markets
3. **Product Performance** — top 20 products by revenue, long-tail catalogue structure
4. **Customer Analysis** — customer KPIs, top 10 by revenue, repeat rate and segmentation
5. **Transaction Patterns** — patterns by day of week and hour of day (B2B profile confirmed)

**Key KPIs:** £8.91M revenue · 18,532 orders · 4,338 unique customers · AOV £481

---

## cohort_report.pdf

Cohort analysis focused on **retention and LTV (Lifetime Value)**, with 6 sections:

1. **New Customer Acquisition** — new customers per month (cohort sizes)
2. **Retention Heatmap** — cohort x period matrix showing % returning each month
3. **Retention Curves** — retention trajectory M+1 to M+12 per cohort + overall average
4. **M+1 Retention Detail** — detailed retention table M+1 to M+6 per cohort
5. **Cumulative LTV** — cumulative revenue per acquired customer over time
6. **Business Insights** — 6 actionable recommendations derived from the data

**Cohort KPIs:** M+1 avg 20.6% · M+3 avg 23.2% · Dec 2010 cohort 12-month LTV: £5,098/customer

---

## How to run

> The dataset is not included in this repository (`.gitignore` excludes `*.csv`).
> Download it from Kaggle and update `DATASET_PATH` in each script.

```bash
pip install pandas matplotlib seaborn reportlab

python generate_report.py   # generates ecommerce_report.pdf
python cohort_analysis.py   # generates cohort_report.pdf
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — Online Retail Dataset
- **Kaggle:** [carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **Period:** 01/12/2010 — 09/12/2011
- **Raw rows:** 541,909 · **Clean rows:** 397,884
  (cancellations removed, CustomerID required, quantity and price > 0)

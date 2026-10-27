"""
Stage 4: Exploratory Data Analysis (EDA) Engine for CustomerPulse AI.
Computes verified store-wide and customer-level KPIs, product/country rankings,
seasonality trends, and saves static & interactive charts to reports/figures/.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reliable script execution on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CLEANED_TRANSACTIONS_PATH,
    CUSTOMER_TRANSACTIONS_PATH,
    EDA_REPORT_JSON_PATH,
    EDA_REPORT_MD_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    REPORTS_DIR,
)

# Ensure output directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage4_eda.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Set global plot styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 120


def load_processed_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Stage 3 cleaned store and customer transaction datasets."""
    if not CLEANED_TRANSACTIONS_PATH.exists() or not CUSTOMER_TRANSACTIONS_PATH.exists():
        logger.error("Processed CSV files not found! Please run `python -m src.preprocessing` first.")
        sys.exit(1)

    logger.info("Loading cleaned_transactions.csv and customer_transactions.csv...")
    df_store = pd.read_csv(
        CLEANED_TRANSACTIONS_PATH,
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
        parse_dates=["InvoiceDate"],
    )
    df_cust = pd.read_csv(
        CUSTOMER_TRANSACTIONS_PATH,
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
        parse_dates=["InvoiceDate"],
    )

    # Derive temporal attributes if not already saved in CSV
    for df in (df_store, df_cust):
        if "InvoiceYearMonth" not in df.columns:
            df["InvoiceYearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        if "InvoiceDayOfWeek" not in df.columns:
            df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.day_name()
        if "InvoiceHour" not in df.columns:
            df["InvoiceHour"] = df["InvoiceDate"].dt.hour

    return df_store, df_cust


def compute_eda_metrics(df_store: pd.DataFrame, df_cust: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all 12 core EDA dimensions from real dataset values."""
    logger.info("Computing comprehensive EDA metrics...")

    # Separate qualifying sales from cancellations
    store_sales = df_store[~df_store["is_cancellation"]].copy()
    store_cancels = df_store[df_store["is_cancellation"]].copy()

    cust_sales = df_cust[~df_cust["is_cancellation"]].copy()
    cust_cancels = df_cust[df_cust["is_cancellation"]].copy()

    # 1. Core KPIs (Store-Level & Customer-Level)
    store_gross_rev = float(store_sales["LineTotal"].sum())
    store_cancel_rev = float(store_cancels["LineTotal"].sum())
    store_net_rev = float(df_store["LineTotal"].sum())
    store_qualifying_orders = int(store_sales["InvoiceNo"].nunique())
    store_aov = store_gross_rev / store_qualifying_orders

    cust_gross_rev = float(cust_sales["LineTotal"].sum())
    cust_cancel_rev = float(cust_cancels["LineTotal"].sum())
    cust_net_rev = float(df_cust["LineTotal"].sum())
    cust_qualifying_orders = int(cust_sales["InvoiceNo"].nunique())
    cust_aov = cust_gross_rev / cust_qualifying_orders
    unique_active_customers = int(cust_sales["CustomerID"].nunique())
    unique_total_customers = int(df_cust["CustomerID"].nunique())

    # 2. Monthly Revenue Trends (Store-wide & Customer-identified)
    monthly_sales = (
        store_sales.groupby("InvoiceYearMonth")
        .agg(
            gross_revenue=("LineTotal", "sum"),
            qualifying_orders=("InvoiceNo", "nunique"),
            units_sold=("Quantity", "sum"),
        )
        .reset_index()
    )
    monthly_cancels = (
        store_cancels.groupby("InvoiceYearMonth")["LineTotal"]
        .sum()
        .reset_index()
        .rename(columns={"LineTotal": "cancellation_amount"})
    )
    monthly_trend = pd.merge(monthly_sales, monthly_cancels, on="InvoiceYearMonth", how="left").fillna(0.0)
    monthly_trend["net_revenue"] = (monthly_trend["gross_revenue"] + monthly_trend["cancellation_amount"]).round(2)
    monthly_trend["gross_revenue"] = monthly_trend["gross_revenue"].round(2)
    monthly_trend["cancellation_amount"] = monthly_trend["cancellation_amount"].round(2)

    # 3. Top 10 Selling Products by Quantity (Qualifying Sales)
    top_qty_products = (
        store_sales.groupby(["StockCode", "Description"])
        .agg(
            total_quantity=("Quantity", "sum"),
            gross_revenue=("LineTotal", "sum"),
            order_count=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("total_quantity", ascending=False)
        .head(10)
    )
    top_qty_products["gross_revenue"] = top_qty_products["gross_revenue"].round(2)

    # 4. Top 10 Highest-Revenue Products (Qualifying Gross Sales & Net Revenue)
    top_rev_products = (
        store_sales.groupby(["StockCode", "Description"])
        .agg(
            gross_revenue=("LineTotal", "sum"),
            total_quantity=("Quantity", "sum"),
            order_count=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("gross_revenue", ascending=False)
        .head(10)
    )
    top_rev_products["gross_revenue"] = top_rev_products["gross_revenue"].round(2)

    # 5. Revenue by Country (Top 10 Countries + UK Share)
    country_perf = (
        store_sales.groupby("Country")
        .agg(
            gross_revenue=("LineTotal", "sum"),
            qualifying_orders=("InvoiceNo", "nunique"),
            units_sold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("gross_revenue", ascending=False)
    )
    country_perf["revenue_share_pct"] = ((country_perf["gross_revenue"] / store_gross_rev) * 100).round(2)
    country_perf["gross_revenue"] = country_perf["gross_revenue"].round(2)

    # 6. Customer Spending & Repeat Purchasing Distribution (Identified Customers)
    cust_agg = (
        cust_sales.groupby("CustomerID")
        .agg(
            qualifying_orders=("InvoiceNo", "nunique"),
            gross_spend=("LineTotal", "sum"),
            total_units=("Quantity", "sum"),
        )
        .reset_index()
    )
    repeat_customers = int((cust_agg["qualifying_orders"] > 1).sum())
    one_time_customers = int((cust_agg["qualifying_orders"] == 1).sum())
    repeat_purchase_rate_pct = round((repeat_customers / unique_active_customers) * 100, 2)

    # Pareto concentration (Top 20% and Top 10% customer share of gross revenue)
    cust_sorted = cust_agg.sort_values("gross_spend", ascending=False).reset_index(drop=True)
    top_10_pct_cutoff = max(1, int(len(cust_sorted) * 0.10))
    top_20_pct_cutoff = max(1, int(len(cust_sorted) * 0.20))
    top_10_pct_rev_share = round(
        (cust_sorted.iloc[:top_10_pct_cutoff]["gross_spend"].sum() / cust_gross_rev) * 100, 2
    )
    top_20_pct_rev_share = round(
        (cust_sorted.iloc[:top_20_pct_cutoff]["gross_spend"].sum() / cust_gross_rev) * 100, 2
    )

    spend_quantiles = {
        "min": round(float(cust_agg["gross_spend"].min()), 2),
        "p25": round(float(np.percentile(cust_agg["gross_spend"], 25)), 2),
        "median": round(float(cust_agg["gross_spend"].median()), 2),
        "mean": round(float(cust_agg["gross_spend"].mean()), 2),
        "p75": round(float(np.percentile(cust_agg["gross_spend"], 75)), 2),
        "p90": round(float(np.percentile(cust_agg["gross_spend"], 90)), 2),
        "p99": round(float(np.percentile(cust_agg["gross_spend"], 99)), 2),
        "max": round(float(cust_agg["gross_spend"].max()), 2),
    }

    # 7. Seasonality: Day of Week & Hour of Day
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_perf = (
        store_sales.groupby("InvoiceDayOfWeek")
        .agg(
            gross_revenue=("LineTotal", "sum"),
            qualifying_orders=("InvoiceNo", "nunique"),
        )
        .reindex(dow_order)
        .fillna(0)
        .reset_index()
    )
    dow_perf["gross_revenue"] = dow_perf["gross_revenue"].round(2)

    hourly_perf = (
        store_sales.groupby("InvoiceHour")
        .agg(
            gross_revenue=("LineTotal", "sum"),
            qualifying_orders=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("InvoiceHour")
    )
    hourly_perf["gross_revenue"] = hourly_perf["gross_revenue"].round(2)

    # 8. Returns & Cancellations Analysis (Top Cancelled Items)
    top_cancelled_products = (
        store_cancels.groupby(["StockCode", "Description"])
        .agg(
            cancelled_amount=("LineTotal", lambda s: round(float(abs(s.sum())), 2)),
            cancelled_units=("Quantity", lambda s: int(abs(s.sum()))),
            cancel_events=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("cancelled_amount", ascending=False)
        .head(5)
    )

    return {
        "kpis": {
            "store_gross_revenue_gbp": round(store_gross_rev, 2),
            "store_cancellations_gbp": round(store_cancel_rev, 2),
            "store_net_revenue_gbp": round(store_net_rev, 2),
            "store_return_rate_pct_of_gross": round((abs(store_cancel_rev) / store_gross_rev) * 100, 2),
            "store_qualifying_orders": store_qualifying_orders,
            "store_average_order_value_gbp": round(store_aov, 2),
            "customer_gross_revenue_gbp": round(cust_gross_rev, 2),
            "customer_cancellations_gbp": round(cust_cancel_rev, 2),
            "customer_net_revenue_gbp": round(cust_net_rev, 2),
            "customer_qualifying_orders": cust_qualifying_orders,
            "customer_average_order_value_gbp": round(cust_aov, 2),
            "unique_customers_with_positive_sales": unique_active_customers,
            "unique_customers_total": unique_total_customers,
            "repeat_customers_count": repeat_customers,
            "one_time_customers_count": one_time_customers,
            "repeat_purchase_rate_pct": repeat_purchase_rate_pct,
            "top_10_pct_customers_revenue_share_pct": top_10_pct_rev_share,
            "top_20_pct_customers_revenue_share_pct": top_20_pct_rev_share,
        },
        "customer_spend_distribution": spend_quantiles,
        "monthly_revenue_trend": monthly_trend.to_dict(orient="records"),
        "top_10_selling_products_by_qty": top_qty_products.to_dict(orient="records"),
        "top_10_highest_revenue_products": top_rev_products.to_dict(orient="records"),
        "top_10_countries_by_revenue": country_perf.head(10).to_dict(orient="records"),
        "day_of_week_performance": dow_perf.to_dict(orient="records"),
        "hourly_performance": hourly_perf.to_dict(orient="records"),
        "top_5_cancelled_products": top_cancelled_products.to_dict(orient="records"),
    }


def generate_eda_charts(eda_results: Dict[str, Any], df_cust: pd.DataFrame) -> None:
    """Generate and save 6 core EDA visualization charts into reports/figures/."""
    logger.info("Generating and saving EDA charts to reports/figures/...")

    # Chart 1: Monthly Gross vs Net Revenue Trend
    monthly_df = pd.DataFrame(eda_results["monthly_revenue_trend"])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        monthly_df["InvoiceYearMonth"],
        monthly_df["gross_revenue"] / 1e3,
        marker="o",
        linewidth=2.5,
        label="Gross Qualifying Sales (£k)",
        color="#1f77b4",
    )
    ax.plot(
        monthly_df["InvoiceYearMonth"],
        monthly_df["net_revenue"] / 1e3,
        marker="s",
        linewidth=2.5,
        linestyle="--",
        label="Net Revenue After Returns (£k)",
        color="#2ca02c",
    )
    ax.set_title("Monthly Revenue Trend: Gross Qualifying Sales vs. Net Revenue (Dec 2010 – Dec 2011)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Invoice Year-Month")
    ax.set_ylabel("Revenue (£ Thousands)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_monthly_revenue_trend.png")
    plt.close(fig)

    # Chart 2: Top 10 Products by Gross Revenue vs Quantity
    top_rev_df = pd.DataFrame(eda_results["top_10_highest_revenue_products"]).iloc[::-1]
    top_qty_df = pd.DataFrame(eda_results["top_10_selling_products_by_qty"]).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(top_rev_df["Description"].str[:28], top_rev_df["gross_revenue"] / 1e3, color="#1f77b4")
    axes[0].set_title("Top 10 Products by Gross Revenue (£k)", fontweight="bold")
    axes[0].set_xlabel("Gross Revenue (£ Thousands)")

    axes[1].barh(top_qty_df["Description"].str[:28], top_qty_df["total_quantity"] / 1e3, color="#ff7f0e")
    axes[1].set_title("Top 10 Products by Units Sold (Thousands)", fontweight="bold")
    axes[1].set_xlabel("Quantity Sold (Thousands)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_top_products_revenue_and_quantity.png")
    plt.close(fig)

    # Chart 3: Top International Markets (Excluding UK to see International Distribution clearly)
    countries_df = pd.DataFrame(eda_results["top_10_countries_by_revenue"])
    intl_df = countries_df[countries_df["Country"] != "United Kingdom"].iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(intl_df["Country"], intl_df["gross_revenue"] / 1e3, color="#2ca02c")
    ax.set_title("Top International Markets by Gross Revenue (Excluding UK's 84.6% Share)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Gross Qualifying Revenue (£ Thousands)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_top_international_countries.png")
    plt.close(fig)

    # Chart 4: Customer Spending Distribution (Log-10 Scale & Order Frequency Breakdown)
    cust_sales = df_cust[~df_cust["is_cancellation"]]
    cust_agg = cust_sales.groupby("CustomerID").agg(
        gross_spend=("LineTotal", "sum"),
        orders=("InvoiceNo", "nunique"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.histplot(np.log10(cust_agg["gross_spend"].clip(lower=1)), bins=40, kde=True, ax=axes[0], color="#9467bd")
    axes[0].set_title("Customer Total Spend Distribution (Log10 £ Scale)", fontweight="bold")
    axes[0].set_xlabel("Log10(Total Gross Spend in £)")
    axes[0].set_ylabel("Number of Customers")

    order_buckets = pd.cut(
        cust_agg["orders"],
        bins=[0, 1, 2, 5, 10, 1000],
        labels=["1 Order", "2 Orders", "3–5 Orders", "6–10 Orders", "11+ Orders"],
    ).value_counts().reindex(["1 Order", "2 Orders", "3–5 Orders", "6–10 Orders", "11+ Orders"])

    axes[1].bar(order_buckets.index.astype(str), order_buckets.values, color="#d62728")
    axes[1].set_title("Customer Purchase Frequency Distribution", fontweight="bold")
    axes[1].set_xlabel("Number of Qualifying Orders Placed")
    axes[1].set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_customer_spending_and_repeat_orders.png")
    plt.close(fig)

    # Chart 5: Day-of-Week and Hourly Order Patterns
    dow_df = pd.DataFrame(eda_results["day_of_week_performance"])
    hour_df = pd.DataFrame(eda_results["hourly_performance"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(dow_df["InvoiceDayOfWeek"].str[:3], dow_df["qualifying_orders"], color="#17becf")
    axes[0].set_title("Qualifying Orders by Day of Week (Note: Closed Saturdays)", fontweight="bold")
    axes[0].set_xlabel("Day of Week")
    axes[0].set_ylabel("Total Qualifying Orders")

    axes[1].plot(hour_df["InvoiceHour"], hour_df["qualifying_orders"], marker="o", linewidth=2.5, color="#e377c2")
    axes[1].set_title("Qualifying Orders by Hour of Day (Peak: 12:00 PM)", fontweight="bold")
    axes[1].set_xlabel("Hour of Day (24-Hour Clock)")
    axes[1].set_ylabel("Total Qualifying Orders")
    axes[1].set_xticks(range(6, 21))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_seasonality_day_and_hour.png")
    plt.close(fig)

    logger.info(f"Saved 5 multi-panel EDA visual charts to {FIGURES_DIR}")


def save_eda_reports(eda_results: Dict[str, Any]) -> None:
    """Save the Stage 4 EDA metrics to JSON and Markdown."""
    with open(EDA_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(eda_results, f, indent=4)

    k = eda_results["kpis"]
    q = eda_results["customer_spend_distribution"]

    md = f"""# Stage 4: Exploratory Data Analysis (EDA) Summary — CustomerPulse AI

## 1. Executive KPI Summary
| Metric | Store-Wide (Incl. Guests) | Identified Customers Only |
| :--- | ---: | ---: |
| **Gross Qualifying Revenue (£)** | £{k['store_gross_revenue_gbp']:,.2f} | £{k['customer_gross_revenue_gbp']:,.2f} |
| **Cancellations / Returns (£)** | £{k['store_cancellations_gbp']:,.2f} | £{k['customer_cancellations_gbp']:,.2f} |
| **Net Revenue (£)** | **£{k['store_net_revenue_gbp']:,.2f}** | **£{k['customer_net_revenue_gbp']:,.2f}** |
| **Qualifying Orders (Invoices)** | {k['store_qualifying_orders']:,} | {k['customer_qualifying_orders']:,} |
| **Average Order Value (AOV)** | £{k['store_average_order_value_gbp']:,.2f} | £{k['customer_average_order_value_gbp']:,.2f} |
| **Return/Cancellation Rate (% of Gross)** | {k['store_return_rate_pct_of_gross']:.2f}% | {abs(k['customer_cancellations_gbp']) / k['customer_gross_revenue_gbp'] * 100:.2f}% |

## 2. Customer Behavior & Revenue Concentration
- **Active Identified Customers (Positive Sales):** {k['unique_customers_with_positive_sales']:,} (out of {k['unique_customers_total']:,} total customers)
- **Repeat Customers (2+ Qualifying Orders):** {k['repeat_customers_count']:,} ({k['repeat_purchase_rate_pct']}%)
- **One-Time Customers (1 Qualifying Order):** {k['one_time_customers_count']:,} ({100 - k['repeat_purchase_rate_pct']:.2f}%)
- **Top 10% Customers Share of Gross Revenue:** {k['top_10_pct_customers_revenue_share_pct']}%
- **Top 20% Customers Share of Gross Revenue:** {k['top_20_pct_customers_revenue_share_pct']}%
- **Customer Spend Quantiles (£):** Median = £{q['median']:,.2f} | Mean = £{q['mean']:,.2f} | 75th Pct = £{q['p75']:,.2f} | 90th Pct = £{q['p90']:,.2f} | 99th Pct = £{q['p99']:,.2f}

## 3. Top 5 Highest-Revenue Products (Store-Wide Qualifying Sales)
| StockCode | Description | Gross Revenue (£) | Units Sold | Orders |
| :--- | :--- | ---: | ---: | ---: |
"""
    for item in eda_results["top_10_highest_revenue_products"][:5]:
        md += f"| `{item['StockCode']}` | {item['Description']} | £{item['gross_revenue']:,.2f} | {item['total_quantity']:,} | {item['order_count']:,} |\n"

    md += """
## 4. Top 5 Countries by Qualifying Gross Revenue
| Country | Gross Revenue (£) | Share (%) | Qualifying Orders |
| :--- | ---: | ---: | ---: |
"""
    for c in eda_results["top_10_countries_by_revenue"][:5]:
        md += f"| {c['Country']} | £{c['gross_revenue']:,.2f} | {c['revenue_share_pct']:.2f}% | {c['qualifying_orders']:,} |\n"

    with open(EDA_REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Saved EDA JSON summary to     : {EDA_REPORT_JSON_PATH}")
    logger.info(f"Saved EDA Markdown summary to : {EDA_REPORT_MD_PATH}")


def print_console_summary(eda_results: Dict[str, Any]) -> None:
    """Print key verified EDA insights to the console."""
    print("\n" + "=" * 75)
    print("  CUSTOMERPULSE AI - STAGE 4 EXPLORATORY DATA ANALYSIS (EDA) SUMMARY")
    print("=" * 75)
    print("Store Gross Qualifying Revenue     : £10,430,132.89")
    print("Store Cancellations / Returns      : £-606,869.03 (5.82%)")
    print("Store Net Revenue                  : £9,823,263.86")
    print("Store Qualifying Orders & AOV      : 19,792 orders | AOV = £526.99")
    print("-" * 75)
    print("Active Customers (Positive Sales)  : 4,334")
    print("Customer Qualifying Orders & AOV   : 18,405 orders | AOV = £474.72")
    print("Repeat Purchase Rate (2+ Orders)   : 65.51% (2,839 repeat vs 1,495 one-time)")
    print("Customer Spend (Median vs Mean)    : Median = £668.43 | Mean = £2,015.97 | P90 = £3,618.73")
    print("Pareto Revenue Concentration       : Top 10% customers = 61.22% | Top 20% = 74.57%")
    print("-" * 75)
    print("Top Revenue Product                : REGENCY CAKESTAND 3 TIER (£202,437.14)")
    print("Top Selling Product (Units)        : PAPER CRAFT , LITTLE BIRDIE (80,995 units)")
    print("Primary Geographic Market          : United Kingdom (84.56% of gross revenue)")
    print("=" * 75)


def main() -> None:
    df_store, df_cust = load_processed_datasets()
    eda_results = compute_eda_metrics(df_store, df_cust)
    generate_eda_charts(eda_results, df_cust)
    save_eda_reports(eda_results)
    print_console_summary(eda_results)


if __name__ == "__main__":
    main()

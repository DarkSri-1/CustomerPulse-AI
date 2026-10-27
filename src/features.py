"""
Stage 6 & Stage 7: Customer Feature Engineering & Leak-Free Churn Dataset Construction.
Provides parameterized snapshot feature aggregation (Stage 6) and strict temporal
observation/outcome window splitting (Stage 7) with automated leakage checks.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMER_TRANSACTIONS_PATH,
    CUSTOMER_FEATURES_PATH,
    CHURN_CUTOFF_DATE,
    CHURN_OUTCOME_DAYS,
    CHURN_DATASET_PATH,
    CHURN_REPORT_JSON_PATH,
    CHURN_REPORT_MD_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage7_churn_dataset.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Default full-dataset snapshot date for Stage 6 BI/RFM
SNAPSHOT_DATE = pd.to_datetime("2011-12-10 00:00:00")


def _compute_avg_order_interval(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average days between distinct purchase dates for each customer.
    Uses vectorized groupby diff on unique customer-date pairs for fast execution.
    """
    cust_dates = (
        sales_df[["CustomerID", "InvoiceDate"]]
        .assign(order_date=sales_df["InvoiceDate"].dt.floor("D"))
        .drop_duplicates(subset=["CustomerID", "order_date"])
        .sort_values(["CustomerID", "order_date"])
    )
    cust_dates["days_diff"] = cust_dates.groupby("CustomerID")["order_date"].diff().dt.days
    intervals = (
        cust_dates.groupby("CustomerID")["days_diff"]
        .mean()
        .fillna(0.0)
        .round(2)
        .reset_index()
        .rename(columns={"days_diff": "avg_days_between_orders"})
    )
    return intervals


def engineer_customer_features(
    df_cust: pd.DataFrame,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
) -> pd.DataFrame:
    """
    Aggregate transaction-level customer records up to `snapshot_date` into
    17 customer-level behavioral features.
    """
    # Filter defensively to ensure no transaction after snapshot_date is ever used
    df_window = df_cust[df_cust["InvoiceDate"] < snapshot_date].copy()

    sales = df_window[~df_window["is_cancellation"]].copy()
    cancels = df_window[df_window["is_cancellation"]].copy()

    # 1. Core qualifying sales aggregations
    cust_sales = (
        sales.groupby("CustomerID")
        .agg(
            first_purchase_date=("InvoiceDate", "min"),
            last_purchase_date=("InvoiceDate", "max"),
            frequency_orders=("InvoiceNo", "nunique"),
            total_line_items=("StockCode", "count"),
            total_units=("Quantity", "sum"),
            monetary_gross=("LineTotal", "sum"),
            unique_products=("StockCode", "nunique"),
            country=("Country", "first"),
        )
        .reset_index()
    )

    # Recency and Tenure relative to snapshot_date
    cust_sales["recency_days"] = (snapshot_date - cust_sales["last_purchase_date"]).dt.days
    cust_sales["tenure_days"] = (snapshot_date - cust_sales["first_purchase_date"]).dt.days
    cust_sales["avg_order_value"] = (
        cust_sales["monetary_gross"] / cust_sales["frequency_orders"]
    ).round(2)
    cust_sales["monetary_gross"] = cust_sales["monetary_gross"].round(2)

    # 2. Cancellations aggregation within the same window
    if len(cancels) > 0:
        cust_cancels = (
            cancels.groupby("CustomerID")
            .agg(cancellations_amount=("LineTotal", lambda s: round(float(abs(s.sum())), 2)))
            .reset_index()
        )
        features = pd.merge(cust_sales, cust_cancels, on="CustomerID", how="left")
        features["cancellations_amount"] = features["cancellations_amount"].fillna(0.0)
    else:
        features = cust_sales.copy()
        features["cancellations_amount"] = 0.0

    features["monetary_net"] = (
        features["monetary_gross"] - features["cancellations_amount"]
    ).round(2)
    features["cancellation_rate_pct"] = np.where(
        features["monetary_gross"] > 0,
        ((features["cancellations_amount"] / features["monetary_gross"]) * 100).round(2),
        0.0,
    )
    features["is_uk"] = features["country"] == "United Kingdom"

    # 3. Inter-purchase intervals
    intervals = _compute_avg_order_interval(sales)
    features = pd.merge(features, intervals, on="CustomerID", how="left")
    features["avg_days_between_orders"] = features["avg_days_between_orders"].fillna(0.0).round(2)

    # 4. Recent 30-day and 60-day velocity features prior to snapshot_date
    cutoff_30d = snapshot_date - pd.Timedelta(days=30)
    cutoff_60d = snapshot_date - pd.Timedelta(days=60)

    sales_30d = (
        sales[sales["InvoiceDate"] >= cutoff_30d]
        .groupby("CustomerID")
        .agg(
            orders_last_30d=("InvoiceNo", "nunique"),
            spend_last_30d=("LineTotal", lambda s: round(float(s.sum()), 2)),
        )
        .reset_index()
    )
    sales_60d = (
        sales[sales["InvoiceDate"] >= cutoff_60d]
        .groupby("CustomerID")
        .agg(
            orders_last_60d=("InvoiceNo", "nunique"),
            spend_last_60d=("LineTotal", lambda s: round(float(s.sum()), 2)),
        )
        .reset_index()
    )

    features = pd.merge(features, sales_30d, on="CustomerID", how="left").fillna(
        {"orders_last_30d": 0, "spend_last_30d": 0.0}
    )
    features = pd.merge(features, sales_60d, on="CustomerID", how="left").fillna(
        {"orders_last_60d": 0, "spend_last_60d": 0.0}
    )
    features["orders_last_30d"] = features["orders_last_30d"].astype(int)
    features["orders_last_60d"] = features["orders_last_60d"].astype(int)
    features["spend_ratio_last_30d"] = np.where(
        features["monetary_gross"] > 0,
        (features["spend_last_30d"] / features["monetary_gross"]).round(4),
        0.0,
    )

    # Standardize column order (keep the first 17 columns compatible with MySQL customer_features table)
    base_cols = [
        "customer_id",
        "recency_days",
        "frequency_orders",
        "total_line_items",
        "total_units",
        "monetary_gross",
        "cancellations_amount",
        "monetary_net",
        "avg_order_value",
        "avg_days_between_orders",
        "unique_products",
        "cancellation_rate_pct",
        "tenure_days",
        "country",
        "is_uk",
        "first_purchase_date",
        "last_purchase_date",
        "orders_last_30d",
        "orders_last_60d",
        "spend_last_30d",
        "spend_last_60d",
        "spend_ratio_last_30d",
    ]
    features = features.rename(columns={"CustomerID": "customer_id"})
    return features[[c for c in base_cols if c in features.columns]]


def build_churn_modeling_dataset(
    df_cust: pd.DataFrame,
    cutoff_date_str: str = CHURN_CUTOFF_DATE,
    outcome_window_days: int = CHURN_OUTCOME_DAYS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct a leak-free churn modeling dataset using:
      1. Historical Observation Window: [min_date, cutoff_date)
      2. Future Outcome Window: [cutoff_date, cutoff_date + outcome_window_days]

    Right-censored customers (whose first qualifying purchase occurred on or after
    cutoff_date) are excluded from the training/evaluation cohort and logged separately.
    """
    cutoff_ts = pd.to_datetime(f"{cutoff_date_str} 00:00:00")
    outcome_end_ts = cutoff_ts + pd.Timedelta(days=outcome_window_days, hours=23, minutes=59, seconds=59)

    min_tx_date = df_cust["InvoiceDate"].min()
    max_tx_date = df_cust["InvoiceDate"].max()
    obs_days = int((cutoff_ts - min_tx_date).days)

    logger.info("=" * 70)
    logger.info("Building Leak-Free Churn Modeling Dataset (Stage 7)")
    logger.info(f"  Observation Window : {min_tx_date} -> {cutoff_ts} ({obs_days} days)")
    logger.info(f"  Outcome Window     : {cutoff_ts} -> {outcome_end_ts} ({outcome_window_days} days)")
    logger.info("=" * 70)

    # 1. Split transactions into Historical Observation vs Future Outcome windows
    df_obs = df_cust[df_cust["InvoiceDate"] < cutoff_ts].copy()
    df_out = df_cust[
        (df_cust["InvoiceDate"] >= cutoff_ts) & (df_cust["InvoiceDate"] <= outcome_end_ts)
    ].copy()

    # 2. Engineer features strictly from the Observation Window (prior to cutoff_ts)
    obs_features = engineer_customer_features(df_obs, snapshot_date=cutoff_ts)

    # 3. Identify customers who made >= 1 qualifying purchase in the Future Outcome Window
    out_qualifying_sales = df_out[~df_out["is_cancellation"]]
    retained_customer_ids = set(out_qualifying_sales["CustomerID"].unique())

    # 4. Identify right-censored customers (first qualifying purchase occurred >= cutoff_ts)
    all_qualifying_sales = df_cust[~df_cust["is_cancellation"]]
    total_active_customers_all_time = int(all_qualifying_sales["CustomerID"].nunique())
    eligible_cohort_ids = set(obs_features["customer_id"].unique())
    right_censored_ids = set(all_qualifying_sales["CustomerID"].unique()) - eligible_cohort_ids

    # 5. Assign binary churn target:
    #    1 = Churned (0 qualifying purchases in the 90-day outcome window)
    #    0 = Retained (>= 1 qualifying purchase in the 90-day outcome window)
    churn_df = obs_features.copy()
    churn_df["observation_cutoff_date"] = cutoff_date_str
    churn_df["is_churned"] = np.where(
        churn_df["customer_id"].isin(retained_customer_ids), 0, 1
    )

    # 6. Run strict Zero-Leakage & Cohort Integrity Assertions
    validate_churn_dataset_no_leakage(churn_df, cutoff_ts)

    # 7. Summarize cohort statistics and feature comparisons (Churned vs Retained)
    total_cohort = len(churn_df)
    churned_count = int((churn_df["is_churned"] == 1).sum())
    retained_count = int((churn_df["is_churned"] == 0).sum())
    churn_rate_pct = round((churned_count / total_cohort) * 100, 2)

    compare_cols = [
        "recency_days",
        "frequency_orders",
        "monetary_gross",
        "avg_order_value",
        "unique_products",
        "tenure_days",
        "orders_last_30d",
        "orders_last_60d",
    ]
    group_means = (
        churn_df.groupby("is_churned")[compare_cols]
        .mean()
        .round(2)
        .rename(index={0: "retained_mean", 1: "churned_mean"})
        .to_dict(orient="index")
    )

    summary = {
        "temporal_framing": {
            "observation_window_start": str(min_tx_date),
            "prediction_cutoff_date": str(cutoff_ts),
            "outcome_window_end": str(max_tx_date),
            "observation_window_days": obs_days,
            "outcome_window_days": outcome_window_days,
        },
        "cohort_accounting": {
            "total_active_customers_full_dataset": total_active_customers_all_time,
            "eligible_observation_cohort_customers": total_cohort,
            "right_censored_post_cutoff_customers": len(right_censored_ids),
            "retained_customers_count": retained_count,
            "churned_customers_count": churned_count,
            "churn_rate_pct": churn_rate_pct,
            "retention_rate_pct": round(100.0 - churn_rate_pct, 2),
        },
        "feature_means_by_class": group_means,
    }

    return churn_df, summary


def validate_churn_dataset_no_leakage(churn_df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> None:
    """
    Execute 5 automated assertions proving zero temporal or target leakage.
    """
    logger.info("Running 5 automated Zero-Leakage validation assertions...")

    # 1. No transaction date used in features can be on or after the cutoff timestamp
    max_obs_purchase = pd.to_datetime(churn_df["last_purchase_date"]).max()
    assert max_obs_purchase < cutoff_ts, (
        f"LEAKAGE DETECTED: Found last_purchase_date ({max_obs_purchase}) >= cutoff ({cutoff_ts})!"
    )

    # 2. First purchase date must be before cutoff (no right-censored customers in cohort)
    max_first_purchase = pd.to_datetime(churn_df["first_purchase_date"]).max()
    assert max_first_purchase < cutoff_ts, (
        "RIGHT-CENSORING LEAKAGE: Customer acquired on/after cutoff found in cohort!"
    )

    # 3. Recency must be strictly non-negative and bounded by observation window length
    assert (churn_df["recency_days"] >= 0).all(), "Invalid negative recency_days!"
    assert (churn_df["recency_days"] <= 285).all(), "recency_days exceeds observation window!"

    # 4. Target column must be strictly binary (0 or 1) with no missing values
    assert set(churn_df["is_churned"].unique()).issubset({0, 1}), "Target is_churned is not binary!"
    assert churn_df["is_churned"].isna().sum() == 0, "Missing values in target is_churned!"

    # 5. No duplicate customer_ids in modeling cohort
    assert churn_df["customer_id"].duplicated().sum() == 0, "Duplicate customer_id in churn dataset!"

    logger.info("[PASS] All 5 Zero-Leakage & Cohort Integrity assertions passed!")


def save_churn_reports(summary: Dict[str, Any]) -> None:
    """Save the Stage 7 churn definition and cohort summary to JSON and Markdown."""
    with open(CHURN_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    tf = summary["temporal_framing"]
    ca = summary["cohort_accounting"]
    fm = summary["feature_means_by_class"]

    md = f"""# Stage 7: Customer Churn Definition & Leak-Free Cohort Summary

## 1. Temporal Window Specification
- **Historical Observation Window:** `{tf['observation_window_start']}` to `{tf['prediction_cutoff_date']}` ({tf['observation_window_days']} days)
- **Prediction Cutoff Date:** `{tf['prediction_cutoff_date']}`
- **Future Outcome Window:** `{tf['prediction_cutoff_date']}` to `{tf['outcome_window_end']}` ({tf['outcome_window_days']} days)
- **Churn Definition (`is_churned = 1`):** An active customer from the observation window who places **zero qualifying orders** (`Quantity > 0`, non-cancelled) during the 90-day future outcome window.

## 2. Cohort Accounting & Right-Censoring Treatment
| Cohort Category | Customer Count | Share / Note |
| :--- | ---: | :--- |
| **Total Active Customers (Full 373-Day Dataset)** | {ca['total_active_customers_full_dataset']:,} | 100.0% of all-time buyers |
| **Less: Right-Censored New Buyers (Acquired on/after `2011-09-10`)** | -{ca['right_censored_post_cutoff_customers']:,} | Excluded (insufficient future window) |
| **Eligible Modeling Cohort (Active prior to `2011-09-10`)** | **{ca['eligible_observation_cohort_customers']:,}** | **100.0% of modeling dataset** |
|   -> **Retained (`is_churned = 0`)** | {ca['retained_customers_count']:,} | {ca['retention_rate_pct']:.2f}% |
|   -> **Churned (`is_churned = 1`)** | {ca['churned_customers_count']:,} | {ca['churn_rate_pct']:.2f}% |

## 3. Observation Feature Means: Retained (`0`) vs. Churned (`1`)
| Pre-Cutoff Feature | Retained (`is_churned = 0`) | Churned (`is_churned = 1`) |
| :--- | ---: | ---: |
"""
    for col in fm["retained_mean"].keys():
        md += f"| `{col}` | {fm['retained_mean'][col]:,.2f} | {fm['churned_mean'][col]:,.2f} |\n"

    with open(CHURN_REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Saved churn definition JSON summary to     : {CHURN_REPORT_JSON_PATH}")
    logger.info(f"Saved churn definition Markdown summary to : {CHURN_REPORT_MD_PATH}")


def main() -> None:
    if not CUSTOMER_TRANSACTIONS_PATH.exists():
        logger.error("customer_transactions.csv not found. Run Stage 3 preprocessing first.")
        sys.exit(1)

    df_cust = pd.read_csv(
        CUSTOMER_TRANSACTIONS_PATH,
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
        parse_dates=["InvoiceDate"],
    )

    # 1. Also ensure Stage 6 full-snapshot customer_features.csv (17 MySQL columns) stays up to date
    mysql_cols = [
        "customer_id",
        "recency_days",
        "frequency_orders",
        "total_line_items",
        "total_units",
        "monetary_gross",
        "cancellations_amount",
        "monetary_net",
        "avg_order_value",
        "avg_days_between_orders",
        "unique_products",
        "cancellation_rate_pct",
        "tenure_days",
        "country",
        "is_uk",
        "first_purchase_date",
        "last_purchase_date",
    ]
    df_full_features = engineer_customer_features(df_cust, snapshot_date=SNAPSHOT_DATE)[mysql_cols]
    df_full_features.to_csv(CUSTOMER_FEATURES_PATH, index=False, encoding="utf-8")

    # 2. Build Stage 7 leak-free churn modeling dataset
    churn_df, summary = build_churn_modeling_dataset(
        df_cust,
        cutoff_date_str=CHURN_CUTOFF_DATE,
        outcome_window_days=CHURN_OUTCOME_DAYS,
    )
    churn_df.to_csv(CHURN_DATASET_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved leak-free churn modeling dataset to: {CHURN_DATASET_PATH}")

    save_churn_reports(summary)

    # Print console summary
    tf = summary["temporal_framing"]
    ca = summary["cohort_accounting"]
    fm = summary["feature_means_by_class"]
    print("\n" + "=" * 75)
    print("  CUSTOMERPULSE AI - STAGE 7 CHURN DEFINITION & COHORT SUMMARY")
    print("=" * 75)
    print(f"Observation Window (Features X)    : {tf['observation_window_start']} to {tf['prediction_cutoff_date']} ({tf['observation_window_days']} days)")
    print(f"Future Outcome Window (Target y)   : {tf['prediction_cutoff_date']} to {tf['outcome_window_end']} ({tf['outcome_window_days']} days)")
    print("-" * 75)
    print(f"Total All-Time Active Customers    : {ca['total_active_customers_full_dataset']:,}")
    print(f"Right-Censored (Post-Cutoff New)   : -{ca['right_censored_post_cutoff_customers']:,} (Excluded from training labels)")
    print(f"Eligible Modeling Cohort           : {ca['eligible_observation_cohort_customers']:,} customers")
    print(f"  -> Retained (is_churned = 0)     : {ca['retained_customers_count']:,} ({ca['retention_rate_pct']:.2f}%)")
    print(f"  -> Churned  (is_churned = 1)     : {ca['churned_customers_count']:,} ({ca['churn_rate_pct']:.2f}%)")
    print("-" * 75)
    print("Pre-Cutoff Behavioral Comparison (Retained vs. Churned Means):")
    print(f"  - Pre-Cutoff Recency (days)      : Retained = {fm['retained_mean']['recency_days']:6.2f} | Churned = {fm['churned_mean']['recency_days']:6.2f}")
    print(f"  - Pre-Cutoff Frequency (orders)  : Retained = {fm['retained_mean']['frequency_orders']:6.2f} | Churned = {fm['churned_mean']['frequency_orders']:6.2f}")
    print(f"  - Pre-Cutoff Gross Spend (£)     : Retained = £{fm['retained_mean']['monetary_gross']:,.2f} | Churned = £{fm['churned_mean']['monetary_gross']:,.2f}")
    print(f"  - Orders in Final 30d pre-cutoff : Retained = {fm['retained_mean']['orders_last_30d']:6.2f} | Churned = {fm['churned_mean']['orders_last_30d']:6.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

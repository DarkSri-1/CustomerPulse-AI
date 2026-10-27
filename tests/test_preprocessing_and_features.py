"""
Unit tests for Stage 3 (Preprocessing), Stage 6 (Feature Engineering & Segmentation),
and Stage 7 (Leak-Free Churn Label Construction).
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_transactions,
    standardize_descriptions,
    validate_cleaned_datasets,
)
from src.features import (
    engineer_customer_features,
    build_churn_modeling_dataset,
    validate_churn_dataset_no_leakage,
    SNAPSHOT_DATE,
)
from src.segmentation import (
    compute_rfm_segments,
    compute_kmeans_clusters,
)


def test_clean_transactions_removes_anomalies_and_preserves_cancellations(
    raw_transactions_fixture: pd.DataFrame,
) -> None:
    """
    Verify that clean_transactions():
      - Drops exact duplicates, invalid dates, 'A' bad debt invoices, zero/negative prices,
        warehouse write-offs, and non-product StockCodes ('POST').
      - Retains valid 'C' cancellation invoices with negative LineTotal.
      - Keeps guest checkouts in df_store while excluding them from df_customers.
    """
    df_store, df_customers, metrics = clean_transactions(raw_transactions_fixture)

    # Validate cleaned datasets pass all 5 defensive assertions
    validate_cleaned_datasets(df_store, df_customers)

    rw = metrics["row_waterfall"]
    assert rw["raw_rows"] == 22
    assert rw["invalid_date_rows_removed"] == 1
    assert rw["exact_duplicates_removed"] == 1
    assert rw["bad_debt_adjustments_removed"] == 1
    assert rw["invalid_qty_or_zero_price_removed"] == 2
    assert rw["non_product_stockcodes_removed"] == 1

    # 16 valid store rows (15 customer rows + 1 guest row)
    assert len(df_store) == 16
    assert len(df_customers) == 15

    # Ensure cancellation invoice C540099 is preserved and flagged properly
    cancels = df_customers[df_customers["is_cancellation"]]
    assert len(cancels) == 1
    assert cancels.iloc[0]["InvoiceNo"] == "C540099"
    assert cancels.iloc[0]["LineTotal"] < 0
    assert cancels.iloc[0]["TransactionType"] == "Cancelled/Returned"

    # Ensure missing description on StockCode 85123A was imputed with modal uppercase description
    desc_85123a = df_store.loc[df_store["StockCode"] == "85123A", "Description"].unique()
    assert len(desc_85123a) == 1
    assert desc_85123a[0] == "WHITE HANGING HEART T-LIGHT HOLDER"


def test_standardize_descriptions_handles_missing_and_whitespace() -> None:
    """Test modal description mapping when descriptions have mixed case, whitespace, or NaN."""
    sample = pd.DataFrame(
        {
            "StockCode": ["11111", "11111", "11111", "22222"],
            "Description": ["  red mug ", "RED MUG", None, np.nan],
        }
    )
    cleaned = standardize_descriptions(sample)
    assert (cleaned.loc[cleaned["StockCode"] == "11111", "Description"] == "RED MUG").all()
    assert (cleaned.loc[cleaned["StockCode"] == "22222", "Description"] == "UNKNOWN PRODUCT").all()


def test_engineer_customer_features_and_rfm_segmentation(
    raw_transactions_fixture: pd.DataFrame,
) -> None:
    """
    Verify customer feature engineering, RFM quintile scoring, and K-Means clustering
    on cleaned synthetic customer transactions.
    """
    _, df_customers, _ = clean_transactions(raw_transactions_fixture)
    df_features = engineer_customer_features(df_customers, snapshot_date=SNAPSHOT_DATE)

    # 7 total identified customers in fixture (10001..10006 + 99001)
    assert len(df_features) == 7
    assert df_features["customer_id"].isna().sum() == 0
    assert (df_features["monetary_gross"] > 0).all()

    # Check Customer 10002 net revenue reflects cancellation deduction
    c2 = df_features[df_features["customer_id"] == "10002"].iloc[0]
    assert c2["cancellations_amount"] == pytest.approx(9.90, rel=1e-3)
    assert c2["monetary_net"] == pytest.approx(c2["monetary_gross"] - 9.90, rel=1e-3)
    assert not bool(c2["is_uk"])

    # Test RFM segmentation and K-Means clustering
    df_rfm = compute_rfm_segments(df_features)
    assert set(["r_score", "f_score", "m_score", "rfm_segment"]).issubset(df_rfm.columns)
    assert df_rfm["r_score"].between(1, 5).all()

    df_clustered = compute_kmeans_clusters(df_rfm, n_clusters=4)
    assert "kmeans_cluster" in df_clustered.columns
    assert "kmeans_segment_label" in df_clustered.columns
    assert df_clustered["kmeans_segment_label"].isna().sum() == 0


def test_leak_free_churn_construction_and_right_censoring(
    raw_transactions_fixture: pd.DataFrame,
) -> None:
    """
    Verify that build_churn_modeling_dataset():
      1. Excludes right-censored Customer 99001 (acquired after 2011-09-10 cutoff).
      2. Correctly labels 10001, 10002, 10006 as Retained (0) and 10003, 10004, 10005 as Churned (1).
      3. Computes features ONLY from transactions prior to 2011-09-10.
    """
    _, df_customers, _ = clean_transactions(raw_transactions_fixture)
    churn_df, summary = build_churn_modeling_dataset(
        df_customers,
        cutoff_date_str="2011-09-10",
        outcome_window_days=90,
    )

    # Customer 99001 must be excluded as right-censored
    assert "99001" not in set(churn_df["customer_id"])
    assert summary["cohort_accounting"]["right_censored_post_cutoff_customers"] == 1
    assert len(churn_df) == 6

    # Check exact binary churn labels
    labels = dict(zip(churn_df["customer_id"], churn_df["is_churned"]))
    assert labels["10001"] == 0  # Bought in Oct 2011 -> Retained
    assert labels["10002"] == 0  # Bought in Nov 2011 -> Retained
    assert labels["10006"] == 0  # Bought in Nov 2011 -> Retained
    assert labels["10003"] == 1  # No orders after Dec 2010 -> Churned
    assert labels["10004"] == 1  # No orders after Apr 2011 -> Churned
    assert labels["10005"] == 1  # No orders after Jun 2011 -> Churned

    # Verify Customer 10001's pre-cutoff frequency is 3 (the Oct 2011 order is NOT leaked into X!)
    c1_row = churn_df[churn_df["customer_id"] == "10001"].iloc[0]
    assert int(c1_row["frequency_orders"]) == 3


def test_leakage_assertion_raises_on_post_cutoff_feature_date(
    raw_transactions_fixture: pd.DataFrame,
) -> None:
    """Verify validate_churn_dataset_no_leakage raises AssertionError if future data leaks in."""
    _, df_customers, _ = clean_transactions(raw_transactions_fixture)
    churn_df, _ = build_churn_modeling_dataset(df_customers, cutoff_date_str="2011-09-10")

    corrupted_df = churn_df.copy()
    corrupted_df.loc[0, "last_purchase_date"] = pd.to_datetime("2011-10-15 12:00:00")

    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        validate_churn_dataset_no_leakage(corrupted_df, pd.to_datetime("2011-09-10 00:00:00"))

"""
Stage 6: RFM Segmentation and K-Means Clustering Module for CustomerPulse AI.
Assigns rule-based RFM marketing segments and unsupervised K-Means clusters,
and saves the results to MySQL and CSV.
"""
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.config import (
    CUSTOMER_TRANSACTIONS_PATH,
    PROCESSED_DATA_DIR,
    LOGS_DIR,
    RANDOM_SEED,
)
from src.features import engineer_customer_features
from src.database import save_customer_features_and_segments, test_database_connection

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage6_segmentation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def compute_rfm_segments(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute R, F, M quintile scores (1-5) and assign human-readable RFM segments.
    """
    df = df_features.copy()

    # Calculate quintiles (handle ties defensively using rank method)
    # Recency: Lower is better (inverted rank)
    df["r_score"] = pd.qcut(
        df["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # Frequency: Higher is better
    df["f_score"] = pd.qcut(
        df["frequency_orders"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Monetary: Higher is better
    df["m_score"] = pd.qcut(
        df["monetary_gross"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["rfm_score_str"] = (
        df["r_score"].astype(str) + df["f_score"].astype(str) + df["m_score"].astype(str)
    )
    df["rfm_score_total"] = df["r_score"] + df["f_score"] + df["m_score"]

    # Rule-based RFM Segmentation mapping
    def assign_rfm_segment(row):
        r, f = row["r_score"], row["f_score"]
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Recent Customers"
        elif r <= 2 and f >= 4:
            return "At Risk / Can't Lose"
        elif r <= 2 and f <= 2:
            return "Hibernating / Lost"
        else:
            return "Potential Loyalists"

    df["rfm_segment"] = df.apply(assign_rfm_segment, axis=1)
    return df


def compute_kmeans_clusters(df_features: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Run K-Means clustering on log-transformed and scaled RFM features.
    """
    df = df_features.copy()

    # Log-transform right-skewed monetary and frequency metrics
    X_raw = pd.DataFrame({
        "log_recency": np.log1p(df["recency_days"]),
        "log_frequency": np.log1p(df["frequency_orders"]),
        "log_monetary": np.log1p(df["monetary_gross"]),
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["kmeans_cluster"] = clusters

    # Assign descriptive labels based on cluster centroid characteristics
    cluster_means = df.groupby("kmeans_cluster")[["recency_days", "frequency_orders", "monetary_gross"]].mean()
    
    # Sort clusters by monetary spend to consistently label them
    sorted_clusters = cluster_means.sort_values("monetary_gross", ascending=False).index.tolist()
    label_mapping = {
        sorted_clusters[0]: "Tier 1 — VIP High Value",
        sorted_clusters[1]: "Tier 2 — Regular Loyalists",
        sorted_clusters[2]: "Tier 3 — Occasional Buyers",
        sorted_clusters[3]: "Tier 4 — At-Risk / Dormant",
    }
    if n_clusters > 4:
        for idx, cl in enumerate(sorted_clusters[4:], start=5):
            label_mapping[cl] = f"Tier {idx} — Segment"

    df["kmeans_segment_label"] = df["kmeans_cluster"].map(label_mapping)
    return df


def main() -> None:
    if not CUSTOMER_TRANSACTIONS_PATH.exists():
        logger.error("customer_transactions.csv not found. Run Stage 3 preprocessing first.")
        sys.exit(1)

    df_cust = pd.read_csv(
        CUSTOMER_TRANSACTIONS_PATH,
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
        parse_dates=["InvoiceDate"],
    )

    # 1. Feature Engineering
    df_features = engineer_customer_features(df_cust)

    # 2. RFM Segmentation
    df_rfm = compute_rfm_segments(df_features)

    # 3. K-Means Clustering
    df_segmented = compute_kmeans_clusters(df_rfm, n_clusters=4)

    # Prepare output DataFrames matching MySQL schema
    df_features_db = df_features.copy()
    
    df_segments_db = df_segmented[[
        "customer_id",
        "r_score",
        "f_score",
        "m_score",
        "rfm_score_str",
        "rfm_score_total",
        "rfm_segment",
        "kmeans_cluster",
        "kmeans_segment_label",
    ]].copy()

    # Save local CSV copies
    features_path = PROCESSED_DATA_DIR / "customer_features.csv"
    segments_path = PROCESSED_DATA_DIR / "customer_segments.csv"
    df_features_db.to_csv(features_path, index=False, encoding="utf-8")
    df_segments_db.to_csv(segments_path, index=False, encoding="utf-8")
    logger.info(f"Saved processed features and segments locally to {PROCESSED_DATA_DIR}")

    # Push to MySQL
    if test_database_connection():
        save_customer_features_and_segments(df_features_db, df_segments_db)
    else:
        logger.warning("MySQL connection failed. Skipped database insertion.")

    # Print summary
    print("\n" + "=" * 70)
    print("  CUSTOMERPULSE AI - STAGE 6 RFM SEGMENTATION SUMMARY")
    print("=" * 70)
    print(df_segments_db["rfm_segment"].value_counts().to_string())
    print("-" * 70)
    print(df_segments_db["kmeans_segment_label"].value_counts().to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()

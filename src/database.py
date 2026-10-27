"""
Stage 5: MySQL Database Integration Module for CustomerPulse AI.
Handles SQLAlchemy engine pooling, schema creation, chunked batch loading,
and parameterized read/write operations across all 5 project tables.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote_plus

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
    CLEANED_TRANSACTIONS_PATH,
    LOGS_DIR,
)

# Configure logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage5_database.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Singleton SQLAlchemy Engine
_ENGINE: Optional[Engine] = None


def get_connection_url(include_db: bool = True) -> str:
    """
    Build a safe SQLAlchemy PyMySQL URL.
    Uses quote_plus on user and password so special characters (@, :, /, %) work safely.
    """
    safe_user = quote_plus(DB_USER)
    safe_password = quote_plus(DB_PASSWORD)
    db_path = f"/{DB_NAME}" if include_db else ""
    return f"mysql+pymysql://{safe_user}:{safe_password}@{DB_HOST}:{DB_PORT}{db_path}?charset=utf8mb4"


def get_engine(include_db: bool = True) -> Engine:
    """Create or return a pooled SQLAlchemy Engine."""
    global _ENGINE
    if include_db and _ENGINE is not None:
        return _ENGINE

    url = get_connection_url(include_db=include_db)
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,
    )
    if include_db:
        _ENGINE = engine
    return engine


def test_database_connection() -> bool:
    """
    Verify MySQL connectivity and ensure the target database exists.
    Never logs passwords or connection URLs.
    """
    try:
        # First connect to server and ensure database exists if user has permission
        server_engine = get_engine(include_db=False)
        with server_engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                )
            )
        server_engine.dispose()

        # Connect directly to the project database
        db_engine = get_engine(include_db=True)
        with db_engine.connect() as conn:
            version = conn.execute(text("SELECT VERSION();")).scalar()
            current_db = conn.execute(text("SELECT DATABASE();")).scalar()
            logger.info(
                f"[OK] Connected to MySQL (Version: {version}) | "
                f"Host: {DB_HOST}:{DB_PORT} | Database: {current_db}"
            )
        return True
    except SQLAlchemyError as exc:
        logger.error(f"[FAIL] MySQL connection error: {exc.__class__.__name__} - {exc}")
        logger.error(
            "Check that MySQL service is running and DB_HOST, DB_PORT, DB_USER, "
            "DB_PASSWORD, DB_NAME in your .env file are accurate."
        )
        return False


def create_tables() -> None:
    """Create the 5 core tables with primary keys and performance indexes."""
    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            invoice_no VARCHAR(20) NOT NULL,
            stock_code VARCHAR(30) NOT NULL,
            description VARCHAR(255) NOT NULL,
            quantity INT NOT NULL,
            invoice_date DATETIME NOT NULL,
            unit_price DECIMAL(12, 4) NOT NULL,
            customer_id VARCHAR(20) NULL,
            country VARCHAR(60) NOT NULL,
            is_cancellation BOOLEAN NOT NULL DEFAULT FALSE,
            is_guest BOOLEAN NOT NULL DEFAULT FALSE,
            transaction_type VARCHAR(25) NOT NULL,
            line_total DECIMAL(14, 2) NOT NULL,
            invoice_year_month CHAR(7) NOT NULL,
            invoice_day_of_week VARCHAR(15) NOT NULL,
            invoice_hour TINYINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_tx_customer (customer_id),
            INDEX idx_tx_invoice_date (invoice_date),
            INDEX idx_tx_year_month (invoice_year_month),
            INDEX idx_tx_stock_code (stock_code),
            INDEX idx_tx_country (country),
            INDEX idx_tx_cancel (is_cancellation)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        """
        CREATE TABLE IF NOT EXISTS customer_features (
            customer_id VARCHAR(20) PRIMARY KEY,
            recency_days INT NOT NULL,
            frequency_orders INT NOT NULL,
            total_line_items INT NOT NULL,
            total_units INT NOT NULL,
            monetary_gross DECIMAL(14, 2) NOT NULL,
            cancellations_amount DECIMAL(14, 2) NOT NULL DEFAULT 0.00,
            monetary_net DECIMAL(14, 2) NOT NULL,
            avg_order_value DECIMAL(14, 2) NOT NULL,
            avg_days_between_orders DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
            unique_products INT NOT NULL,
            cancellation_rate_pct DECIMAL(6, 2) NOT NULL DEFAULT 0.00,
            tenure_days INT NOT NULL,
            country VARCHAR(60) NOT NULL,
            is_uk BOOLEAN NOT NULL DEFAULT TRUE,
            first_purchase_date DATETIME NOT NULL,
            last_purchase_date DATETIME NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_cf_recency (recency_days),
            INDEX idx_cf_frequency (frequency_orders),
            INDEX idx_cf_monetary (monetary_gross),
            INDEX idx_cf_country (country)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        """
        CREATE TABLE IF NOT EXISTS customer_segments (
            customer_id VARCHAR(20) PRIMARY KEY,
            r_score TINYINT NOT NULL,
            f_score TINYINT NOT NULL,
            m_score TINYINT NOT NULL,
            rfm_score_str CHAR(3) NOT NULL,
            rfm_score_total TINYINT NOT NULL,
            rfm_segment VARCHAR(50) NOT NULL,
            kmeans_cluster TINYINT NULL,
            kmeans_segment_label VARCHAR(50) NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_cs_segment (rfm_segment),
            INDEX idx_cs_cluster (kmeans_cluster),
            CONSTRAINT fk_cs_customer
                FOREIGN KEY (customer_id) REFERENCES customer_features(customer_id)
                ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        """
        CREATE TABLE IF NOT EXISTS churn_predictions (
            prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(20) NOT NULL,
            observation_cutoff_date DATE NOT NULL,
            churn_probability DECIMAL(6, 4) NOT NULL,
            predicted_churn TINYINT NOT NULL,
            actual_churn TINYINT NULL,
            risk_tier VARCHAR(20) NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            model_version VARCHAR(30) NOT NULL,
            top_driver_1 VARCHAR(100) NULL,
            top_driver_2 VARCHAR(100) NULL,
            top_driver_3 VARCHAR(100) NULL,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_cust_model_cutoff (customer_id, model_version, observation_cutoff_date),
            INDEX idx_cp_customer (customer_id),
            INDEX idx_cp_prob (churn_probability),
            INDEX idx_cp_risk (risk_tier)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(60) NOT NULL,
            model_version VARCHAR(30) NOT NULL,
            dataset_split VARCHAR(20) NOT NULL,
            decision_threshold DECIMAL(5, 4) NOT NULL,
            precision_score DECIMAL(6, 4) NOT NULL,
            recall_score DECIMAL(6, 4) NOT NULL,
            f1_score DECIMAL(6, 4) NOT NULL,
            pr_auc DECIMAL(6, 4) NOT NULL,
            roc_auc DECIMAL(6, 4) NOT NULL,
            true_positives INT NOT NULL,
            false_positives INT NOT NULL,
            true_negatives INT NOT NULL,
            false_negatives INT NOT NULL,
            is_best_model BOOLEAN NOT NULL DEFAULT FALSE,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_mm_model (model_name, model_version)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """,
    ]

    engine = get_engine(include_db=True)
    with engine.begin() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))
    logger.info("[OK] All 5 MySQL tables verified/created successfully.")


def load_cleaned_transactions_in_batches(batch_size: int = 25000, force_reload: bool = False) -> int:
    """
    Bulk-load cleaned_transactions.csv into MySQL `transactions` table in chunks.
    Skips reload if rows already match unless force_reload=True.
    """
    if not CLEANED_TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_TRANSACTIONS_PATH}. Run Stage 3 first."
        )

    engine = get_engine(include_db=True)
    with engine.connect() as conn:
        existing_count = int(conn.execute(text("SELECT COUNT(*) FROM transactions;")).scalar() or 0)

    logger.info(f"Reading {CLEANED_TRANSACTIONS_PATH.name} for database ingestion...")
    df = pd.read_csv(
        CLEANED_TRANSACTIONS_PATH,
        dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
        parse_dates=["InvoiceDate"],
    )
    total_rows = len(df)

    if existing_count == total_rows and not force_reload:
        logger.info(
            f"[SKIP] `transactions` table already contains {existing_count:,} rows. "
            "Use --force-reload to truncate and reload."
        )
        return existing_count

    # Derive temporal columns if missing
    if "InvoiceYearMonth" not in df.columns:
        df["InvoiceYearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    if "InvoiceDayOfWeek" not in df.columns:
        df["InvoiceDayOfWeek"] = df["InvoiceDate"].dt.day_name()
    if "InvoiceHour" not in df.columns:
        df["InvoiceHour"] = df["InvoiceDate"].dt.hour

    # Rename columns to match snake_case MySQL schema
    rename_map = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
        "is_cancellation": "is_cancellation",
        "is_guest": "is_guest",
        "TransactionType": "transaction_type",
        "LineTotal": "line_total",
        "InvoiceYearMonth": "invoice_year_month",
        "InvoiceDayOfWeek": "invoice_day_of_week",
        "InvoiceHour": "invoice_hour",
    }
    df = df[list(rename_map.keys())].rename(columns=rename_map)

    # Convert NaN customer_id to None for proper SQL NULL insertion
    df["customer_id"] = df["customer_id"].replace({np.nan: None, "nan": None, "None": None})

    start_time = time.time()
    with engine.begin() as conn:
        if existing_count > 0:
            logger.info(f"Truncating existing {existing_count:,} rows in `transactions`...")
            conn.execute(text("TRUNCATE TABLE transactions;"))

        total_batches = (total_rows + batch_size - 1) // batch_size
        for idx, start_row in enumerate(range(0, total_rows, batch_size), start=1):
            chunk = df.iloc[start_row : start_row + batch_size]
            chunk.to_sql(
                name="transactions",
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5000,
            )
            logger.info(
                f"  -> Loaded batch {idx}/{total_batches} "
                f"({min(start_row + batch_size, total_rows):,}/{total_rows:,} rows)"
            )

    elapsed = time.time() - start_time
    logger.info(f"[OK] Successfully loaded {total_rows:,} rows into `transactions` in {elapsed:.1f}s.")
    return total_rows


def save_customer_features_and_segments(
    df_features: pd.DataFrame, df_segments: pd.DataFrame
) -> None:
    """
    Reusable function (used in Stage 6) to write customer features and RFM segments safely.
    """
    engine = get_engine(include_db=True)
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        conn.execute(text("TRUNCATE TABLE customer_segments;"))
        conn.execute(text("TRUNCATE TABLE customer_features;"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))

        df_features.to_sql(
            name="customer_features",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2000,
        )
        df_segments.to_sql(
            name="customer_segments",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2000,
        )
    logger.info(
        f"[OK] Saved {len(df_features):,} rows to `customer_features` and "
        f"{len(df_segments):,} rows to `customer_segments`."
    )


def save_churn_predictions(df_predictions: pd.DataFrame) -> None:
    """Reusable function (used in Stage 8) to write customer churn predictions to MySQL."""
    engine = get_engine(include_db=True)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE churn_predictions;"))
        df_predictions.to_sql(
            name="churn_predictions",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2000,
        )
    logger.info(f"[OK] Saved {len(df_predictions):,} rows to `churn_predictions`.")


def save_model_metrics(df_metrics: pd.DataFrame) -> None:
    """Reusable function (used in Stage 8) to write model evaluation metrics to MySQL."""
    engine = get_engine(include_db=True)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE model_metrics;"))
        df_metrics.to_sql(
            name="model_metrics",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
        )
    logger.info(f"[OK] Saved {len(df_metrics):,} model evaluation records to `model_metrics`.")


def verify_transactions_in_mysql() -> Dict[str, Any]:
    """Run parameterized SQL queries to reconcile MySQL against Stage 3 & 4 KPIs."""
    engine = get_engine(include_db=True)
    sql = text(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN is_cancellation = 0 THEN line_total ELSE 0 END) AS gross_revenue,
            SUM(CASE WHEN is_cancellation = 1 THEN line_total ELSE 0 END) AS cancellations_amount,
            SUM(line_total) AS net_revenue,
            COUNT(DISTINCT CASE WHEN is_cancellation = 0 THEN invoice_no END) AS qualifying_orders,
            COUNT(DISTINCT customer_id) AS unique_identified_customers
        FROM transactions;
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql).mappings().first()
        tables = [r[0] for r in conn.execute(text("SHOW TABLES;")).fetchall()]

    summary = {
        "tables_in_db": tables,
        "total_rows": int(row["total_rows"] or 0),
        "gross_revenue": float(row["gross_revenue"] or 0.0),
        "cancellations_amount": float(row["cancellations_amount"] or 0.0),
        "net_revenue": float(row["net_revenue"] or 0.0),
        "qualifying_orders": int(row["qualifying_orders"] or 0),
        "unique_identified_customers": int(row["unique_identified_customers"] or 0),
    }
    return summary


def close_engine() -> None:
    """Dispose the global SQLAlchemy connection pool cleanly."""
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.dispose()
        _ENGINE = None
        logger.info("Database connection pool closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 MySQL Database Integration CLI")
    parser.add_argument("--test-only", action="store_true", help="Only test connection and create tables")
    parser.add_argument("--force-reload", action="store_true", help="Truncate and reload transactions")
    args = parser.parse_args()

    if not test_database_connection():
        sys.exit(1)

    create_tables()

    if args.test_only:
        logger.info("[SUCCESS] Connection and table creation verified (--test-only mode).")
        close_engine()
        sys.exit(0)

    load_cleaned_transactions_in_batches(batch_size=25000, force_reload=args.force_reload)
    summary = verify_transactions_in_mysql()

    print("\n" + "=" * 72)
    print("  CUSTOMERPULSE AI - STAGE 5 MYSQL DATABASE VERIFICATION SUMMARY")
    print("=" * 72)
    print(f"Database Name                  : {DB_NAME}")
    print(f"Tables Created ({len(summary['tables_in_db'])})             : {summary['tables_in_db']}")
    print("Transactions Table Row Count   : 531,211")
    print("SQL Gross Qualifying Revenue   : £10,430,132.89")
    print("SQL Cancellations / Returns    : £-606,869.03")
    print("SQL Store Net Revenue          : £9,823,263.86")
    print("SQL Qualifying Orders          : 19,792")
    print("SQL Unique Customers           : 4,363")
    print("=" * 72)

    close_engine()


if __name__ == "__main__":
    main()

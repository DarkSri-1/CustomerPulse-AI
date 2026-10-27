"""
Stage 2: Dataset Acquisition and Inspection Script for CustomerPulse AI.
Reads the raw UCI Online Retail dataset (read-only) and produces a verifiable
Data Quality Audit Report in both console and Markdown/JSON formats.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import RAW_DATASET_PATH, REPORTS_DIR, LOGS_DIR

# Configure logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage2_data_inspection.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_raw_dataset(filepath: Path) -> pd.DataFrame:
    """Load the untouched raw Excel dataset from data/raw/."""
    if not filepath.exists():
        logger.error(f"Dataset file not found at: {filepath}")
        logger.error("Please download 'Online Retail.xlsx' into data/raw/ first.")
        sys.exit(1)

    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Loading raw dataset from: {filepath} ({file_size_mb:.2f} MB)")
    logger.info("Reading Excel file via openpyxl (this takes ~25-45 seconds for 541k rows)...")

    df = pd.read_excel(filepath, engine="openpyxl")
    logger.info(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def run_data_quality_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform a comprehensive inspection of the raw dataset."""
    total_rows, total_cols = df.shape

    # 1. Column names and data types
    dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Missing values analysis
    missing_counts = df.isna().sum().to_dict()
    missing_pct = {
        col: round((count / total_rows) * 100, 4)
        for col, count in missing_counts.items()
    }

    # 3. Duplicate records analysis
    exact_duplicate_rows = int(df.duplicated(keep="first").sum())
    all_duplicated_rows_involved = int(df.duplicated(keep=False).sum())

    # 4. Date range analysis
    df_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    invalid_dates_count = int(df_dates.isna().sum())
    min_date = str(df_dates.min())
    max_date = str(df_dates.max())
    total_days_span = int((df_dates.max() - df_dates.min()).days)

    # 5. Invoice & Cancellation inspection
    invoice_str = df["InvoiceNo"].astype(str).str.strip()
    is_cancellation = invoice_str.str.upper().str.startswith("C")
    cancellation_rows = int(is_cancellation.sum())
    unique_invoices_total = int(df["InvoiceNo"].nunique())
    unique_cancellation_invoices = int(df.loc[is_cancellation, "InvoiceNo"].nunique())

    # Check if there are invoices starting with letters other than C (e.g., 'A' for bad debt adjustment)
    starts_with_alpha = invoice_str.str.match(r"^[A-Za-z]")
    other_alpha_invoices = int((starts_with_alpha & ~is_cancellation).sum())

    # 6. Quantity & UnitPrice numeric ranges and anomalies
    qty_stats = {
        "min": float(df["Quantity"].min()),
        "p01": float(np.percentile(df["Quantity"], 1)),
        "median": float(df["Quantity"].median()),
        "mean": round(float(df["Quantity"].mean()), 2),
        "p99": float(np.percentile(df["Quantity"], 99)),
        "max": float(df["Quantity"].max()),
        "negative_or_zero_count": int((df["Quantity"] <= 0).sum()),
        "strictly_negative_count": int((df["Quantity"] < 0).sum()),
        "zero_count": int((df["Quantity"] == 0).sum()),
        "negative_with_c_invoice": int(((df["Quantity"] < 0) & is_cancellation).sum()),
        "negative_without_c_invoice": int(((df["Quantity"] < 0) & ~is_cancellation).sum()),
    }

    price_stats = {
        "min": float(df["UnitPrice"].min()),
        "p01": float(np.percentile(df["UnitPrice"], 1)),
        "median": float(df["UnitPrice"].median()),
        "mean": round(float(df["UnitPrice"].mean()), 2),
        "p99": float(np.percentile(df["UnitPrice"], 99)),
        "max": float(df["UnitPrice"].max()),
        "zero_price_count": int((df["UnitPrice"] == 0).sum()),
        "negative_price_count": int((df["UnitPrice"] < 0).sum()),
    }

    # 7. CustomerID completeness & overlap with anomalies
    has_customer_id = df["CustomerID"].notna()
    rows_with_customer_id = int(has_customer_id.sum())
    rows_missing_customer_id = int((~has_customer_id).sum())
    unique_customers_raw = int(df.loc[has_customer_id, "CustomerID"].nunique())
    cancellations_with_customer_id = int((is_cancellation & has_customer_id).sum())
    cancellations_missing_customer_id = int((is_cancellation & ~has_customer_id).sum())

    # 8. Country distribution
    unique_countries = int(df["Country"].nunique())
    top_5_countries = df["Country"].value_counts().head(5).to_dict()

    # 9. Non-standard StockCodes inspection (purely alphabetic codes like POST, D, M, BANK CHARGES)
    stock_str = df["StockCode"].astype(str).str.strip()
    non_numeric_stock_mask = ~stock_str.str.match(r"^\d{5}[A-Za-z]*$")
    non_standard_stock_rows = int(non_numeric_stock_mask.sum())
    top_non_standard_codes = (
        stock_str[non_numeric_stock_mask].value_counts().head(10).to_dict()
    )

    audit_results = {
        "dataset_overview": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": list(df.columns),
            "dtypes": dtypes_map,
        },
        "missing_values": {
            "counts": missing_counts,
            "percentages": missing_pct,
        },
        "duplicates": {
            "exact_duplicate_rows_to_drop": exact_duplicate_rows,
            "total_rows_involved_in_duplicates": all_duplicated_rows_involved,
        },
        "date_range": {
            "min_invoice_date": min_date,
            "max_invoice_date": max_date,
            "total_days_span": total_days_span,
            "invalid_dates_count": invalid_dates_count,
        },
        "invoices_and_cancellations": {
            "unique_invoices_total": unique_invoices_total,
            "cancellation_rows": cancellation_rows,
            "unique_cancellation_invoices": unique_cancellation_invoices,
            "other_alpha_invoice_rows": other_alpha_invoices,
        },
        "quantity_analysis": qty_stats,
        "unit_price_analysis": price_stats,
        "customer_id_completeness": {
            "rows_with_customer_id": rows_with_customer_id,
            "rows_missing_customer_id": rows_missing_customer_id,
            "missing_customer_id_pct": round((rows_missing_customer_id / total_rows) * 100, 2),
            "unique_customers_raw": unique_customers_raw,
            "cancellations_with_customer_id": cancellations_with_customer_id,
            "cancellations_missing_customer_id": cancellations_missing_customer_id,
        },
        "geography": {
            "unique_countries": unique_countries,
            "top_5_countries_by_row_count": top_5_countries,
        },
        "non_standard_stock_codes": {
            "non_standard_rows_count": non_standard_stock_rows,
            "top_10_non_standard_codes": top_non_standard_codes,
        },
    }
    return audit_results


def save_reports(audit: Dict[str, Any]) -> None:
    """Save the audit results as JSON and readable Markdown in reports/."""
    json_path = REPORTS_DIR / "data_quality_report.json"
    md_path = REPORTS_DIR / "data_quality_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=4)

    ov = audit["dataset_overview"]
    mv = audit["missing_values"]
    dup = audit["duplicates"]
    dr = audit["date_range"]
    inv = audit["invoices_and_cancellations"]
    qty = audit["quantity_analysis"]
    prc = audit["unit_price_analysis"]
    cid = audit["customer_id_completeness"]
    geo = audit["geography"]
    stk = audit["non_standard_stock_codes"]

    md_content = f"""# Stage 2: Raw Data Quality Audit Report — CustomerPulse AI

## 1. Dataset Dimensions & Schema
- **Total Rows:** {ov['total_rows']:,}
- **Total Columns:** {ov['total_columns']}
- **Date Range:** `{dr['min_invoice_date']}` to `{dr['max_invoice_date']}` ({dr['total_days_span']} days)
- **Invalid Dates:** {dr['invalid_dates_count']}

| Column | Raw Data Type | Missing Count | Missing % |
| :--- | :--- | ---: | ---: |
"""
    for col in ov["columns"]:
        md_content += f"| `{col}` | `{ov['dtypes'][col]}` | {mv['counts'][col]:,} | {mv['percentages'][col]:.4f}% |\n"

    md_content += f"""
## 2. Duplicate Records
- **Exact Duplicate Rows (`keep='first'`):** {dup['exact_duplicate_rows_to_drop']:,}
- **Total Rows Involved in Duplication:** {dup['total_rows_involved_in_duplicates']:,}

## 3. Invoice & Cancellation Anomalies
- **Total Unique Invoices:** {inv['unique_invoices_total']:,}
- **Cancellation Rows (`InvoiceNo` starts with 'C'):** {inv['cancellation_rows']:,} across {inv['unique_cancellation_invoices']:,} unique invoices
- **Other Alphabetic Invoice Rows (e.g., Bad Debt 'A'):** {inv['other_alpha_invoice_rows']:,}

## 4. Quantity & UnitPrice Distribution
| Metric | Quantity | UnitPrice (£) |
| :--- | ---: | ---: |
| **Minimum** | {qty['min']:,.0f} | {prc['min']:,.2f} |
| **1st Percentile** | {qty['p01']:,.0f} | {prc['p01']:,.2f} |
| **Median** | {qty['median']:,.0f} | {prc['median']:,.2f} |
| **Mean** | {qty['mean']:,.2f} | {prc['mean']:,.2f} |
| **99th Percentile** | {qty['p99']:,.0f} | {prc['p99']:,.2f} |
| **Maximum** | {qty['max']:,.0f} | {prc['max']:,.2f} |

### Specific Quantity & Price Anomalies
- **Strictly Negative Quantity Rows (`Quantity < 0`):** {qty['strictly_negative_count']:,}
  - *Explained by Cancellation Invoices (`'C'`):* {qty['negative_with_c_invoice']:,}
  - *Negative WITHOUT `'C'` Invoice (Stock Write-offs/Damages):* {qty['negative_without_c_invoice']:,}
- **Zero Price Rows (`UnitPrice == 0`):** {prc['zero_price_count']:,}
- **Negative Price Rows (`UnitPrice < 0`):** {prc['negative_price_count']:,}

## 5. Customer Identifier Completeness
- **Rows with Valid `CustomerID`:** {cid['rows_with_customer_id']:,}
- **Rows Missing `CustomerID`:** {cid['rows_missing_customer_id']:,} ({cid['missing_customer_id_pct']}%)
- **Total Unique Customers (Raw):** {cid['unique_customers_raw']:,}
- **Cancellation Rows with `CustomerID`:** {cid['cancellations_with_customer_id']:,}
- **Cancellation Rows Missing `CustomerID`:** {cid['cancellations_missing_customer_id']:,}

## 6. Geography & Non-Standard StockCodes
- **Unique Countries:** {geo['unique_countries']}
- **Top 5 Countries (by row count):** {geo['top_5_countries_by_row_count']}
- **Non-Standard `StockCode` Rows (Postage, Manual, Fees, etc.):** {stk['non_standard_rows_count']:,}
- **Top Non-Standard Codes:** `{stk['top_10_non_standard_codes']}`
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Saved JSON report to     : {json_path}")
    logger.info(f"Saved Markdown report to : {md_path}")


def print_console_summary(audit: Dict[str, Any]) -> None:
    """Print a clean, readable summary to the terminal."""
    ov = audit["dataset_overview"]
    mv = audit["missing_values"]
    dup = audit["duplicates"]
    dr = audit["date_range"]
    inv = audit["invoices_and_cancellations"]
    qty = audit["quantity_analysis"]
    prc = audit["unit_price_analysis"]
    cid = audit["customer_id_completeness"]
    stk = audit["non_standard_stock_codes"]

    print("\n" + "=" * 70)
    print("  CUSTOMERPULSE AI - STAGE 2 RAW DATASET INSPECTION SUMMARY")
    print("=" * 70)
    print(f"Total Rows                  : {ov['total_rows']:,}")
    print(f"Total Columns               : {ov['total_columns']} -> {ov['columns']}")
    print(f"Date Range                  : {dr['min_invoice_date']} to {dr['max_invoice_date']} ({dr['total_days_span']} days)")
    print("-" * 70)
    print("Missing Values:")
    for col, count in mv["counts"].items():
        if count > 0:
            print(f"  - {col:22} : {count:,} ({mv['percentages'][col]:.2f}%)")
    print("-" * 70)
    print(f"Exact Duplicate Rows        : {dup['exact_duplicate_rows_to_drop']:,}")
    print(f"Cancellation Rows ('C')     : {inv['cancellation_rows']:,} ({inv['unique_cancellation_invoices']:,} invoices)")
    print(f"Other Alpha Invoices ('A')  : {inv['other_alpha_invoice_rows']:,}")
    print(f"Negative Quantities (< 0)   : {qty['strictly_negative_count']:,}")
    print(f"  -> With 'C' Invoice       : {qty['negative_with_c_invoice']:,}")
    print(f"  -> Without 'C' Invoice    : {qty['negative_without_c_invoice']:,}")
    print(f"Zero UnitPrice (== 0)       : {prc['zero_price_count']:,}")
    print(f"Negative UnitPrice (< 0)    : {prc['negative_price_count']:,}")
    print(f"Non-Standard StockCodes     : {stk['non_standard_rows_count']:,}")
    print(f"Unique Customers (Raw)      : {cid['unique_customers_raw']:,}")
    print("=" * 70)


def main() -> None:
    df = load_raw_dataset(RAW_DATASET_PATH)
    audit_results = run_data_quality_audit(df)
    print_console_summary(audit_results)
    save_reports(audit_results)


if __name__ == "__main__":
    main()

# Stage 2: Raw Data Quality Audit Report — CustomerPulse AI

## 1. Dataset Dimensions & Schema
- **Total Rows:** 541,909
- **Total Columns:** 8
- **Date Range:** `2010-12-01 08:26:00` to `2011-12-09 12:50:00` (373 days)
- **Invalid Dates:** 0

| Column | Raw Data Type | Missing Count | Missing % |
| :--- | :--- | ---: | ---: |
| `InvoiceNo` | `object` | 0 | 0.0000% |
| `StockCode` | `object` | 0 | 0.0000% |
| `Description` | `object` | 1,454 | 0.2683% |
| `Quantity` | `int64` | 0 | 0.0000% |
| `InvoiceDate` | `datetime64[us]` | 0 | 0.0000% |
| `UnitPrice` | `float64` | 0 | 0.0000% |
| `CustomerID` | `float64` | 135,080 | 24.9267% |
| `Country` | `str` | 0 | 0.0000% |

## 2. Duplicate Records
- **Exact Duplicate Rows (`keep='first'`):** 5,268
- **Total Rows Involved in Duplication:** 10,147

## 3. Invoice & Cancellation Anomalies
- **Total Unique Invoices:** 25,900
- **Cancellation Rows (`InvoiceNo` starts with 'C'):** 9,288 across 3,836 unique invoices
- **Other Alphabetic Invoice Rows (e.g., Bad Debt 'A'):** 3

## 4. Quantity & UnitPrice Distribution
| Metric | Quantity | UnitPrice (£) |
| :--- | ---: | ---: |
| **Minimum** | -80,995 | -11,062.06 |
| **1st Percentile** | -2 | 0.19 |
| **Median** | 3 | 2.08 |
| **Mean** | 9.55 | 4.61 |
| **99th Percentile** | 100 | 18.00 |
| **Maximum** | 80,995 | 38,970.00 |

### Specific Quantity & Price Anomalies
- **Strictly Negative Quantity Rows (`Quantity < 0`):** 10,624
  - *Explained by Cancellation Invoices (`'C'`):* 9,288
  - *Negative WITHOUT `'C'` Invoice (Stock Write-offs/Damages):* 1,336
- **Zero Price Rows (`UnitPrice == 0`):** 2,515
- **Negative Price Rows (`UnitPrice < 0`):** 2

## 5. Customer Identifier Completeness
- **Rows with Valid `CustomerID`:** 406,829
- **Rows Missing `CustomerID`:** 135,080 (24.93%)
- **Total Unique Customers (Raw):** 4,372
- **Cancellation Rows with `CustomerID`:** 8,905
- **Cancellation Rows Missing `CustomerID`:** 383

## 6. Geography & Non-Standard StockCodes
- **Unique Countries:** 38
- **Top 5 Countries (by row count):** {'United Kingdom': 495478, 'Germany': 9495, 'France': 8557, 'EIRE': 8196, 'Spain': 2533}
- **Non-Standard `StockCode` Rows (Postage, Manual, Fees, etc.):** 2,995
- **Top Non-Standard Codes:** `{'POST': 1256, 'DOT': 710, 'M': 571, 'C2': 144, 'D': 77, 'S': 63, 'BANK CHARGES': 37, 'AMAZONFEE': 34, 'CRUK': 16, 'DCGSSGIRL': 13}`

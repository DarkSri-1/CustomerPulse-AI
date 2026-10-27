# Stage 3: Data Cleaning & Preprocessing Report — CustomerPulse AI

## 1. Row Waterfall Summary
| Pipeline Step | Row Count |
| :--- | ---: |
| 1. Raw Dataset Loaded | 541,909 |
| 2. Exact Duplicates Dropped | -5,268 |
| 3. Bad Debt Rows Dropped ('A') | -3 |
| 4. Invalid Qty / Zero Price Dropped | -2,515 |
| 5. Non-Product Codes Dropped | -2,912 |
| 6. Cleaned Store Transactions | 531,211 |
| 7. Guest Checkouts (Store Only) | 131,555 |
| 8. Identified Customer Transactions | 399,656 |

## 2. Financial Reconciliation
| Metric | Store-Level (All Checkouts) | Customer-Level (Identified Only) |
| :--- | ---: | ---: |
| **Total Rows** | 531,211 | 399,656 |
| **Gross Sales (£)** | £10,430,132.89 | £8,737,228.64 |
| **Returns & Cancellations (£)** | £-606,869.03 | £-458,910.40 |
| **Net Revenue (£)** | £9,823,263.86 | £8,278,318.24 |
| **Cancellation Rate** | 1.64% | 2.13% |

- **Unique Identified Customers (Total / Active):** 4,363 / 4,334
- **Return-Only Customers:** 29

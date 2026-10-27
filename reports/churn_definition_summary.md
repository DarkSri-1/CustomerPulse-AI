# Stage 7: Customer Churn Definition & Leak-Free Cohort Summary

## 1. Temporal Window Specification
- **Historical Observation Window:** `2010-12-01 08:26:00` to `2011-09-10 00:00:00` (282 days)
- **Prediction Cutoff Date:** `2011-09-10 00:00:00`
- **Future Outcome Window:** `2011-09-10 00:00:00` to `2011-12-09 12:50:00` (90 days)
- **Churn Definition (`is_churned = 1`):** An active customer from the observation window who places **zero qualifying orders** (`Quantity > 0`, non-cancelled) during the 90-day future outcome window.

## 2. Cohort Accounting & Right-Censoring Treatment
| Cohort Category | Customer Count | Share / Note |
| :--- | ---: | :--- |
| **Total Active Customers (Full 373-Day Dataset)** | 4,334 | 100.0% of all-time buyers |
| **Less: Right-Censored New Buyers (Acquired on/after `2011-09-10`)** | -966 | Excluded (insufficient future window) |
| **Eligible Modeling Cohort (Active prior to `2011-09-10`)** | **3,368** | **100.0% of modeling dataset** |
|   -> **Retained (`is_churned = 0`)** | 1,919 | 56.98% |
|   -> **Churned (`is_churned = 1`)** | 1,449 | 43.02% |

## 3. Observation Feature Means: Retained (`0`) vs. Churned (`1`)
| Pre-Cutoff Feature | Retained (`is_churned = 0`) | Churned (`is_churned = 1`) |
| :--- | ---: | ---: |
| `recency_days` | 72.90 | 122.47 |
| `frequency_orders` | 4.77 | 1.84 |
| `monetary_gross` | 2,263.04 | 701.66 |
| `avg_order_value` | 391.53 | 401.88 |
| `unique_products` | 65.47 | 29.14 |
| `tenure_days` | 191.66 | 168.84 |
| `orders_last_30d` | 0.56 | 0.18 |
| `orders_last_60d` | 1.12 | 0.34 |

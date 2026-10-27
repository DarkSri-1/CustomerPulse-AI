# Stage 9: Model Explainability Report (SHAP) — CustomerPulse AI

## 1. Explainability Overview
- **Explained Model:** `Random_Forest` (`v1.0.0`)
- **SHAP Baseline Expected Value:** `0.5000`
- **Cohort Explained:** `3,381` active pre-cutoff customers

## 2. Top 10 Global Churn Features (Ranked by Mean |SHAP|)
| Rank | Feature | Display Name | Mean |SHAP| | Share (%) | Feature-SHAP Corr | Directional Influence |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- |
| 1 | `frequency_orders` | Order Frequency (Total Orders) | 0.0412 | 13.12% | -0.5874 | Higher value DECREASES churn risk (-) |
| 2 | `total_line_items` | Total Line Items Purchased | 0.0409 | 13.04% | -0.4437 | Higher value DECREASES churn risk (-) |
| 3 | `monetary_gross` | Gross Spend (£) | 0.0330 | 10.52% | -0.3189 | Higher value DECREASES churn risk (-) |
| 4 | `avg_days_between_orders` | Avg Days Between Orders | 0.0295 | 9.41% | -0.5205 | Higher value DECREASES churn risk (-) |
| 5 | `monetary_net` | Net Revenue (£) | 0.0267 | 8.51% | -0.3183 | Higher value DECREASES churn risk (-) |
| 6 | `recency_days` | Recency (Days Since Last Order) | 0.0266 | 8.47% | +0.8780 | Higher value INCREASES churn risk (+) |
| 7 | `total_units` | Total Units Purchased | 0.0263 | 8.38% | -0.2944 | Higher value DECREASES churn risk (-) |
| 8 | `spend_last_60d` | Spend in Last 60 Days (£) | 0.0225 | 7.17% | -0.2380 | Higher value DECREASES churn risk (-) |
| 9 | `unique_products` | Product Diversity (Unique SKUs) | 0.0155 | 4.95% | -0.5907 | Higher value DECREASES churn risk (-) |
| 10 | `avg_order_value` | Average Order Value (£) | 0.0133 | 4.22% | +0.1903 | Higher value INCREASES churn risk (+) |

## 3. Individual Customer Prediction Examples (Privacy-Masked IDs)
| Profile | Masked ID | Churn Prob | Risk Tier | Recency (d) | Orders | Spend (£) | Top SHAP Driver 1 | Top SHAP Driver 2 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| High Risk Example | `CUST-***68` | 85.7% | High Risk | 126 | 1 | £76.32 | `total_line_items=1 (+0.07 SHAP, +risk)` | `unique_products=1 (+0.05 SHAP, +risk)` |
| Medium Risk Example | `CUST-***54` | 58.3% | Medium Risk | 36 | 1 | £300.36 | `spend_last_60d=300.4 (-0.04 SHAP, -risk)` | `avg_days_between_orders=0.0 (+0.03 SHAP, +risk)` |
| Low Risk Example | `CUST-***96` | 0.1% | Low Risk | 3 | 14 | £6,579.62 | `frequency_orders=14 (-0.09 SHAP, -risk)` | `monetary_gross=6579.6 (-0.07 SHAP, -risk)` |

## 4. Predictive Association vs. Causality & Methodological Limitations
1. **Predictive Association vs. Causation:** High SHAP importance on `recency_days`, `frequency_orders`, `tenure_days`, or `unique_products` indicates strong predictive association with future 90-day inactivity—not a guaranteed causal mechanism.
2. **Collinearity Across Spend & Volume Metrics:** Because `monetary_gross`, `monetary_net`, `total_line_items`, and `total_units` move together, tree ensembles distribute SHAP credit across them.
3. **Privacy Safeguards:** Operational logs and static reports mask customer identifiers (`CUST-***XX`).

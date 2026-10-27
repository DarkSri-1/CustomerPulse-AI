# Stage 8: Machine Learning Model Evaluation Summary — CustomerPulse AI

## 1. Selected Production Model
- **Winning Model:** `Random_Forest` (Version `v1.0.0`)
- **Selection Criterion:** Highest Holdout Test **PR-AUC** & **F1-Score**
- **Optimized Decision Threshold (from 5-Fold OOF Validation):** `0.3450`
- **Train / Holdout Split:** `2,694` Train Customers (80%) | `674` Holdout Test Customers (20%)

## 2. Holdout Test Set Performance Comparison (Unseen Customers)
| Model | Threshold | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | TP | FP | TN | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy_Baseline | 0.43 | 0.4303 | 0.5000 | 0.6017 | 1.0000 | 0.4303 | 290 | 384 | 0 | 0 |
| Logistic_Regression | 0.45 | 0.6146 | 0.7084 | 0.6552 | 0.7897 | 0.5599 | 229 | 180 | 204 | 61 |
|  **Random_Forest ** | 0.34 | 0.6258 | 0.7244 | 0.6667 | 0.8690 | 0.5408 | 252 | 214 | 170 | 38 |
| XGBoost | 0.34 | 0.6259 | 0.7220 | 0.6587 | 0.8517 | 0.5370 | 247 | 213 | 171 | 43 |

## 3. 5-Fold Stratified Cross-Validation (Out-of-Fold Training Set)
| Model | Threshold | OOF PR-AUC | OOF ROC-AUC | OOF F1-Score | OOF Recall | OOF Precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy_Baseline | 0.43 | 0.4300 | 0.4997 | 0.2722 | 0.1993 | 0.4294 |
| Logistic_Regression | 0.45 | 0.6322 | 0.7463 | 0.6827 | 0.8430 | 0.5737 |
| Random_Forest | 0.34 | 0.6314 | 0.7485 | 0.6924 | 0.9137 | 0.5574 |
| XGBoost | 0.34 | 0.6305 | 0.7465 | 0.6919 | 0.9068 | 0.5593 |

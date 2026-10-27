"""
Stage 8: Model Inference & Churn Prediction Module for CustomerPulse AI.
Loads the versioned Scikit-Learn/XGBoost pipeline artifact, validates input schemas,
generates customer churn probabilities and risk tiers, and saves results to MySQL.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CHURN_DATASET_PATH,
    CHURN_PREDICTIONS_PATH,
    CHURN_CUTOFF_DATE,
    MODEL_ARTIFACT_PATH,
    LOGS_DIR,
)
from src.database import save_churn_predictions, test_database_connection

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage8_predictions.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_model_bundle() -> Dict[str, Any]:
    """Load the serialized model pipeline and metadata bundle from disk."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {MODEL_ARTIFACT_PATH}. "
            "Run `python -m src.model_training` first."
        )
    bundle = joblib.load(MODEL_ARTIFACT_PATH)
    return bundle


def assign_risk_tier(prob: float, threshold: float) -> str:
    """Map churn probability to a business-actionable risk tier."""
    if prob >= max(0.60, threshold + 0.12):
        return "High Risk"
    elif prob >= threshold:
        return "Medium Risk"
    return "Low Risk"


def derive_customer_risk_drivers(row: pd.Series, medians: Dict[str, float]) -> Tuple[str, str, str]:
    """
    Provide human-readable behavioral risk indicators for each customer.
    (Note: In Stage 9, exact per-customer SHAP values will further enrich this!)
    """
    reasons: List[str] = []

    if row["recency_days"] > medians["recency_days"]:
        reasons.append(f"High inactivity ({int(row['recency_days'])}d since last order)")
    else:
        reasons.append(f"Recent activity ({int(row['recency_days'])}d since last order)")

    if row["frequency_orders"] <= 1:
        reasons.append("Single-order buyer (1 order prior to cutoff)")
    elif row["frequency_orders"] < medians["frequency_orders"]:
        reasons.append(f"Low order frequency ({int(row['frequency_orders'])} orders)")
    else:
        reasons.append(f"Repeat buyer ({int(row['frequency_orders'])} orders)")

    if row["orders_last_60d"] == 0:
        reasons.append("Zero orders in last 60 days before cutoff")
    elif row["monetary_gross"] < medians["monetary_gross"]:
        reasons.append(f"Below-median spend (£{row['monetary_gross']:,.0f})")
    else:
        reasons.append(f"Strong historical spend (£{row['monetary_gross']:,.0f})")

    while len(reasons) < 3:
        reasons.append("Standard purchasing profile")

    return reasons[0], reasons[1], reasons[2]


def predict_single_customer(payload: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Score a single customer dictionary (used by FastAPI POST /predict/churn in Stage 10).
    """
    if bundle is None:
        bundle = load_model_bundle()

    pipeline = bundle["pipeline"]
    threshold = float(bundle["decision_threshold"])
    req_cols = bundle["numeric_features"] + bundle["categorical_features"]

    # Fill optional derived fields defensively if caller provides core features
    input_df = pd.DataFrame([payload])
    for col in req_cols:
        if col not in input_df.columns:
            input_df[col] = 0.0 if col in bundle["numeric_features"] else "United Kingdom"

    input_df = input_df[req_cols]
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    pred_label = int(prob >= threshold)
    risk_tier = assign_risk_tier(prob, threshold)

    return {
        "churn_probability": round(prob, 4),
        "predicted_churn": pred_label,
        "decision_threshold": round(threshold, 4),
        "risk_tier": risk_tier,
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
    }


def generate_and_save_all_predictions() -> pd.DataFrame:
    """
    Score the full 3,381-customer Stage 7 cohort, save to CSV, and push to MySQL.
    """
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    threshold = float(bundle["decision_threshold"])
    model_name = bundle["model_name"]
    model_version = bundle["model_version"]
    req_cols = bundle["numeric_features"] + bundle["categorical_features"]

    df = pd.read_csv(CHURN_DATASET_PATH, dtype={"customer_id": str})
    X = df[req_cols].copy()

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    medians = {
        "recency_days": float(df["recency_days"].median()),
        "frequency_orders": float(df["frequency_orders"].median()),
        "monetary_gross": float(df["monetary_gross"].median()),
    }

    drivers = df.apply(lambda r: derive_customer_risk_drivers(r, medians), axis=1)

    df_preds = pd.DataFrame({
        "customer_id": df["customer_id"],
        "observation_cutoff_date": df.get("observation_cutoff_date", CHURN_CUTOFF_DATE),
        "churn_probability": np.round(probs, 4),
        "predicted_churn": preds,
        "actual_churn": df["is_churned"].astype(int),
        "risk_tier": [assign_risk_tier(float(p), threshold) for p in probs],
        "model_name": model_name,
        "model_version": model_version,
        "top_driver_1": [d[0] for d in drivers],
        "top_driver_2": [d[1] for d in drivers],
        "top_driver_3": [d[2] for d in drivers],
    })

    df_preds.to_csv(CHURN_PREDICTIONS_PATH, index=False, encoding="utf-8")
    logger.info(f"[OK] Saved {len(df_preds):,} customer churn predictions to {CHURN_PREDICTIONS_PATH}")

    if test_database_connection():
        save_churn_predictions(df_preds)
    else:
        logger.warning("MySQL connection unavailable; skipped saving churn_predictions to DB.")

    return df_preds


def main() -> None:
    df_preds = generate_and_save_all_predictions()
    tier_counts = df_preds["risk_tier"].value_counts()

    print("\n" + "=" * 72)
    print("  CUSTOMERPULSE AI - STAGE 8 CHURN PREDICTIONS SUMMARY")
    print("=" * 72)
    print(f"Total Customers Scored        : {len(df_preds):,}")
    print(f"Model Used                    : {df_preds['model_name'].iloc[0]} ({df_preds['model_version'].iloc[0]})")
    print(f"Predicted Churners (Label=1)  : {int((df_preds['predicted_churn'] == 1).sum()):,} ({(df_preds['predicted_churn'] == 1).mean() * 100:.2f}%)")
    print(f"Actual Churners (Label=1)     : {int((df_preds['actual_churn'] == 1).sum()):,} ({(df_preds['actual_churn'] == 1).mean() * 100:.2f}%)")
    print("-" * 72)
    print("Customer Distribution by Risk Tier:")
    for tier, count in tier_counts.items():
        pct = (count / len(df_preds)) * 100
        print(f"  - {tier:15} : {count:,} customers ({pct:.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()

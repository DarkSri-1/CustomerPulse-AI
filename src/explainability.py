"""
Stage 9: Model Explainability Engine (SHAP) for CustomerPulse AI.
Computes global and local SHAP feature attributions, generates visual explainability
charts, updates customer predictions with exact SHAP drivers, and logs privacy-safe reports.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from src.config import (
    CHURN_DATASET_PATH,
    CHURN_PREDICTIONS_PATH,
    MODEL_ARTIFACT_PATH,
    SHAP_GLOBAL_IMPORTANCE_PATH,
    SHAP_REPORT_JSON_PATH,
    SHAP_REPORT_MD_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)
from src.database import save_churn_predictions, test_database_connection

LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage9_explainability.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Friendly human-readable labels for features
FEATURE_LABELS: Dict[str, str] = {
    "recency_days": "Recency (Days Since Last Order)",
    "frequency_orders": "Order Frequency (Total Orders)",
    "total_line_items": "Total Line Items Purchased",
    "total_units": "Total Units Purchased",
    "monetary_gross": "Gross Spend (£)",
    "cancellations_amount": "Cancelled/Returned Amount (£)",
    "monetary_net": "Net Revenue (£)",
    "avg_order_value": "Average Order Value (£)",
    "avg_days_between_orders": "Avg Days Between Orders",
    "unique_products": "Product Diversity (Unique SKUs)",
    "cancellation_rate_pct": "Cancellation Rate (%)",
    "tenure_days": "Customer Tenure (Days)",
    "orders_last_30d": "Orders in Last 30 Days",
    "orders_last_60d": "Orders in Last 60 Days",
    "spend_last_30d": "Spend in Last 30 Days (£)",
    "spend_last_60d": "Spend in Last 60 Days (£)",
    "spend_ratio_last_30d": "30-Day Spend Momentum Ratio",
}


def mask_customer_id(cid: str) -> str:
    """
    Mask customer identifier for logs and public charts to protect customer privacy.
    Example: '17850' -> 'CUST-***50'
    """
    cid_str = str(cid).strip()
    if len(cid_str) <= 2:
        return "CUST-***"
    return f"CUST-***{cid_str[-2:]}"


def compute_shap_matrix(
    pipeline: Any,
    X_raw: pd.DataFrame,
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """
    Transform raw features via the fitted pipeline preprocessor and compute
    SHAP values for the positive class (is_churned = 1).
    Includes resilient handling for TreeExplainer, LinearExplainer, and Permutation fallback.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_trans = preprocessor.transform(X_raw)
    if not isinstance(X_trans, pd.DataFrame):
        feat_names = list(preprocessor.get_feature_names_out())
        X_trans = pd.DataFrame(X_trans, columns=feat_names, index=X_raw.index)

    logger.info(
        f"Computing SHAP values for {len(X_trans):,} customers across "
        f"{X_trans.shape[1]} transformed features using {classifier.__class__.__name__}..."
    )

    try:
        # Attempt fast TreeExplainer for XGBoost / RandomForest
        explainer = shap.TreeExplainer(classifier)
        raw_shap = explainer.shap_values(X_trans)
        expected_val = explainer.expected_value

        # RandomForest returns 3D array (n_samples, n_features, 2) or list of 2 arrays
        if isinstance(raw_shap, list):
            shap_matrix = np.asarray(raw_shap[1], dtype=float)
            base_val = float(expected_val[1]) if hasattr(expected_val, "__len__") else float(expected_val)
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_matrix = raw_shap[:, :, 1].astype(float)
            base_val = float(expected_val[1]) if hasattr(expected_val, "__len__") else float(expected_val)
        else:
            shap_matrix = np.asarray(raw_shap, dtype=float)
            base_val = float(expected_val[0]) if hasattr(expected_val, "__len__") else float(expected_val)

    except Exception as tree_exc:
        logger.info(
            f"TreeExplainer fallback triggered ({tree_exc.__class__.__name__}). "
            "Using model-agnostic probability SHAP Explainer with background sample..."
        )
        bg_sample = shap.sample(X_trans, min(150, len(X_trans)), random_state=RANDOM_SEED)
        explainer = shap.Explainer(lambda data: classifier.predict_proba(data)[:, 1], bg_sample)
        explanation = explainer(X_trans)
        shap_matrix = np.asarray(explanation.values, dtype=float)
        base_val = float(np.mean(explanation.base_values))

    return shap_matrix, base_val, X_trans


def build_global_importance_table(
    shap_matrix: np.ndarray,
    X_trans: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean(|SHAP|) and directional correlation with SHAP value for every feature.
    """
    feature_names = list(X_trans.columns)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    total_importance = float(np.sum(mean_abs_shap)) or 1.0

    rows = []
    for idx, feat in enumerate(feature_names):
        feat_vals = X_trans[feat].values
        shap_vals = shap_matrix[:, idx]
        if np.std(feat_vals) > 1e-9 and np.std(shap_vals) > 1e-9:
            corr = float(np.corrcoef(feat_vals, shap_vals)[0, 1])
        else:
            corr = 0.0

        direction = (
            "Higher value INCREASES churn risk (+)"
            if corr >= 0
            else "Higher value DECREASES churn risk (-)"
        )
        rows.append(
            {
                "feature": feat,
                "display_name": FEATURE_LABELS.get(feat, feat),
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                "relative_share_pct": round(float((mean_abs_shap[idx] / total_importance) * 100), 2),
                "feature_shap_correlation": round(corr, 4),
                "directional_effect": direction,
            }
        )

    df_imp = (
        pd.DataFrame(rows)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    df_imp["rank"] = df_imp.index + 1
    return df_imp


def format_customer_shap_drivers(
    df_raw: pd.DataFrame,
    shap_matrix: np.ndarray,
    feature_names: List[str],
) -> List[Tuple[str, str, str]]:
    """
    For each customer, extract the top 3 numeric features with the largest absolute
    SHAP contribution and format readable explanation strings for MySQL & dashboard display.
    """
    numeric_cols_set = set(df_raw.columns)
    drivers_list: List[Tuple[str, str, str]] = []

    for row_idx in range(len(df_raw)):
        row_shap = shap_matrix[row_idx]
        # Rank feature indices by descending |SHAP|
        sorted_indices = np.argsort(np.abs(row_shap))[::-1]
        top_strings: List[str] = []

        for feat_idx in sorted_indices:
            feat_name = feature_names[feat_idx]
            shap_val = float(row_shap[feat_idx])
            direction_arrow = "+risk" if shap_val >= 0 else "-risk"

            if feat_name in numeric_cols_set and feat_name != "country":
                raw_val = df_raw.iloc[row_idx][feat_name]
                if isinstance(raw_val, (int, np.integer)) or feat_name in {
                    "recency_days",
                    "frequency_orders",
                    "total_line_items",
                    "total_units",
                    "unique_products",
                    "tenure_days",
                    "orders_last_30d",
                    "orders_last_60d",
                }:
                    val_str = f"{int(round(float(raw_val)))}"
                else:
                    val_str = f"{float(raw_val):.1f}"
                top_strings.append(f"{feat_name}={val_str} ({shap_val:+.2f} SHAP, {direction_arrow})")
            else:
                top_strings.append(f"{feat_name} ({shap_val:+.2f} SHAP, {direction_arrow})")

            if len(top_strings) == 3:
                break

        while len(top_strings) < 3:
            top_strings.append("Baseline profile (0.00 SHAP)")

        drivers_list.append((top_strings[0], top_strings[1], top_strings[2]))

    return drivers_list


def generate_shap_visualizations(
    shap_matrix: np.ndarray,
    X_trans: pd.DataFrame,
    df_imp: pd.DataFrame,
    df_raw: pd.DataFrame,
    df_preds: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Create and save the 3 Stage 9 SHAP visualizations in reports/figures/:
      - 08_shap_global_importance_bar.png
      - 09_shap_directional_beeswarm.png
      - 10_shap_individual_customer_profiles.png
    Returns privacy-masked example profiles for High, Medium, and Low risk customers.
    """
    # 1. Chart 08: Global Feature Importance Bar Plot (Top 12 Features)
    top12 = df_imp.head(12).iloc[::-1]
    colors = [
        "#d62728" if c >= 0 else "#1f77b4"
        for c in top12["feature_shap_correlation"]
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(top12["display_name"], top12["mean_abs_shap"], color=colors)
    ax.set_title(
        "Global SHAP Feature Importance — Top 12 Churn Drivers\n"
        "(Red = Higher Feature Value Increases Churn Risk | Blue = Higher Value Reduces Churn Risk)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)")
    for bar, share in zip(bars, top12["relative_share_pct"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, float(top12["mean_abs_shap"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_shap_global_importance_bar.png")
    plt.close(fig)

    # 2. Chart 09: Directional SHAP Summary Beeswarm Plot
    fig = plt.figure(figsize=(11, 6.5))
    shap.summary_plot(
        shap_matrix,
        X_trans,
        max_display=12,
        show=False,
        plot_size=(11, 6.5),
    )
    plt.title(
        "SHAP Directional Beeswarm Summary (How Feature Values Influence Churn Risk)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_shap_directional_beeswarm.png", bbox_inches="tight")
    plt.close("all")

    # 3. Chart 10: Local Individual Customer Explanations (High, Medium, Low Risk Examples)
    high_idx = int(
        df_preds[df_preds["risk_tier"] == "High Risk"]
        .sort_values("churn_probability", ascending=False)
        .index[0]
    )
    med_candidates = df_preds[df_preds["risk_tier"] == "Medium Risk"]
    if len(med_candidates) > 0:
        med_idx = int(med_candidates.iloc[len(med_candidates) // 2].name)
    else:
        med_idx = int(df_preds.sort_values("churn_probability").iloc[len(df_preds) // 2].name)

    low_idx = int(
        df_preds[df_preds["risk_tier"] == "Low Risk"]
        .sort_values("churn_probability", ascending=True)
        .index[0]
    )

    sample_indices = [
        ("High Risk Example", high_idx),
        ("Medium Risk Example", med_idx),
        ("Low Risk Example", low_idx),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    feature_names = list(X_trans.columns)
    example_profiles: List[Dict[str, Any]] = []

    for ax, (label_title, idx) in zip(axes, sample_indices):
        cust_row = df_raw.iloc[idx]
        pred_row = df_preds.iloc[idx]
        masked_id = mask_customer_id(cust_row["customer_id"])
        prob = float(pred_row["churn_probability"])
        row_shap = shap_matrix[idx]

        # Top 6 features for this specific customer
        top6_idx = np.argsort(np.abs(row_shap))[::-1][:6][::-1]
        bar_labels = []
        bar_vals = []
        for f_i in top6_idx:
            fname = feature_names[f_i]
            sval = float(row_shap[f_i])
            if fname in cust_row and fname != "country":
                bar_labels.append(f"{fname}={cust_row[fname]:.0f}" if isinstance(cust_row[fname], (int, float, np.number)) else fname)
            else:
                bar_labels.append(fname)
            bar_vals.append(sval)

        bar_colors = ["#d62728" if v >= 0 else "#2ca02c" for v in bar_vals]
        ax.barh(bar_labels, bar_vals, color=bar_colors)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(
            f"{label_title} ({masked_id})\nPred Churn Prob: {prob:.1%}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Local SHAP Contribution (Red=+Churn, Green=-Churn)")

        example_profiles.append(
            {
                "profile_type": label_title,
                "masked_customer_id": masked_id,
                "churn_probability": round(prob, 4),
                "risk_tier": str(pred_row["risk_tier"]),
                "recency_days": int(cust_row["recency_days"]),
                "frequency_orders": int(cust_row["frequency_orders"]),
                "monetary_gross_gbp": round(float(cust_row["monetary_gross"]), 2),
                "top_driver_1": str(pred_row["top_driver_1"]),
                "top_driver_2": str(pred_row["top_driver_2"]),
                "top_driver_3": str(pred_row["top_driver_3"]),
            }
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_shap_individual_customer_profiles.png")
    plt.close(fig)

    logger.info(f"Saved 3 Stage 9 SHAP figures to {FIGURES_DIR}")
    return example_profiles


def save_explainability_reports(
    model_name: str,
    model_version: str,
    base_value: float,
    df_imp: pd.DataFrame,
    example_profiles: List[Dict[str, Any]],
) -> None:
    """Save the Stage 9 SHAP summary report as JSON and Markdown."""
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "shap_expected_base_value": round(float(base_value), 4),
        "global_feature_importance": df_imp.to_dict(orient="records"),
        "individual_customer_examples": example_profiles,
        "methodological_notes": {
            "association_vs_causality": (
                "SHAP values quantify how strongly each historical feature influences the model's "
                "predicted churn score (predictive association), NOT causal impact."
            ),
            "collinearity_limitation": (
                "Correlated behavioral features (e.g., monetary_gross, monetary_net, total_units, "
                "and frequency_orders) share attribution credit across trees."
            ),
            "privacy_protection": (
                "Customer IDs in logs and static reports are masked (CUST-***XX) to avoid exposing "
                "unnecessary customer identifiers."
            ),
        },
    }

    with open(SHAP_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    md = f"""# Stage 9: Model Explainability Report (SHAP) — CustomerPulse AI

## 1. Explainability Overview
- **Explained Model:** `{model_name}` (`{model_version}`)
- **SHAP Baseline Expected Value:** `{base_value:.4f}`
- **Cohort Explained:** `3,381` active pre-cutoff customers

## 2. Top 10 Global Churn Features (Ranked by Mean |SHAP|)
| Rank | Feature | Display Name | Mean |SHAP| | Share (%) | Feature-SHAP Corr | Directional Influence |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- |
"""
    for row in df_imp.head(10).to_dict(orient="records"):
        md += (
            f"| {row['rank']} | `{row['feature']}` | {row['display_name']} | "
            f"{row['mean_abs_shap']:.4f} | {row['relative_share_pct']:.2f}% | "
            f"{row['feature_shap_correlation']:+.4f} | {row['directional_effect']} |\n"
        )

    md += """
## 3. Individual Customer Prediction Examples (Privacy-Masked IDs)
| Profile | Masked ID | Churn Prob | Risk Tier | Recency (d) | Orders | Spend (£) | Top SHAP Driver 1 | Top SHAP Driver 2 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
"""
    for ex in example_profiles:
        md += (
            f"| {ex['profile_type']} | `{ex['masked_customer_id']}` | "
            f"{ex['churn_probability']:.1%} | {ex['risk_tier']} | "
            f"{ex['recency_days']} | {ex['frequency_orders']} | "
            f"£{ex['monetary_gross_gbp']:,.2f} | `{ex['top_driver_1']}` | `{ex['top_driver_2']}` |\n"
        )

    md += """
## 4. Predictive Association vs. Causality & Methodological Limitations
1. **Predictive Association vs. Causation:** High SHAP importance on `recency_days`, `frequency_orders`, `tenure_days`, or `unique_products` indicates strong predictive association with future 90-day inactivity—not a guaranteed causal mechanism.
2. **Collinearity Across Spend & Volume Metrics:** Because `monetary_gross`, `monetary_net`, `total_line_items`, and `total_units` move together, tree ensembles distribute SHAP credit across them.
3. **Privacy Safeguards:** Operational logs and static reports mask customer identifiers (`CUST-***XX`).
"""
    with open(SHAP_REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Saved SHAP JSON report to     : {SHAP_REPORT_JSON_PATH}")
    logger.info(f"Saved SHAP Markdown report to : {SHAP_REPORT_MD_PATH}")


def run_explainability_pipeline() -> Dict[str, Any]:
    """Execute the complete Stage 9 SHAP explainability workflow."""
    if not MODEL_ARTIFACT_PATH.exists() or not CHURN_DATASET_PATH.exists():
        logger.error("Model artifact or churn dataset missing. Run Stages 7 and 8 first.")
        sys.exit(1)

    bundle = joblib.load(MODEL_ARTIFACT_PATH)
    pipeline = bundle["pipeline"]
    model_name = bundle["model_name"]
    model_version = bundle["model_version"]
    req_cols = bundle["numeric_features"] + bundle["categorical_features"]

    df_raw = pd.read_csv(CHURN_DATASET_PATH, dtype={"customer_id": str})
    df_preds = pd.read_csv(CHURN_PREDICTIONS_PATH, dtype={"customer_id": str})

    X_raw = df_raw[req_cols].copy()
    shap_matrix, base_val, X_trans = compute_shap_matrix(pipeline, X_raw)

    # 1. Build and save global feature importance table
    df_imp = build_global_importance_table(shap_matrix, X_trans)
    df_imp.to_csv(SHAP_GLOBAL_IMPORTANCE_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved global SHAP importance table to {SHAP_GLOBAL_IMPORTANCE_PATH}")

    # 2. Update every customer's top_driver_1..3 with exact per-customer SHAP attributions
    shap_drivers = format_customer_shap_drivers(df_raw, shap_matrix, list(X_trans.columns))
    df_preds["top_driver_1"] = [d[0] for d in shap_drivers]
    df_preds["top_driver_2"] = [d[1] for d in shap_drivers]
    df_preds["top_driver_3"] = [d[2] for d in shap_drivers]

    df_preds.to_csv(CHURN_PREDICTIONS_PATH, index=False, encoding="utf-8")
    logger.info(f"Updated {CHURN_PREDICTIONS_PATH} with per-customer SHAP drivers.")

    if test_database_connection():
        save_churn_predictions(df_preds)
    else:
        logger.warning("MySQL connection unavailable; skipped updating churn_predictions in DB.")

    # 3. Generate visual charts and individual masked examples
    example_profiles = generate_shap_visualizations(
        shap_matrix=shap_matrix,
        X_trans=X_trans,
        df_imp=df_imp,
        df_raw=df_raw,
        df_preds=df_preds,
    )

    save_explainability_reports(
        model_name=model_name,
        model_version=model_version,
        base_value=base_val,
        df_imp=df_imp,
        example_profiles=example_profiles,
    )

    # Console summary (with masked customer IDs)
    print("\n" + "=" * 78)
    print("  CUSTOMERPULSE AI - STAGE 9 MODEL EXPLAINABILITY (SHAP) SUMMARY")
    print("=" * 78)
    print(f"Model Explained               : {model_name} ({model_version})")
    print(f"Customers Explained           : {len(df_raw):,}")
    print("-" * 78)
    print("Top 5 Global Churn Drivers (by Mean |SHAP|):")
    for r in df_imp.head(5).to_dict(orient="records"):
        print(
            f"  {r['rank']}. {r['feature']:24} | Mean |SHAP| = {r['mean_abs_shap']:.4f} "
            f"({r['relative_share_pct']:5.1f}%) | {r['directional_effect']}"
        )
    print("-" * 78)
    print("Individual Customer SHAP Examples (Privacy-Masked):")
    for ex in example_profiles:
        print(
            f"  - {ex['profile_type']:19} [{ex['masked_customer_id']}] -> "
            f"Prob={ex['churn_probability']:.1%} ({ex['risk_tier']}) | "
            f"Driver 1: {ex['top_driver_1']}"
        )
    print("=" * 78)

    return {"global_importance": df_imp, "examples": example_profiles}


if __name__ == "__main__":
    run_explainability_pipeline()

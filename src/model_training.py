"""
Stage 8: Machine Learning Training & Selection Pipeline for CustomerPulse AI.
Trains and compares Dummy Baseline, Logistic Regression, Random Forest, and XGBoost
using Scikit-Learn Pipelines, 5-fold Stratified CV, and OOF threshold tuning.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CHURN_DATASET_PATH,
    MODELS_DIR,
    LOGS_DIR,
    MODEL_VERSION,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    RANDOM_SEED,
)
from src.database import save_model_metrics, test_database_connection
from src.evaluation import (
    evaluate_predictions,
    find_optimal_threshold,
    plot_model_comparisons,
    save_evaluation_reports,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage8_model_training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Features strictly available prior to prediction cutoff date
NUMERIC_FEATURES: List[str] = [
    "recency_days",
    "frequency_orders",
    "total_line_items",
    "total_units",
    "monetary_gross",
    "cancellations_amount",
    "monetary_net",
    "avg_order_value",
    "avg_days_between_orders",
    "unique_products",
    "cancellation_rate_pct",
    "tenure_days",
    "orders_last_30d",
    "orders_last_60d",
    "spend_last_30d",
    "spend_last_60d",
    "spend_ratio_last_30d",
]

CATEGORICAL_FEATURES: List[str] = [
    "country",
]

TARGET_COL = "is_churned"


def build_preprocessor() -> ColumnTransformer:
    """
    Build a Scikit-Learn ColumnTransformer that imputes and scales numeric features
    and one-hot encodes country (grouping infrequent countries with min_frequency=10).
    Uses pandas output so downstream tree models and SHAP retain readable column names.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def get_candidate_models(scale_pos_weight: float) -> Dict[str, Pipeline]:
    """
    Create Scikit-Learn Pipelines for Baseline, Logistic Regression, Random Forest, and XGBoost.
    """
    models = {
        "Dummy_Baseline": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            ]
        ),
        "Logistic_Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random_Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_alpha=0.5,
                        reg_lambda=2.0,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models


def train_and_evaluate_models() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load Stage 7 leak-free churn dataset, run 5-fold Stratified CV on the 80% training split,
    tune decision thresholds on OOF probabilities, evaluate on the 20% holdout test set,
    persist the winning pipeline and metadata, and write metrics to MySQL.
    """
    if not CHURN_DATASET_PATH.exists():
        logger.error(f"Churn dataset not found at {CHURN_DATASET_PATH}. Run Stage 7 first.")
        sys.exit(1)

    df = pd.read_csv(CHURN_DATASET_PATH, dtype={"customer_id": str})
    logger.info(f"Loaded leak-free churn dataset: {len(df):,} customers.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COL].astype(int).values

    # Stratified 80/20 Train/Holdout Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = round(neg_count / max(1, pos_count), 4)
    logger.info(
        f"Train Split: {len(X_train):,} rows (Retained={neg_count:,}, Churned={pos_count:,}, "
        f"scale_pos_weight={scale_pos_weight:.3f}) | Holdout Test Split: {len(X_test):,} rows"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    candidate_models = get_candidate_models(scale_pos_weight=scale_pos_weight)

    cv_metrics_list: List[Dict[str, Any]] = []
    test_metrics_list: List[Dict[str, Any]] = []
    test_probs_by_model: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, Pipeline] = {}
    optimal_thresholds: Dict[str, float] = {}

    for model_name, pipeline in candidate_models.items():
        logger.info(f"Evaluating model: {model_name} via 5-Fold Stratified CV...")

        # 1. Out-of-Fold (OOF) probability predictions on Training Split
        oof_probs = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        # 2. Select optimal probability threshold using OOF predictions (except Dummy_Baseline)
        if model_name == "Dummy_Baseline":
            opt_thresh = float(np.mean(y_train))
        else:
            opt_thresh, _ = find_optimal_threshold(y_train, oof_probs, min_recall=0.65)
        optimal_thresholds[model_name] = opt_thresh

        oof_eval = evaluate_predictions(y_train, oof_probs, threshold=opt_thresh)
        oof_eval["model_name"] = model_name
        oof_eval["model_version"] = MODEL_VERSION
        oof_eval["dataset_split"] = "cv_5fold_oof"
        cv_metrics_list.append(oof_eval)

        # 3. Fit pipeline on full 80% Training Split and evaluate on 20% Holdout Test Set
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_probs_by_model[model_name] = test_probs

        test_eval = evaluate_predictions(y_test, test_probs, threshold=opt_thresh)
        test_eval["model_name"] = model_name
        test_eval["model_version"] = MODEL_VERSION
        test_eval["dataset_split"] = "holdout_test"
        test_metrics_list.append(test_eval)

        logger.info(
            f"  -> {model_name:20} | Thresh={opt_thresh:.2f} | "
            f"Test PR-AUC={test_eval['pr_auc']:.4f} | ROC-AUC={test_eval['roc_auc']:.4f} | "
            f"F1={test_eval['f1_score']:.4f} (Prec={test_eval['precision_score']:.4f}, Rec={test_eval['recall_score']:.4f})"
        )

    # Select best non-dummy model by composite score of Holdout Test PR-AUC + F1
    real_models = [m for m in test_metrics_list if m["model_name"] != "Dummy_Baseline"]
    best_model_entry = max(
        real_models,
        key=lambda m: (round(m["pr_auc"], 3), round(m["f1_score"], 3), round(m["roc_auc"], 3)),
    )
    best_model_name = best_model_entry["model_name"]
    best_threshold = optimal_thresholds[best_model_name]
    best_pipeline = fitted_models[best_model_name]

    # Mark is_best_model flag on all records for MySQL storage
    all_db_metrics = []
    for row in cv_metrics_list + test_metrics_list:
        row_copy = dict(row)
        row_copy["is_best_model"] = (
            row_copy["model_name"] == best_model_name
            and row_copy["dataset_split"] == "holdout_test"
        )
        all_db_metrics.append(row_copy)

    # Generate diagnostic comparison plots
    plot_model_comparisons(
        y_test=y_test,
        test_probs_by_model=test_probs_by_model,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
    )

    # Extract post-preprocessing feature names from the fitted winning pipeline
    transformed_feature_names = list(
        best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    # Save versioned model bundle and JSON metadata
    artifact_bundle = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "model_version": MODEL_VERSION,
        "decision_threshold": best_threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "transformed_feature_names": transformed_feature_names,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact_bundle, MODEL_ARTIFACT_PATH)
    logger.info(f"[OK] Saved winning model bundle ({best_model_name}) to {MODEL_ARTIFACT_PATH}")

    metadata = {
        "model_name": best_model_name,
        "model_version": MODEL_VERSION,
        "decision_threshold": best_threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "transformed_feature_names": transformed_feature_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_holdout_metrics": best_model_entry,
        "trained_at_utc": artifact_bundle["trained_at_utc"],
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    report_payload = {
        "best_model_name": best_model_name,
        "model_version": MODEL_VERSION,
        "best_decision_threshold": best_threshold,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "holdout_test_metrics": test_metrics_list,
        "cv_oof_metrics": cv_metrics_list,
    }
    save_evaluation_reports(report_payload)

    # Save metrics to MySQL model_metrics table
    if test_database_connection():
        df_metrics_db = pd.DataFrame(all_db_metrics)
        save_model_metrics(df_metrics_db)
    else:
        logger.warning("MySQL connection unavailable; skipped saving model_metrics to DB.")

    return best_pipeline, report_payload


if __name__ == "__main__":
    train_and_evaluate_models()

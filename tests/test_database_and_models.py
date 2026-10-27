"""
Unit and integration tests for Stage 5 (MySQL Database Integration) and
Stage 8 (Model Evaluation, Threshold Selection, and Prediction Inference).
"""
from pathlib import Path
import numpy as np
import pytest

import src.predictions as predictions_module
from src.database import (
    get_connection_url,
    test_database_connection as _test_db_connection,
    verify_transactions_in_mysql,
)
from src.evaluation import evaluate_predictions, find_optimal_threshold
from src.predictions import (
    load_model_bundle,
    assign_risk_tier,
    predict_single_customer,
)


def test_database_url_escapes_special_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure special characters in DB_PASSWORD (@, #, /) are URL-encoded safely."""
    import src.database as db_mod

    monkeypatch.setattr(db_mod, "DB_USER", "test_user")
    monkeypatch.setattr(db_mod, "DB_PASSWORD", "P@ss#word/123")
    url = db_mod.get_connection_url(include_db=True)
    assert "P%40ss%23word%2F123" in url


def test_mysql_connection_and_reconciliation() -> None:
    """Verify live MySQL connection and confirm transactions table is populated."""
    assert _test_db_connection() is True
    summary = verify_transactions_in_mysql()
    assert summary["total_rows"] in (531211, 531236)
    assert summary["qualifying_orders"] in (19792, 19773)
    assert (
        summary["net_revenue"] == pytest.approx(9823263.86, rel=1e-3)
        or summary["net_revenue"] == pytest.approx(9772003.97, rel=1e-3)
    )


def test_evaluate_predictions_and_threshold_optimizer() -> None:
    """Test classification metric calculation and OOF threshold selection."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.10, 0.25, 0.45, 0.48, 0.75, 0.90])

    metrics_50 = evaluate_predictions(y_true, y_prob, threshold=0.50)
    assert metrics_50["true_positives"] == 2
    assert metrics_50["false_negatives"] == 1
    assert metrics_50["true_negatives"] == 3
    assert metrics_50["false_positives"] == 0

    opt_thresh, stats = find_optimal_threshold(y_true, y_prob, min_recall=0.65)
    assert 0.20 <= opt_thresh <= 0.80
    assert stats["f1"] >= metrics_50["f1_score"]


def test_load_model_bundle_and_predict_single_customer() -> None:
    """Verify trained model artifact loads and produces valid probability scores."""
    bundle = load_model_bundle()
    assert "pipeline" in bundle
    assert "decision_threshold" in bundle

    high_risk_payload = {
        "recency_days": 200,
        "frequency_orders": 1,
        "total_line_items": 5,
        "total_units": 10,
        "monetary_gross": 85.0,
        "cancellations_amount": 0.0,
        "monetary_net": 85.0,
        "avg_order_value": 85.0,
        "avg_days_between_orders": 0.0,
        "unique_products": 4,
        "cancellation_rate_pct": 0.0,
        "tenure_days": 200,
        "orders_last_30d": 0,
        "orders_last_60d": 0,
        "spend_last_30d": 0.0,
        "spend_last_60d": 0.0,
        "spend_ratio_last_30d": 0.0,
        "country": "United Kingdom",
    }
    res_high = predict_single_customer(high_risk_payload, bundle=bundle)
    assert 0.0 <= res_high["churn_probability"] <= 1.0
    assert res_high["predicted_churn"] in (0, 1)
    assert res_high["risk_tier"] in ("Low Risk", "Medium Risk", "High Risk")

    low_risk_payload = {
        **high_risk_payload,
        "recency_days": 5,
        "frequency_orders": 18,
        "monetary_gross": 6500.0,
        "monetary_net": 6500.0,
        "unique_products": 95,
        "tenure_days": 270,
        "orders_last_30d": 3,
        "orders_last_60d": 6,
        "spend_last_30d": 1200.0,
        "spend_last_60d": 2400.0,
        "spend_ratio_last_30d": 0.1846,
    }
    res_low = predict_single_customer(low_risk_payload, bundle=bundle)
    # A loyal recent buyer must have a strictly lower churn probability than an inactive 1-time buyer
    assert res_low["churn_probability"] < res_high["churn_probability"]


def test_missing_model_file_raises_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Edge Case: Verify load_model_bundle raises FileNotFoundError when model file is missing."""
    fake_path = Path("models/non_existent_model_v999.joblib")
    monkeypatch.setattr(predictions_module, "MODEL_ARTIFACT_PATH", fake_path)

    with pytest.raises(FileNotFoundError, match="Trained model artifact not found"):
        predictions_module.load_model_bundle()


def test_assign_risk_tier_boundaries() -> None:
    """Verify risk tier mapping across Low, Medium, and High thresholds."""
    assert assign_risk_tier(0.20, threshold=0.45) == "Low Risk"
    assert assign_risk_tier(0.48, threshold=0.45) == "Medium Risk"
    assert assign_risk_tier(0.75, threshold=0.45) == "High Risk"

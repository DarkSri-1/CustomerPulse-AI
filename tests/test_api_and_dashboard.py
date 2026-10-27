"""
End-to-end tests for Stage 10 (FastAPI REST Endpoints & Validation Edge Cases)
and Stage 11 (Streamlit Dashboard Data-Access Layer).
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from app.dashboard import (
    load_master_customer_table,
    load_reports_bundle,
    fetch_customer_360_profile,
    score_what_if_profile,
)


def test_health_endpoint(api_client: TestClient) -> None:
    """Verify GET /health returns 200 OK with database and model status."""
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert data["model_loaded"] is True


def test_list_customers_endpoint_and_filters(api_client: TestClient) -> None:
    """Verify GET /customers pagination and RFM segment filtering."""
    resp = api_client.get("/customers?segment=Champions&limit=5&sort_by=monetary_gross&sort_order=desc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_matching"] > 0
    assert len(data["customers"]) <= 5
    for cust in data["customers"]:
        assert cust["rfm_segment"] == "Champions"


def test_get_customer_detail_valid_and_unknown_id(api_client: TestClient) -> None:
    """Verify GET /customers/{customer_id} for known ID (17850) and unknown ID (404 edge case)."""
    resp_ok = api_client.get("/customers/17850")
    assert resp_ok.status_code == 200
    profile = resp_ok.json()
    assert profile["customer_id"] == "17850"
    assert "features" in profile
    assert "segmentation" in profile
    assert "churn_intelligence" in profile

    # Edge Case: Unknown customer ID must return 404 Not Found
    resp_404 = api_client.get("/customers/99999999")
    assert resp_404.status_code == 404
    assert "not found" in resp_404.json()["detail"].lower()


def test_analytics_overview_and_revenue_endpoints(api_client: TestClient) -> None:
    """Verify GET /analytics/overview and GET /analytics/revenue return verified KPIs."""
    resp_ov = api_client.get("/analytics/overview")
    assert resp_ov.status_code == 200
    ov = resp_ov.json()
    assert "kpis" in ov
    assert len(ov["rfm_segments"]) > 0
    assert len(ov["churn_risk_breakdown"]) > 0

    resp_rev = api_client.get("/analytics/revenue")
    assert resp_rev.status_code == 200
    rev = resp_rev.json()
    assert len(rev["monthly_revenue_trend"]) == 13
    assert len(rev["top_10_highest_revenue_products"]) == 10


def test_predict_churn_endpoint_valid_and_edge_cases(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test POST /predict/churn with:
      1. Valid customer payload -> 200 OK
      2. Empty request body {} -> 422 Unprocessable Entity
      3. Invalid negative monetary_gross or zero frequency_orders -> 422 Unprocessable Entity
      4. Missing model artifact when app.state.model_bundle is None -> 503 Service Unavailable
    """
    valid_payload = {
        "recency_days": 145,
        "frequency_orders": 1,
        "monetary_gross": 160.00,
        "unique_products": 7,
        "tenure_days": 145,
        "country": "United Kingdom",
    }
    resp = api_client.post("/predict/churn", json=valid_payload)
    assert resp.status_code == 200
    pred = resp.json()
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["risk_tier"] in ("Low Risk", "Medium Risk", "High Risk")
    assert len(pred["top_risk_indicators"]) == 3

    # Edge Case 1: Empty JSON request body
    resp_empty = api_client.post("/predict/churn", json={})
    assert resp_empty.status_code == 422

    # Edge Case 2: Invalid negative spend and zero orders
    resp_invalid = api_client.post(
        "/predict/churn",
        json={"recency_days": -10, "frequency_orders": 0, "monetary_gross": -50.0},
    )
    assert resp_invalid.status_code == 422

    # Edge Case 3: Simulate missing model file when cache is cleared
    monkeypatch.setattr(api_client.app.state, "model_bundle", None)
    monkeypatch.setattr(
        api_main,
        "load_model_bundle",
        lambda: (_ for _ in ()).throw(FileNotFoundError("Simulated missing model")),
    )
    resp_503 = api_client.post("/predict/churn", json=valid_payload)
    assert resp_503.status_code == 503


def test_models_metrics_endpoint(api_client: TestClient) -> None:
    """Verify GET /models/metrics returns evaluation metrics and SHAP rankings."""
    resp = api_client.get("/models/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["evaluation_metrics"]) >= 4
    assert len(data["global_shap_importance"]) >= 10


def test_dashboard_data_access_functions() -> None:
    """Verify Stage 11 Streamlit dashboard data loaders and 360 profile lookup."""
    df_cust = load_master_customer_table()
    assert len(df_cust) == 4334
    assert "rfm_segment" in df_cust.columns
    assert "risk_tier" in df_cust.columns

    eda_rep, churn_rep, model_rep, shap_df, shap_rep = load_reports_bundle()
    assert (
        eda_rep["kpis"]["store_net_revenue_gbp"] == pytest.approx(9823263.86, rel=1e-2)
        or eda_rep["kpis"]["store_net_revenue_gbp"] == pytest.approx(9772003.97, rel=1e-2)
    )
    assert churn_rep["cohort_accounting"]["eligible_observation_cohort_customers"] in (3381, 3368)
    assert not shap_df.empty

    # Verify known and unknown customer lookup in dashboard helper
    prof_ok = fetch_customer_360_profile("17850", df_cust)
    assert prof_ok is not None
    assert prof_ok["customer_id"] == "17850"

    prof_none = fetch_customer_360_profile("UNKNOWN_ID_999", df_cust)
    assert prof_none is None

    # Verify What-If simulator helper
    sim_res = score_what_if_profile(
        {
            "recency_days": 110,
            "frequency_orders": 2,
            "total_line_items": 12,
            "total_units": 50,
            "monetary_gross": 310.0,
            "cancellations_amount": 0.0,
            "monetary_net": 310.0,
            "avg_order_value": 155.0,
            "avg_days_between_orders": 30.0,
            "unique_products": 10,
            "cancellation_rate_pct": 0.0,
            "tenure_days": 150,
            "orders_last_30d": 0,
            "orders_last_60d": 0,
            "spend_last_30d": 0.0,
            "spend_last_60d": 0.0,
            "spend_ratio_last_30d": 0.0,
            "country": "United Kingdom",
        }
    )
    assert 0.0 <= sim_res["churn_probability"] <= 1.0

"""
Stage 11: Interactive Streamlit Dashboard for CustomerPulse AI.
Delivers 7 analytical modules connected to the FastAPI REST backend with automatic
fallback to verified MySQL / processed dataset artifacts.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import text

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EDA_REPORT_JSON_PATH,
    CHURN_REPORT_JSON_PATH,
    MODEL_REPORT_JSON_PATH,
    SHAP_GLOBAL_IMPORTANCE_PATH,
    SHAP_REPORT_JSON_PATH,
    CUSTOMER_FEATURES_PATH,
    CUSTOMER_SEGMENTS_PATH,
    CHURN_PREDICTIONS_PATH,
    CUSTOMER_TRANSACTIONS_PATH,
    FIGURES_DIR,
)
from src.database import get_engine, test_database_connection
from src.predictions import load_model_bundle, predict_single_customer, derive_customer_risk_drivers

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# =====================================================================
# 1. Data-Access Layer (Testable & Cached)
# =====================================================================

def check_api_health() -> Dict[str, Any]:
    """Ping FastAPI /health endpoint with a short timeout."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            data["api_online"] = True
            return data
    except requests.RequestException:
        pass
    return {
        "api_online": False,
        "status": "offline (using local data layer)",
        "database_connected": test_database_connection(),
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_master_customer_table() -> pd.DataFrame:
    """
    Load and join customer_features, customer_segments, and churn_predictions.
    Used across Customer Analytics, Segmentation, Churn Intelligence, and 360 Explorer.
    """
    if not CUSTOMER_FEATURES_PATH.exists():
        return pd.DataFrame()

    df_feat = pd.read_csv(CUSTOMER_FEATURES_PATH, dtype={"customer_id": str})

    if CUSTOMER_SEGMENTS_PATH.exists():
        df_seg = pd.read_csv(
            CUSTOMER_SEGMENTS_PATH,
            dtype={"customer_id": str, "rfm_score_str": str},
        )
        df = pd.merge(df_feat, df_seg, on="customer_id", how="left")
    else:
        df = df_feat.copy()

    if CHURN_PREDICTIONS_PATH.exists():
        df_pred = pd.read_csv(CHURN_PREDICTIONS_PATH, dtype={"customer_id": str})
        pred_cols = [
            "customer_id",
            "observation_cutoff_date",
            "churn_probability",
            "predicted_churn",
            "actual_churn",
            "risk_tier",
            "model_name",
            "model_version",
            "top_driver_1",
            "top_driver_2",
            "top_driver_3",
        ]
        available_cols = [c for c in pred_cols if c in df_pred.columns]
        df = pd.merge(df, df_pred[available_cols], on="customer_id", how="left")

    df["risk_tier"] = df["risk_tier"].fillna("Unscored (Post-Cutoff New Buyer)")
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_reports_bundle() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Load verified JSON/CSV reports from Stages 4, 7, 8, and 9."""
    eda_rep: Dict[str, Any] = {}
    if EDA_REPORT_JSON_PATH.exists():
        with open(EDA_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            eda_rep = json.load(f)

    churn_rep: Dict[str, Any] = {}
    if CHURN_REPORT_JSON_PATH.exists():
        with open(CHURN_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            churn_rep = json.load(f)

    model_rep: Dict[str, Any] = {}
    if MODEL_REPORT_JSON_PATH.exists():
        with open(MODEL_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            model_rep = json.load(f)

    shap_df = (
        pd.read_csv(SHAP_GLOBAL_IMPORTANCE_PATH)
        if SHAP_GLOBAL_IMPORTANCE_PATH.exists()
        else pd.DataFrame()
    )

    shap_rep: Dict[str, Any] = {}
    if SHAP_REPORT_JSON_PATH.exists():
        with open(SHAP_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            shap_rep = json.load(f)

    return eda_rep, churn_rep, model_rep, shap_df, shap_rep


def fetch_customer_360_profile(customer_id: str, df_cust: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Fetch a customer's 360-degree profile and recent transactions.
    Calls FastAPI GET /customers/{customer_id} first; falls back to MySQL/CSV if API is offline.
    """
    cid = str(customer_id).strip()
    try:
        resp = requests.get(f"{API_BASE_URL}/customers/{cid}", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
    except requests.RequestException:
        pass

    # Fallback to local master table + MySQL/CSV transactions
    match = df_cust[df_cust["customer_id"] == cid]
    if match.empty:
        return None

    row = match.iloc[0]
    recent_tx = []
    if test_database_connection():
        try:
            engine = get_engine(include_db=True)
            with engine.connect() as conn:
                tx_rows = conn.execute(
                    text(
                        """
                        SELECT invoice_no, stock_code, description, quantity,
                               invoice_date, unit_price, line_total, transaction_type
                        FROM transactions
                        WHERE customer_id = :cid
                        ORDER BY invoice_date DESC
                        LIMIT 15;
                        """
                    ),
                    {"cid": cid},
                ).mappings().all()
                recent_tx = [dict(r) for r in tx_rows]
        except Exception:
            recent_tx = []

    if not recent_tx and CUSTOMER_TRANSACTIONS_PATH.exists():
        df_tx = pd.read_csv(
            CUSTOMER_TRANSACTIONS_PATH,
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
        )
        sub_tx = df_tx[df_tx["CustomerID"] == cid].sort_values("InvoiceDate", ascending=False).head(15)
        recent_tx = [
            {
                "invoice_no": r["InvoiceNo"],
                "stock_code": r["StockCode"],
                "description": r["Description"],
                "quantity": int(r["Quantity"]),
                "invoice_date": str(r["InvoiceDate"]),
                "unit_price": float(r["UnitPrice"]),
                "line_total": float(r["LineTotal"]),
                "transaction_type": str(r["TransactionType"]),
            }
            for _, r in sub_tx.iterrows()
        ]

    prob = float(row["churn_probability"]) if pd.notna(row.get("churn_probability")) else None
    risk_tier = str(row.get("risk_tier", "Unscored (Post-Cutoff New Buyer)"))

    return {
        "customer_id": cid,
        "country": str(row["country"]),
        "is_uk": bool(row["is_uk"]),
        "features": {
            "recency_days": int(row["recency_days"]),
            "frequency_orders": int(row["frequency_orders"]),
            "total_line_items": int(row["total_line_items"]),
            "total_units": int(row["total_units"]),
            "monetary_gross": float(row["monetary_gross"]),
            "cancellations_amount": float(row["cancellations_amount"]),
            "monetary_net": float(row["monetary_net"]),
            "avg_order_value": float(row["avg_order_value"]),
            "avg_days_between_orders": float(row["avg_days_between_orders"]),
            "unique_products": int(row["unique_products"]),
            "cancellation_rate_pct": float(row["cancellation_rate_pct"]),
            "tenure_days": int(row["tenure_days"]),
            "first_purchase_date": str(row["first_purchase_date"]),
            "last_purchase_date": str(row["last_purchase_date"]),
        },
        "segmentation": {
            "r_score": int(row["r_score"]) if pd.notna(row.get("r_score")) else None,
            "f_score": int(row["f_score"]) if pd.notna(row.get("f_score")) else None,
            "m_score": int(row["m_score"]) if pd.notna(row.get("m_score")) else None,
            "rfm_score_str": str(row.get("rfm_score_str", "N/A")),
            "rfm_segment": str(row.get("rfm_segment", "N/A")),
            "kmeans_segment_label": str(row.get("kmeans_segment_label", "N/A")),
        },
        "churn_intelligence": {
            "scored_in_pre_cutoff_cohort": prob is not None,
            "churn_probability": prob,
            "predicted_churn": int(row["predicted_churn"]) if pd.notna(row.get("predicted_churn")) else None,
            "actual_churn": int(row["actual_churn"]) if pd.notna(row.get("actual_churn")) else None,
            "risk_tier": risk_tier,
            "top_shap_drivers": [
                str(row[c])
                for c in ["top_driver_1", "top_driver_2", "top_driver_3"]
                if c in row and pd.notna(row[c])
            ],
            "recommended_action": (
                "Priority Win-Back Campaign: Trigger account outreach and restock incentive."
                if risk_tier == "High Risk"
                else "Maintain regular seasonal wholesale cadence."
            ),
        },
        "recent_transactions": recent_tx,
    }


def score_what_if_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a custom customer profile via FastAPI POST /predict/churn,
    falling back to local model inference if the API server is offline.
    """
    try:
        resp = requests.post(f"{API_BASE_URL}/predict/churn", json=payload, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            data["source"] = "FastAPI Live Endpoint (POST /predict/churn)"
            return data
    except requests.RequestException:
        pass

    bundle = load_model_bundle()
    res = predict_single_customer(payload, bundle=bundle)
    d1, d2, d3 = derive_customer_risk_drivers(
        pd.Series(payload),
        {"recency_days": 75.0, "frequency_orders": 2.0, "monetary_gross": 650.0},
    )
    res["top_risk_indicators"] = [d1, d2, d3]
    res["recommended_action"] = (
        "Trigger targeted win-back campaign with top anchor SKUs."
        if res["risk_tier"] in ("High Risk", "Medium Risk")
        else "Healthy retention profile — continue standard loyalty cadence."
    )
    res["source"] = "Local Model Artifact Fallback (churn_model_v1.joblib)"
    return res


# =====================================================================
# 2. Streamlit UI Layout & Renderers
# =====================================================================

def render_dashboard() -> None:
    st.set_page_config(
        page_title="CustomerPulse AI — E-Commerce Intelligence & Churn System",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.spinner("Loading verified analytics, customer segments, and ML predictions..."):
        eda_rep, churn_rep, model_rep, shap_df, shap_rep = load_reports_bundle()
        df_cust = load_master_customer_table()
        health_info = check_api_health()

    if df_cust.empty or not eda_rep:
        st.error(
            "⚠️ Required processed data files were not found in `data/processed/` or `reports/`. "
            "Please run Stages 3–9 (`src.preprocessing`, `src.eda`, `src.segmentation`, `src.features`, "
            "`src.model_training`, `src.predictions`, `src.explainability`) first."
        )
        st.stop()

    # Sidebar Navigation & System Status
    st.sidebar.title("⚡ CustomerPulse AI")
    st.sidebar.caption("E-Commerce Customer Intelligence & Churn Prediction System")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Analytical Module:",
        [
            "1. Business Overview",
            "2. Customer Analytics",
            "3. Customer Segmentation",
            "4. Churn Intelligence",
            "5. Product & Country Analytics",
            "6. Model Performance & SHAP",
            "7. Individual Customer Explorer",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 System Telemetry")
    if health_info.get("api_online"):
        st.sidebar.success(f"FastAPI Backend: Online ({API_BASE_URL})")
    else:
        st.sidebar.info("FastAPI: Offline (Using Direct Data Layer)")

    if health_info.get("database_connected"):
        st.sidebar.success("MySQL Database: Connected")
    else:
        st.sidebar.warning("MySQL: Using Local CSV Cache")

    best_model_name = model_rep.get("best_model_name", "ML Pipeline")
    best_thresh = model_rep.get("best_decision_threshold", 0.45)
    st.sidebar.caption(
        f"**Active Model:** `{best_model_name}` (`{model_rep.get('model_version', 'v1.0.0')}`)\n\n"
        f"**Decision Threshold:** `{best_thresh:.2f}`"
    )

    # -----------------------------------------------------------------
    # PAGE 1: BUSINESS OVERVIEW
    # -----------------------------------------------------------------
    if page.startswith("1."):
        st.title("📊 Executive Business Overview")
        st.markdown(
            "Verified store-wide and customer-level financial performance across **531,211** cleaned "
            "transactions (`2010-12-01` to `2011-12-09`)."
        )

        kpis = eda_rep.get("kpis", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Store Gross Sales", f"£{kpis.get('store_gross_revenue_gbp', 0):,.0f}")
        c2.metric(
            "Returns / Cancellations",
            f"£{kpis.get('store_cancellations_gbp', 0):,.0f}",
            delta=f"-{kpis.get('store_return_rate_pct_of_gross', 0):.2f}% of gross",
            delta_color="inverse",
        )
        c3.metric("Store Net Revenue", f"£{kpis.get('store_net_revenue_gbp', 0):,.0f}")
        c4.metric("Qualifying Orders", f"{kpis.get('store_qualifying_orders', 0):,}")
        c5.metric("Store AOV", f"£{kpis.get('store_average_order_value_gbp', 0):,.2f}")
        c6.metric(
            "Active Customers",
            f"{kpis.get('unique_customers_with_positive_sales', 0):,}",
            delta=f"{kpis.get('repeat_purchase_rate_pct', 0):.1f}% Repeat",
        )

        with st.expander("ℹ️ How are these KPIs defined and reconciled?", expanded=False):
            st.markdown(
                "- **Gross Qualifying Sales (£10,430,132.89):** Sum of `Quantity * UnitPrice` across all positive, non-cancelled product orders (including guest checkouts).\n"
                "- **Returns / Cancellations (£-606,869.03):** Sum of line totals on cancellation invoices (`InvoiceNo` prefixed with `'C'`).\n"
                "- **Store Net Revenue (£9,823,263.86):** Gross Qualifying Sales minus Cancellations/Returns.\n"
                "- **Customer-Identified Net Revenue (£8,278,318.24):** Net revenue from the `4,334` active customers with a verified `CustomerID`."
            )

        st.markdown("---")
        monthly_df = pd.DataFrame(eda_rep.get("monthly_revenue_trend", []))
        if not monthly_df.empty:
            col_l, col_r = st.columns([3, 2])
            with col_l:
                fig_rev = go.Figure()
                fig_rev.add_trace(
                    go.Scatter(
                        x=monthly_df["InvoiceYearMonth"],
                        y=monthly_df["gross_revenue"],
                        mode="lines+markers",
                        name="Gross Qualifying Sales (£)",
                        line=dict(color="#1f77b4", width=3),
                    )
                )
                fig_rev.add_trace(
                    go.Scatter(
                        x=monthly_df["InvoiceYearMonth"],
                        y=monthly_df["net_revenue"],
                        mode="lines+markers",
                        name="Net Revenue After Returns (£)",
                        line=dict(color="#2ca02c", width=3, dash="dash"),
                    )
                )
                fig_rev.update_layout(
                    title="Monthly Gross Sales vs. Net Revenue Trend (Dec 2010 – Dec 2011)",
                    xaxis_title="Invoice Year-Month",
                    yaxis_title="Revenue (£ GBP)",
                    hovermode="x unified",
                    height=420,
                )
                st.plotly_chart(fig_rev, use_container_width=True)

            with col_r:
                fig_ord = px.bar(
                    monthly_df,
                    x="InvoiceYearMonth",
                    y="qualifying_orders",
                    title="Monthly Qualifying Order Volume",
                    labels={"InvoiceYearMonth": "Year-Month", "qualifying_orders": "Orders"},
                    color="qualifying_orders",
                    color_continuous_scale="Blues",
                )
                fig_ord.update_layout(height=420, coloraxis_showscale=False)
                st.plotly_chart(fig_ord, use_container_width=True)

        # Day of week and Hourly seasonality
        dow_df = pd.DataFrame(eda_rep.get("day_of_week_performance", []))
        hour_df = pd.DataFrame(eda_rep.get("hourly_performance", []))
        if not dow_df.empty and not hour_df.empty:
            c_dow, c_hr = st.columns(2)
            with c_dow:
                fig_dow = px.bar(
                    dow_df,
                    x="InvoiceDayOfWeek",
                    y="qualifying_orders",
                    title="Order Volume by Day of Week (Closed Saturdays)",
                    labels={"InvoiceDayOfWeek": "Day of Week", "qualifying_orders": "Qualifying Orders"},
                    color_discrete_sequence=["#17becf"],
                )
                fig_dow.update_layout(height=360)
                st.plotly_chart(fig_dow, use_container_width=True)
            with c_hr:
                fig_hr = px.line(
                    hour_df,
                    x="InvoiceHour",
                    y="qualifying_orders",
                    markers=True,
                    title="Order Volume by Hour of Day (Peak 10:00 AM – 3:00 PM)",
                    labels={"InvoiceHour": "Hour of Day (24h)", "qualifying_orders": "Qualifying Orders"},
                    color_discrete_sequence=["#e377c2"],
                )
                fig_hr.update_layout(height=360)
                st.plotly_chart(fig_hr, use_container_width=True)

    # -----------------------------------------------------------------
    # PAGE 2: CUSTOMER ANALYTICS
    # -----------------------------------------------------------------
    elif page.startswith("2."):
        st.title("📈 Customer Purchasing Behavior & Pareto Concentration")
        st.markdown("Interactive exploration of customer spend distributions, repeat order rates, and revenue concentration.")

        # Interactive Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        countries_list = ["All Countries"] + sorted(df_cust["country"].dropna().unique().tolist())
        sel_country = f_col1.selectbox("Filter by Country:", countries_list, index=0)
        min_orders = f_col2.slider("Minimum Qualifying Orders:", min_value=1, max_value=20, value=1)
        max_recency = f_col3.slider("Maximum Recency (Days):", min_value=0, max_value=375, value=375)

        filtered_cust = df_cust[
            (df_cust["frequency_orders"] >= min_orders)
            & (df_cust["recency_days"] <= max_recency)
        ].copy()
        if sel_country != "All Countries":
            filtered_cust = filtered_cust[filtered_cust["country"] == sel_country].copy()

        if filtered_cust.empty:
            st.warning("No customers match the selected filter criteria. Please broaden your filters.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Filtered Customers", f"{len(filtered_cust):,}")
            k2.metric("Median Gross Spend", f"£{filtered_cust['monetary_gross'].median():,.2f}")
            k3.metric("Mean Gross Spend", f"£{filtered_cust['monetary_gross'].mean():,.2f}")
            repeat_pct = (filtered_cust["frequency_orders"] > 1).mean() * 100
            k4.metric("Repeat Buyers (2+ Orders)", f"{repeat_pct:.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                filtered_cust["log10_spend"] = np.log10(filtered_cust["monetary_gross"].clip(lower=1.0))
                fig_sp = px.histogram(
                    filtered_cust,
                    x="log10_spend",
                    nbins=40,
                    title="Customer Gross Spend Distribution — Log10(£) Scale",
                    labels={"log10_spend": "Log10(Gross Spend in £)"},
                    color_discrete_sequence=["#9467bd"],
                )
                fig_sp.update_layout(height=400)
                st.plotly_chart(fig_sp, use_container_width=True)

            with col2:
                # Pareto cumulative revenue curve
                pareto_df = filtered_cust.sort_values("monetary_gross", ascending=False).reset_index(drop=True)
                pareto_df["customer_pct"] = ((pareto_df.index + 1) / len(pareto_df)) * 100
                total_rev = pareto_df["monetary_gross"].sum() or 1.0
                pareto_df["cum_revenue_pct"] = (pareto_df["monetary_gross"].cumsum() / total_rev) * 100

                fig_par = px.line(
                    pareto_df,
                    x="customer_pct",
                    y="cum_revenue_pct",
                    title="Pareto Revenue Concentration Curve (Cumulative %)",
                    labels={
                        "customer_pct": "Cumulative % of Customers (Ranked by Spend)",
                        "cum_revenue_pct": "Cumulative % of Gross Revenue",
                    },
                    color_discrete_sequence=["#d62728"],
                )
                fig_par.add_hline(y=74.57, line_dash="dash", line_color="gray", annotation_text="Top 20% = ~74.6% Revenue")
                fig_par.update_layout(height=400)
                st.plotly_chart(fig_par, use_container_width=True)

            st.subheader("Top Customers Matching Filters")
            display_cols = [
                "customer_id",
                "country",
                "recency_days",
                "frequency_orders",
                "monetary_gross",
                "monetary_net",
                "avg_order_value",
                "unique_products",
                "rfm_segment",
                "risk_tier",
            ]
            st.dataframe(
                filtered_cust.sort_values("monetary_gross", ascending=False)[display_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------------------
    # PAGE 3: CUSTOMER SEGMENTATION
    # -----------------------------------------------------------------
    elif page.startswith("3."):
        st.title("🏷️ RFM Segmentation & K-Means Clustering")
        st.markdown(
            "Comparison of rule-based **RFM Quintile Segments** and unsupervised **K-Means (`k=4`) Clusters** "
            "trained on log-transformed, standardized behavioral features."
        )

        seg_options = ["All Segments"] + sorted(df_cust["rfm_segment"].dropna().unique().tolist())
        selected_seg = st.selectbox("Filter Table & Scatter by RFM Segment:", seg_options, index=0)

        rfm_summary = (
            df_cust.groupby("rfm_segment")
            .agg(
                customers=("customer_id", "count"),
                mean_recency=("recency_days", "mean"),
                mean_frequency=("frequency_orders", "mean"),
                mean_monetary=("monetary_gross", "mean"),
                total_revenue=("monetary_gross", "sum"),
            )
            .round(2)
            .reset_index()
            .sort_values("total_revenue", ascending=False)
        )
        rfm_summary["revenue_share_pct"] = (
            (rfm_summary["total_revenue"] / rfm_summary["total_revenue"].sum()) * 100
        ).round(2)

        col1, col2 = st.columns(2)
        with col1:
            fig_seg_rev = px.bar(
                rfm_summary,
                x="rfm_segment",
                y="total_revenue",
                color="rfm_segment",
                text="revenue_share_pct",
                title="Gross Revenue by RFM Marketing Segment (£)",
                labels={"rfm_segment": "RFM Segment", "total_revenue": "Gross Revenue (£)"},
            )
            fig_seg_rev.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_seg_rev.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_seg_rev, use_container_width=True)

        with col2:
            km_summary = (
                df_cust.groupby("kmeans_segment_label")
                .agg(
                    customers=("customer_id", "count"),
                    total_revenue=("monetary_gross", "sum"),
                    mean_recency=("recency_days", "mean"),
                    mean_orders=("frequency_orders", "mean"),
                )
                .round(2)
                .reset_index()
                .sort_values("total_revenue", ascending=False)
            )
            fig_km = px.pie(
                km_summary,
                names="kmeans_segment_label",
                values="customers",
                hole=0.45,
                title="Customer Distribution Across K-Means (k=4) Clusters",
            )
            fig_km.update_layout(height=420)
            st.plotly_chart(fig_km, use_container_width=True)

        st.subheader("RFM Segment Summary Metrics")
        st.dataframe(rfm_summary, use_container_width=True, hide_index=True)

        # Interactive RFM Behavioral Scatter Plot
        seg_df = df_cust if selected_seg == "All Segments" else df_cust[df_cust["rfm_segment"] == selected_seg]
        fig_scatter = px.scatter(
            seg_df,
            x="recency_days",
            y="frequency_orders",
            size="monetary_gross",
            color="rfm_segment",
            hover_data=["customer_id", "monetary_gross", "country", "kmeans_segment_label"],
            log_y=True,
            size_max=45,
            title="Customer Behavioral Map: Recency vs. Order Frequency (Bubble Size = Gross Spend £)",
            labels={"recency_days": "Recency (Days Since Last Order)", "frequency_orders": "Order Frequency (Log Scale)"},
        )
        fig_scatter.update_layout(height=460)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------------------
    # PAGE 4: CHURN INTELLIGENCE & LIVE SIMULATOR
    # -----------------------------------------------------------------
    elif page.startswith("4."):
        st.title("⚡ Customer Churn Intelligence & Live Risk Simulator")
        st.markdown(
            "Leak-free churn predictions for the **3,381** eligible pre-cutoff cohort customers "
            "(`Observation Window: 2010-12-01 to 2011-09-09` | `Outcome Window: 90 Days`)."
        )

        scored_df = df_cust[df_cust["churn_probability"].notna()].copy()
        high_cnt = int((scored_df["risk_tier"] == "High Risk").sum())
        med_cnt = int((scored_df["risk_tier"] == "Medium Risk").sum())
        low_cnt = int((scored_df["risk_tier"] == "Low Risk").sum())
        rev_at_risk = float(scored_df.loc[scored_df["risk_tier"] == "High Risk", "monetary_gross"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scored Cohort Customers", f"{len(scored_df):,}")
        m2.metric("High-Risk Customers", f"{high_cnt:,}", delta=f"{(high_cnt / len(scored_df)) * 100:.1f}% of cohort", delta_color="inverse")
        m3.metric("Medium-Risk Customers", f"{med_cnt:,}")
        m4.metric("High-Risk Historical Spend", f"£{rev_at_risk:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            fig_prob = px.histogram(
                scored_df,
                x="churn_probability",
                color="risk_tier",
                nbins=35,
                title="Predicted Churn Probability Distribution Across Cohort",
                labels={"churn_probability": "Predicted 90-Day Churn Probability"},
                color_discrete_map={"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
            )
            fig_prob.add_vline(
                x=best_thresh,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Threshold ({best_thresh:.2f})",
            )
            fig_prob.update_layout(height=400)
            st.plotly_chart(fig_prob, use_container_width=True)

        with c2:
            risk_seg = (
                scored_df.groupby(["rfm_segment", "risk_tier"])
                .size()
                .reset_index(name="customers")
            )
            fig_rs = px.bar(
                risk_seg,
                x="rfm_segment",
                y="customers",
                color="risk_tier",
                barmode="group",
                title="Churn Risk Tier Breakdown by RFM Segment",
                labels={"rfm_segment": "RFM Segment", "customers": "Customer Count"},
                color_discrete_map={"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"},
            )
            fig_rs.update_layout(height=400)
            st.plotly_chart(fig_rs, use_container_width=True)

        st.markdown("---")
        st.subheader("🧪 Interactive What-If Churn Risk Simulator (Connected to FastAPI / ML Pipeline)")
        with st.form("what_if_simulator_form"):
            sc1, sc2, sc3, sc4 = st.columns(4)
            sim_recency = sc1.number_input("Recency (Days Since Last Order)", min_value=0, max_value=365, value=140)
            sim_freq = sc2.number_input("Frequency (Qualifying Orders)", min_value=1, max_value=200, value=1)
            sim_spend = sc3.number_input("Gross Spend (£ GBP)", min_value=10.0, max_value=250000.0, value=180.0, step=25.0)
            sim_skus = sc4.number_input("Unique Products Purchased", min_value=1, max_value=500, value=8)

            sc5, sc6, sc7, sc8 = st.columns(4)
            sim_tenure = sc5.number_input("Customer Tenure (Days)", min_value=0, max_value=400, value=150)
            sim_ord_30d = sc6.number_input("Orders in Last 30 Days", min_value=0, max_value=50, value=0)
            sim_ord_60d = sc7.number_input("Orders in Last 60 Days", min_value=0, max_value=100, value=0)
            sim_country = sc8.selectbox("Country", ["United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Spain"])

            submitted = st.form_submit_button("⚡ Calculate Real-Time Churn Risk")

        if submitted:
            sim_payload = {
                "recency_days": int(sim_recency),
                "frequency_orders": int(sim_freq),
                "total_line_items": int(max(sim_skus, sim_freq * 5)),
                "total_units": int(max(sim_skus * 4, 10)),
                "monetary_gross": float(sim_spend),
                "cancellations_amount": 0.0,
                "monetary_net": float(sim_spend),
                "avg_order_value": round(float(sim_spend) / max(1, int(sim_freq)), 2),
                "avg_days_between_orders": 35.0 if int(sim_freq) > 1 else 0.0,
                "unique_products": int(sim_skus),
                "cancellation_rate_pct": 0.0,
                "tenure_days": int(max(sim_tenure, sim_recency)),
                "orders_last_30d": int(sim_ord_30d),
                "orders_last_60d": int(max(sim_ord_60d, sim_ord_30d)),
                "spend_last_30d": round(float(sim_spend) * 0.25, 2) if int(sim_ord_30d) > 0 else 0.0,
                "spend_last_60d": round(float(sim_spend) * 0.40, 2) if int(sim_ord_60d) > 0 else 0.0,
                "spend_ratio_last_30d": 0.25 if int(sim_ord_30d) > 0 else 0.0,
                "country": sim_country,
            }
            sim_result = score_what_if_profile(sim_payload)
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Churn Probability", f"{sim_result['churn_probability']:.1%}")
            r2.metric("Assigned Risk Tier", sim_result["risk_tier"])
            r3.metric("Binary Prediction", "Churned (1)" if sim_result["predicted_churn"] == 1 else "Retained (0)")
            st.info(
                f"**Recommended Retention Action:** {sim_result['recommended_action']}\n\n"
                f"**Key Risk Indicators:** {', '.join(sim_result.get('top_risk_indicators', []))}\n\n"
                f"*Scored via: {sim_result['source']}*"
            )

        st.markdown("---")
        st.subheader("📋 Priority High-Risk Retention Queue (Filterable)")
        tier_filter = st.selectbox("Filter Queue by Risk Tier:", ["High Risk", "Medium Risk", "Low Risk"], index=0)
        queue_df = scored_df[scored_df["risk_tier"] == tier_filter].sort_values(
            ["churn_probability", "monetary_gross"], ascending=[False, False]
        )
        st.dataframe(
            queue_df[
                [
                    "customer_id",
                    "country",
                    "rfm_segment",
                    "churn_probability",
                    "risk_tier",
                    "recency_days",
                    "frequency_orders",
                    "monetary_gross",
                    "top_driver_1",
                    "top_driver_2",
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # PAGE 5: PRODUCT & COUNTRY ANALYTICS
    # -----------------------------------------------------------------
    elif page.startswith("5."):
        st.title("🌍 Product & Geographic Performance")
        st.markdown("Top-performing physical products, return anomalies, and international market breakdown.")

        top_rev_df = pd.DataFrame(eda_rep.get("top_10_highest_revenue_products", []))
        top_qty_df = pd.DataFrame(eda_rep.get("top_10_selling_products_by_qty", []))
        top_can_df = pd.DataFrame(eda_rep.get("top_5_cancelled_products", []))
        countries_df = pd.DataFrame(eda_rep.get("top_10_countries_by_revenue", []))

        p1, p2 = st.columns(2)
        with p1:
            if not top_rev_df.empty:
                fig_pr = px.bar(
                    top_rev_df.iloc[::-1],
                    x="gross_revenue",
                    y="Description",
                    orientation="h",
                    title="Top 10 Products by Gross Qualifying Revenue (£)",
                    labels={"gross_revenue": "Gross Revenue (£)", "Description": "Product Description"},
                    color="gross_revenue",
                    color_continuous_scale="Blues",
                )
                fig_pr.update_layout(height=430, coloraxis_showscale=False)
                st.plotly_chart(fig_pr, use_container_width=True)

        with p2:
            if not top_qty_df.empty:
                fig_pq = px.bar(
                    top_qty_df.iloc[::-1],
                    x="total_quantity",
                    y="Description",
                    orientation="h",
                    title="Top 10 Products by Total Units Sold",
                    labels={"total_quantity": "Units Sold", "Description": "Product Description"},
                    color="total_quantity",
                    color_continuous_scale="Oranges",
                )
                fig_pq.update_layout(height=430, coloraxis_showscale=False)
                st.plotly_chart(fig_pq, use_container_width=True)

        st.markdown("---")
        c_left, c_right = st.columns([3, 2])
        with c_left:
            exclude_uk = st.checkbox(
                "Exclude United Kingdom (84.56% Share) to Zoom in on International Wholesale Markets",
                value=True,
            )
            plot_countries = (
                countries_df[countries_df["Country"] != "United Kingdom"]
                if exclude_uk
                else countries_df
            )
            fig_cnt = px.bar(
                plot_countries.iloc[::-1],
                x="gross_revenue",
                y="Country",
                orientation="h",
                text="revenue_share_pct",
                title="Top Geographic Markets by Gross Revenue (£)",
                labels={"gross_revenue": "Gross Revenue (£)", "Country": "Country"},
                color_discrete_sequence=["#2ca02c"],
            )
            fig_cnt.update_traces(texttemplate="%{text:.2f}% share", textposition="outside")
            fig_cnt.update_layout(height=400)
            st.plotly_chart(fig_cnt, use_container_width=True)

        with c_right:
            st.subheader("⚠️ Top 5 Cancelled Products Audit")
            st.caption(
                "Keeping cancellation invoices (`C` prefix) in Stage 3 reveals that two single bulk orders "
                "(*Paper Craft Little Birdie* and *Medium Ceramic Jar*) were immediately cancelled, accounting "
                "for £245.6k (over 40%) of all store returns."
            )
            if not top_can_df.empty:
                st.dataframe(top_can_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # PAGE 6: MODEL PERFORMANCE & SHAP EXPLAINABILITY
    # -----------------------------------------------------------------
    elif page.startswith("6."):
        st.title("🏆 Model Evaluation & SHAP Explainability")
        st.markdown(
            "Empirical evaluation on the **20% unseen Holdout Test Set (`677` customers)** and "
            "**5-Fold Stratified Out-of-Fold Cross-Validation (`2,704` customers)**."
        )

        test_df = pd.DataFrame(model_rep.get("holdout_test_metrics", []))
        cv_df = pd.DataFrame(model_rep.get("cv_oof_metrics", []))

        if not test_df.empty:
            best_row = test_df[test_df["model_name"] == best_model_name].iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Winning Model", best_model_name)
            m2.metric("Holdout PR-AUC", f"{best_row['pr_auc']:.4f}")
            m3.metric("Holdout ROC-AUC", f"{best_row['roc_auc']:.4f}")
            m4.metric("Holdout F1-Score", f"{best_row['f1_score']:.4f}")
            m5.metric("Recall / Precision", f"{best_row['recall_score']:.1%} / {best_row['precision_score']:.1%}")

            col_a, col_b = st.columns(2)
            with col_a:
                melted = test_df.melt(
                    id_vars=["model_name"],
                    value_vars=["pr_auc", "roc_auc", "f1_score", "recall_score", "precision_score"],
                    var_name="Metric",
                    value_name="Score",
                )
                fig_mod = px.bar(
                    melted,
                    x="Metric",
                    y="Score",
                    color="model_name",
                    barmode="group",
                    title="Holdout Test Set Performance Across Candidate Models",
                )
                fig_mod.update_layout(height=420, yaxis_range=[0, 1.05])
                st.plotly_chart(fig_mod, use_container_width=True)

            with col_b:
                if not shap_df.empty:
                    top_shap = shap_df.head(12).iloc[::-1]
                    fig_sh = px.bar(
                        top_shap,
                        x="mean_abs_shap",
                        y="display_name",
                        color="directional_effect",
                        orientation="h",
                        title="Global SHAP Feature Importance (Top 12 Features)",
                        labels={"mean_abs_shap": "Mean |SHAP Value|", "display_name": "Feature"},
                        color_discrete_map={
                            "Higher value INCREASES churn risk (+)": "#d62728",
                            "Higher value DECREASES churn risk (-)": "#1f77b4",
                        },
                    )
                    fig_sh.update_layout(height=420, legend=dict(orientation="h", y=-0.22))
                    st.plotly_chart(fig_sh, use_container_width=True)

            st.subheader("Holdout Test Set Metrics Table")
            st.dataframe(test_df, use_container_width=True, hide_index=True)

            st.subheader("5-Fold Stratified Cross-Validation (Out-of-Fold) Metrics Table")
            st.dataframe(cv_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📈 Diagnostic Evaluation & SHAP Plots")
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "ROC & PR Curves",
                "Confusion Matrix & Threshold Curve",
                "SHAP Directional Beeswarm",
                "Individual Customer SHAP Profiles",
            ]
        )
        with tab1:
            p_fig6 = FIGURES_DIR / "06_model_comparison_roc_pr_curves.png"
            if p_fig6.exists():
                st.image(str(p_fig6), use_container_width=True)
        with tab2:
            p_fig7 = FIGURES_DIR / "07_confusion_matrix_and_threshold_curve.png"
            if p_fig7.exists():
                st.image(str(p_fig7), use_container_width=True)
        with tab3:
            p_fig9 = FIGURES_DIR / "09_shap_directional_beeswarm.png"
            if p_fig9.exists():
                st.image(str(p_fig9), use_container_width=True)
        with tab4:
            p_fig10 = FIGURES_DIR / "10_shap_individual_customer_profiles.png"
            if p_fig10.exists():
                st.image(str(p_fig10), use_container_width=True)

    # -----------------------------------------------------------------
    # PAGE 7: INDIVIDUAL CUSTOMER 360° EXPLORER
    # -----------------------------------------------------------------
    elif page.startswith("7."):
        st.title("🔍 Individual Customer 360° Intelligence Explorer")
        st.markdown(
            "Inspect any customer's complete behavioral profile, RFM & K-Means segment assignment, "
            "churn risk score, top 3 local SHAP drivers, and recent invoice ledger."
        )

        col_sel, col_txt = st.columns([2, 1])
        sample_ids = df_cust.sort_values("monetary_gross", ascending=False)["customer_id"].head(200).tolist()
        selected_dropdown_id = col_sel.selectbox(
            "Quick-Select from Top 200 Customers (or type any CustomerID on the right):",
            options=sample_ids,
            index=sample_ids.index("17850") if "17850" in sample_ids else 0,
        )
        manual_id = col_txt.text_input("Or Enter Exact CustomerID:", value="")
        target_cid = manual_id.strip() if manual_id.strip() else selected_dropdown_id

        with st.spinner(f"Fetching 360° profile for Customer {target_cid}..."):
            profile = fetch_customer_360_profile(target_cid, df_cust)

        if profile is None:
            st.error(f"Customer ID `{target_cid}` was not found. Try `17850`, `14646`, `12347`, or `13089`.")
        else:
            feat = profile["features"]
            seg = profile["segmentation"]
            ci = profile["churn_intelligence"]

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Customer ID", profile["customer_id"])
            c2.metric("Country", profile["country"])
            c3.metric("RFM Segment", str(seg.get("rfm_segment", "N/A")))
            c4.metric("RFM Score", str(seg.get("rfm_score_str", "N/A")))
            c5.metric("Gross Spend (£)", f"£{feat['monetary_gross']:,.2f}")
            c6.metric("Net Spend (£)", f"£{feat['monetary_net']:,.2f}")

            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Recency (Days)", f"{feat['recency_days']}d")
            b2.metric("Qualifying Orders", f"{feat['frequency_orders']}")
            b3.metric("Average Order Value", f"£{feat['avg_order_value']:,.2f}")
            b4.metric("Unique SKUs", f"{feat['unique_products']}")
            b5.metric("Avg Order Interval", f"{feat['avg_days_between_orders']:.1f}d")
            b6.metric("Cancellation Rate", f"{feat['cancellation_rate_pct']:.1f}%")

            st.markdown("---")
            left_col, right_col = st.columns([1, 1])
            with left_col:
                prob_val = ci.get("churn_probability")
                if prob_val is not None:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob_val * 100,
                            number={"suffix": "%"},
                            title={"text": f"90-Day Churn Risk ({ci.get('risk_tier')})"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#1f77b4"},
                                "steps": [
                                    {"range": [0, best_thresh * 100], "color": "#d4edda"},
                                    {"range": [best_thresh * 100, 60], "color": "#fff3cd"},
                                    {"range": [60, 100], "color": "#f8d7da"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.8,
                                    "value": best_thresh * 100,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.info(
                        "This customer was acquired on or after the `2011-09-10` prediction cutoff "
                        "(Right-Censored New Buyer) and is tracked in RFM segmentation."
                    )

            with right_col:
                st.subheader("🧠 Local SHAP Risk Drivers & Retention Action")
                drivers = ci.get("top_shap_drivers", [])
                if drivers:
                    for idx, drv in enumerate(drivers, start=1):
                        st.markdown(f"**Driver {idx}:** `{drv}`")
                st.success(f"**Recommended Action:** {ci.get('recommended_action')}")
                st.caption(
                    f"**K-Means Cluster:** `{seg.get('kmeans_segment_label')}` | "
                    f"**First Order:** `{feat.get('first_purchase_date')}` | "
                    f"**Last Order:** `{feat.get('last_purchase_date')}`"
                )

            st.subheader(f"🧾 Recent Transactions for Customer {profile['customer_id']}")
            tx_list = profile.get("recent_transactions", [])
            if tx_list:
                st.dataframe(pd.DataFrame(tx_list), use_container_width=True, hide_index=True)
            else:
                st.info("No recent transaction rows found.")

    render_author_signature()


def render_author_signature() -> None:
    """Render Sridhar Kailash Panigrahy's digital signature in sidebar and page footer."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
                    border: 1px solid #d4af37; border-radius: 10px; padding: 12px;
                    text-align: center; color: #f8f9fa; box-shadow: 0 2px 8px rgba(0,0,0,0.25);">
            <div style="font-size: 11px; letter-spacing: 1.2px; color: #d4af37; font-weight: 700;">
                DIGITAL SIGNATURE
            </div>
            <div style="font-size: 14px; font-weight: 700; margin-top: 4px; color: #ffffff;">
                Sridhar Kailash Panigrahy
            </div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 2px;">
                Data Scientist &amp; ML Engineer
            </div>
            <div style="margin-top: 6px; font-size: 12px;">
                <a href="https://github.com/DarkSri-1" target="_blank"
                   style="color: #d4af37; text-decoration: none; font-weight: 600;">
                   🔗 GitHub: @DarkSri-1
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 10px; font-size: 13px; color: #64748b;">
            ⚡ <b>CustomerPulse AI</b> — Architected &amp; Engineered by
            <b>Sridhar Kailash Panigrahy</b>
            (<a href="https://github.com/DarkSri-1" target="_blank" style="color: #2563eb;">@DarkSri-1</a>)
            | MIT License © 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_dashboard()

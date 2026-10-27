"""
Stage 10: FastAPI REST Backend for CustomerPulse AI.
Serves customer intelligence, RFM segments, store & revenue analytics,
model evaluation metrics, and real-time churn predictions.
"""
from contextlib import asynccontextmanager
import json
import logging
import sys
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Path as PathParam, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src import __author__, __github__, __signature__
from src.config import (
    APP_ENV,
    MODEL_VERSION,
    MODEL_ARTIFACT_PATH,
    EDA_REPORT_JSON_PATH,
    MODEL_REPORT_JSON_PATH,
    SHAP_GLOBAL_IMPORTANCE_PATH,
    LOGS_DIR,
)
from src.database import get_engine, test_database_connection, close_engine
from src.predictions import (
    load_model_bundle,
    predict_single_customer,
    derive_customer_risk_drivers,
)

# Configure API Logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "stage10_fastapi.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =====================================================================
# 1. Pydantic v2 Request & Response Schemas
# =====================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    environment: str = Field(..., examples=["development"])
    database_connected: bool = Field(..., examples=[True])
    model_loaded: bool = Field(..., examples=[True])
    model_name: Optional[str] = Field(None, examples=["XGBoost"])
    model_version: str = Field(..., examples=["v1.0.0"])
    author: str = Field(default=__author__, examples=["Sridhar Kailash Panigrahy"])
    github: str = Field(default=__github__, examples=["https://github.com/DarkSri-1"])


class CustomerSummaryItem(BaseModel):
    customer_id: str
    country: str
    recency_days: int
    frequency_orders: int
    monetary_gross: float
    monetary_net: float
    avg_order_value: float
    unique_products: int
    rfm_segment: Optional[str] = None
    kmeans_segment_label: Optional[str] = None
    churn_probability: Optional[float] = None
    predicted_churn: Optional[int] = None
    risk_tier: Optional[str] = None


class CustomerListResponse(BaseModel):
    total_matching: int
    limit: int
    offset: int
    customers: List[CustomerSummaryItem]


class ChurnPredictionRequest(BaseModel):
    """
    Input schema for real-time customer churn scoring (POST /predict/churn).
    Requires core behavioral metrics and supplies sensible defaults for optional fields.
    """
    recency_days: int = Field(
        ..., ge=0, le=3650, description="Days since last qualifying purchase (0 to 3650)", examples=[135]
    )
    frequency_orders: int = Field(
        ..., ge=1, le=10000, description="Number of qualifying orders placed (>= 1)", examples=[2]
    )
    total_line_items: int = Field(
        default=15, ge=1, le=100000, description="Total invoice line items purchased", examples=[15]
    )
    total_units: int = Field(
        default=120, ge=1, le=1000000, description="Total product units purchased", examples=[120]
    )
    monetary_gross: float = Field(
        ..., gt=0.0, le=10000000.0, description="Total gross spend in GBP (> 0)", examples=[420.50]
    )
    cancellations_amount: float = Field(
        default=0.0, ge=0.0, le=10000000.0, description="Total cancelled/returned amount in GBP", examples=[0.0]
    )
    monetary_net: Optional[float] = Field(
        default=None, description="Net spend after cancellations (auto-computed if omitted)", examples=[420.50]
    )
    avg_order_value: Optional[float] = Field(
        default=None, gt=0.0, description="Average Order Value in GBP (auto-computed if omitted)", examples=[210.25]
    )
    avg_days_between_orders: float = Field(
        default=30.0, ge=0.0, le=3650.0, description="Average days between orders", examples=[45.0]
    )
    unique_products: int = Field(
        default=12, ge=1, le=50000, description="Count of distinct StockCodes purchased", examples=[12]
    )
    cancellation_rate_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Cancellation percentage of gross spend", examples=[0.0]
    )
    tenure_days: int = Field(
        default=180, ge=0, le=3650, description="Days since first qualifying purchase", examples=[180]
    )
    orders_last_30d: int = Field(
        default=0, ge=0, le=1000, description="Orders placed in final 30 days before cutoff", examples=[0]
    )
    orders_last_60d: int = Field(
        default=0, ge=0, le=2000, description="Orders placed in final 60 days before cutoff", examples=[0]
    )
    spend_last_30d: float = Field(
        default=0.0, ge=0.0, le=10000000.0, description="Gross spend in final 30 days (GBP)", examples=[0.0]
    )
    spend_last_60d: float = Field(
        default=0.0, ge=0.0, le=10000000.0, description="Gross spend in final 60 days (GBP)", examples=[0.0]
    )
    spend_ratio_last_30d: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Ratio of last 30d spend to total gross spend", examples=[0.0]
    )
    country: str = Field(
        default="United Kingdom", min_length=2, max_length=60, description="Customer country", examples=["United Kingdom"]
    )

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "ChurnPredictionRequest":
        """Auto-calculate derived financial ratios if not explicitly passed by caller."""
        if self.monetary_net is None:
            self.monetary_net = round(self.monetary_gross - self.cancellations_amount, 2)
        if self.avg_order_value is None:
            self.avg_order_value = round(self.monetary_gross / max(1, self.frequency_orders), 2)
        if self.cancellation_rate_pct is None:
            self.cancellation_rate_pct = round(
                min(100.0, (self.cancellations_amount / self.monetary_gross) * 100.0), 2
            )
        if self.spend_ratio_last_30d is None:
            self.spend_ratio_last_30d = round(
                min(1.0, self.spend_last_30d / self.monetary_gross), 4
            )
        if self.tenure_days < self.recency_days:
            self.tenure_days = self.recency_days
        return self


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    predicted_churn: int
    decision_threshold: float
    risk_tier: str
    model_name: str
    model_version: str
    top_risk_indicators: List[str]
    recommended_action: str


# =====================================================================
# 2. Application Lifespan (Startup Model & DB Initialization)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained ML model bundle once when FastAPI starts up,
    and cleanly close the SQLAlchemy connection pool on shutdown.
    """
    logger.info(f"Starting CustomerPulse AI FastAPI Backend | {__signature__}")
    app.state.db_ready = test_database_connection()

    if MODEL_ARTIFACT_PATH.exists():
        try:
            app.state.model_bundle = load_model_bundle()
            logger.info(
                f"[OK] Loaded ML model bundle into memory: "
                f"{app.state.model_bundle['model_name']} ({app.state.model_bundle['model_version']})"
            )
        except Exception as exc:
            logger.error(f"Failed to load model artifact on startup: {exc}")
            app.state.model_bundle = None
    else:
        logger.warning(f"Model artifact not found at {MODEL_ARTIFACT_PATH}")
        app.state.model_bundle = None

    yield

    logger.info("Shutting down CustomerPulse AI FastAPI Backend...")
    close_engine()


app = FastAPI(
    title="CustomerPulse AI — E-Commerce Intelligence & Churn Prediction API",
    description=(
        "Production REST API serving verified store analytics, RFM customer segmentation, "
        f"SHAP model explainability, and real-time customer churn predictions.\n\n**{__signature__}**"
    ),
    version=MODEL_VERSION,
    contact={
        "name": __author__,
        "url": __github__,
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_digital_signature_header(request: Request, call_next):
    """Attach author digital signature header to all HTTP responses."""
    response = await call_next(request)
    response.headers["X-Developed-By"] = f"{__author__} ({__github__})"
    return response


# =====================================================================
# 3. Helper Function for Retention Action Mapping
# =====================================================================

def get_retention_recommendation(risk_tier: str, recency_days: int, frequency_orders: int) -> str:
    """Return a concrete, business-grounded retention action based on risk profile."""
    if risk_tier == "High Risk":
        if frequency_orders >= 3:
            return (
                "Priority VIP Win-Back: Trigger immediate account manager outreach and "
                "offer a tiered wholesale restock incentive before the 90-day inactivity mark."
            )
        return (
            "Automated Win-Back Campaign: Send personalized product bundle recommendation "
            "featuring top anchor SKUs (Regency Cakestand / Heart T-Light Holder)."
        )
    elif risk_tier == "Medium Risk":
        return (
            "Proactive Nurture: Schedule mid-cycle reorder reminder email on Tuesday-Thursday "
            "morning (10:00-12:00 UK time) aligned with peak wholesale ordering hours."
        )
    return (
        "Healthy Retention: Maintain standard loyalty cadence and cross-sell complementary "
        "seasonal lines ahead of Q4."
    )


# =====================================================================
# 4. REST API Endpoints
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request) -> HealthResponse:
    """
    Check API health, MySQL database connectivity, in-memory ML model readiness, and author metadata.
    """
    db_ok = test_database_connection()
    bundle = getattr(request.app.state, "model_bundle", None)
    model_loaded = bundle is not None

    return HealthResponse(
        status="healthy" if (db_ok and model_loaded) else "degraded",
        environment=APP_ENV,
        database_connected=db_ok,
        model_loaded=model_loaded,
        model_name=bundle["model_name"] if model_loaded else None,
        model_version=bundle["model_version"] if model_loaded else MODEL_VERSION,
        author=__author__,
        github=__github__,
    )


@app.get("/customers", response_model=CustomerListResponse, tags=["Customers"])
def list_customers(
    limit: int = Query(50, ge=1, le=500, description="Number of customers to return (1-500)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    segment: Optional[str] = Query(None, description="Filter by rfm_segment (e.g., Champions)"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk_tier (High Risk, Medium Risk, Low Risk)"),
    country: Optional[str] = Query(None, description="Filter by customer country"),
    sort_by: str = Query(
        "monetary_gross",
        pattern="^(monetary_gross|recency_days|frequency_orders|churn_probability)$",
        description="Column to sort by",
    ),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort direction (asc or desc)"),
) -> CustomerListResponse:
    """
    Retrieve a paginated, filterable list of customers joining features, RFM segments,
    and latest churn predictions from MySQL.
    """
    where_clauses = ["1=1"]
    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if segment:
        where_clauses.append("cs.rfm_segment = :segment")
        params["segment"] = segment
    if risk_tier:
        where_clauses.append("cp.risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if country:
        where_clauses.append("cf.country = :country")
        params["country"] = country

    where_sql = " AND ".join(where_clauses)

    sort_col_map = {
        "monetary_gross": "cf.monetary_gross",
        "recency_days": "cf.recency_days",
        "frequency_orders": "cf.frequency_orders",
        "churn_probability": "COALESCE(cp.churn_probability, 0.0)",
    }
    order_col = sort_col_map[sort_by]
    order_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

    count_query = text(
        f"""
        SELECT COUNT(*)
        FROM customer_features cf
        LEFT JOIN customer_segments cs ON cf.customer_id = cs.customer_id
        LEFT JOIN churn_predictions cp ON cf.customer_id = cp.customer_id
        WHERE {where_sql};
        """
    )

    data_query = text(
        f"""
        SELECT
            cf.customer_id,
            cf.country,
            cf.recency_days,
            cf.frequency_orders,
            cf.monetary_gross,
            cf.monetary_net,
            cf.avg_order_value,
            cf.unique_products,
            cs.rfm_segment,
            cs.kmeans_segment_label,
            cp.churn_probability,
            cp.predicted_churn,
            cp.risk_tier
        FROM customer_features cf
        LEFT JOIN customer_segments cs ON cf.customer_id = cs.customer_id
        LEFT JOIN churn_predictions cp ON cf.customer_id = cp.customer_id
        WHERE {where_sql}
        ORDER BY {order_col} {order_dir}
        LIMIT :limit OFFSET :offset;
        """
    )

    try:
        engine = get_engine(include_db=True)
        with engine.connect() as conn:
            total_matching = int(conn.execute(count_query, params).scalar() or 0)
            rows = conn.execute(data_query, params).mappings().all()

        items = [
            CustomerSummaryItem(
                customer_id=str(r["customer_id"]),
                country=str(r["country"]),
                recency_days=int(r["recency_days"]),
                frequency_orders=int(r["frequency_orders"]),
                monetary_gross=float(r["monetary_gross"]),
                monetary_net=float(r["monetary_net"]),
                avg_order_value=float(r["avg_order_value"]),
                unique_products=int(r["unique_products"]),
                rfm_segment=r["rfm_segment"],
                kmeans_segment_label=r["kmeans_segment_label"],
                churn_probability=float(r["churn_probability"]) if r["churn_probability"] is not None else None,
                predicted_churn=int(r["predicted_churn"]) if r["predicted_churn"] is not None else None,
                risk_tier=r["risk_tier"],
            )
            for r in rows
        ]
        return CustomerListResponse(
            total_matching=total_matching,
            limit=limit,
            offset=offset,
            customers=items,
        )
    except SQLAlchemyError as exc:
        logger.error(f"Database error in GET /customers: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database query failed while retrieving customer list.",
        )


@app.get("/customers/{customer_id}", tags=["Customers"])
def get_customer_detail(
    customer_id: str = PathParam(..., min_length=1, max_length=20, description="Unique CustomerID (e.g., 17850)")
) -> Dict[str, Any]:
    """
    Retrieve a complete 360-degree profile for a single customer:
    behavioral features, RFM & K-Means segment, churn risk score, top 3 SHAP drivers,
    and their 15 most recent transactions.
    """
    profile_query = text(
        """
        SELECT
            cf.customer_id,
            cf.country,
            cf.is_uk,
            cf.recency_days,
            cf.frequency_orders,
            cf.total_line_items,
            cf.total_units,
            cf.monetary_gross,
            cf.cancellations_amount,
            cf.monetary_net,
            cf.avg_order_value,
            cf.avg_days_between_orders,
            cf.unique_products,
            cf.cancellation_rate_pct,
            cf.tenure_days,
            cf.first_purchase_date,
            cf.last_purchase_date,
            cs.r_score,
            cs.f_score,
            cs.m_score,
            cs.rfm_score_str,
            cs.rfm_score_total,
            cs.rfm_segment,
            cs.kmeans_cluster,
            cs.kmeans_segment_label,
            cp.observation_cutoff_date,
            cp.churn_probability,
            cp.predicted_churn,
            cp.actual_churn,
            cp.risk_tier,
            cp.model_name,
            cp.model_version,
            cp.top_driver_1,
            cp.top_driver_2,
            cp.top_driver_3
        FROM customer_features cf
        LEFT JOIN customer_segments cs ON cf.customer_id = cs.customer_id
        LEFT JOIN churn_predictions cp ON cf.customer_id = cp.customer_id
        WHERE cf.customer_id = :customer_id;
        """
    )

    tx_query = text(
        """
        SELECT
            invoice_no,
            stock_code,
            description,
            quantity,
            invoice_date,
            unit_price,
            line_total,
            transaction_type
        FROM transactions
        WHERE customer_id = :customer_id
        ORDER BY invoice_date DESC
        LIMIT 15;
        """
    )

    try:
        engine = get_engine(include_db=True)
        with engine.connect() as conn:
            profile = conn.execute(profile_query, {"customer_id": customer_id}).mappings().first()
            if not profile:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Customer '{customer_id}' not found.",
                )
            recent_tx = conn.execute(tx_query, {"customer_id": customer_id}).mappings().all()

        risk_tier = profile["risk_tier"] or "Unscored (Post-Cutoff New Buyer)"
        recommendation = get_retention_recommendation(
            risk_tier=risk_tier,
            recency_days=int(profile["recency_days"]),
            frequency_orders=int(profile["frequency_orders"]),
        )

        return {
            "customer_id": str(profile["customer_id"]),
            "country": str(profile["country"]),
            "is_uk": bool(profile["is_uk"]),
            "features": {
                "recency_days": int(profile["recency_days"]),
                "frequency_orders": int(profile["frequency_orders"]),
                "total_line_items": int(profile["total_line_items"]),
                "total_units": int(profile["total_units"]),
                "monetary_gross": float(profile["monetary_gross"]),
                "cancellations_amount": float(profile["cancellations_amount"]),
                "monetary_net": float(profile["monetary_net"]),
                "avg_order_value": float(profile["avg_order_value"]),
                "avg_days_between_orders": float(profile["avg_days_between_orders"]),
                "unique_products": int(profile["unique_products"]),
                "cancellation_rate_pct": float(profile["cancellation_rate_pct"]),
                "tenure_days": int(profile["tenure_days"]),
                "first_purchase_date": str(profile["first_purchase_date"]),
                "last_purchase_date": str(profile["last_purchase_date"]),
            },
            "segmentation": {
                "r_score": int(profile["r_score"]) if profile["r_score"] is not None else None,
                "f_score": int(profile["f_score"]) if profile["f_score"] is not None else None,
                "m_score": int(profile["m_score"]) if profile["m_score"] is not None else None,
                "rfm_score_str": profile["rfm_score_str"],
                "rfm_score_total": int(profile["rfm_score_total"]) if profile["rfm_score_total"] is not None else None,
                "rfm_segment": profile["rfm_segment"],
                "kmeans_cluster": int(profile["kmeans_cluster"]) if profile["kmeans_cluster"] is not None else None,
                "kmeans_segment_label": profile["kmeans_segment_label"],
            },
            "churn_intelligence": {
                "scored_in_pre_cutoff_cohort": profile["churn_probability"] is not None,
                "observation_cutoff_date": str(profile["observation_cutoff_date"]) if profile["observation_cutoff_date"] else None,
                "churn_probability": float(profile["churn_probability"]) if profile["churn_probability"] is not None else None,
                "predicted_churn": int(profile["predicted_churn"]) if profile["predicted_churn"] is not None else None,
                "actual_churn": int(profile["actual_churn"]) if profile["actual_churn"] is not None else None,
                "risk_tier": risk_tier,
                "model_name": profile["model_name"],
                "model_version": profile["model_version"],
                "top_shap_drivers": [
                    d for d in [profile["top_driver_1"], profile["top_driver_2"], profile["top_driver_3"]] if d
                ],
                "recommended_action": recommendation,
            },
            "recent_transactions": [
                {
                    "invoice_no": str(t["invoice_no"]),
                    "stock_code": str(t["stock_code"]),
                    "description": str(t["description"]),
                    "quantity": int(t["quantity"]),
                    "invoice_date": str(t["invoice_date"]),
                    "unit_price": float(t["unit_price"]),
                    "line_total": float(t["line_total"]),
                    "transaction_type": str(t["transaction_type"]),
                }
                for t in recent_tx
            ],
        }
    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        logger.error(f"Database error in GET /customers/{customer_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while retrieving customer profile.",
        )


@app.get("/analytics/overview", tags=["Analytics"])
def get_analytics_overview() -> Dict[str, Any]:
    """
    Return executive business KPIs, RFM segment distribution, K-Means cluster summary,
    and churn risk tier breakdown from MySQL and verified EDA artifacts.
    """
    try:
        engine = get_engine(include_db=True)
        with engine.connect() as conn:
            seg_rows = conn.execute(
                text(
                    """
                    SELECT
                        cs.rfm_segment,
                        COUNT(*) AS customer_count,
                        ROUND(AVG(cf.recency_days), 2) AS avg_recency_days,
                        ROUND(AVG(cf.frequency_orders), 2) AS avg_frequency_orders,
                        ROUND(AVG(cf.monetary_gross), 2) AS avg_monetary_gross,
                        ROUND(SUM(cf.monetary_gross), 2) AS total_revenue_gbp
                    FROM customer_segments cs
                    JOIN customer_features cf ON cs.customer_id = cf.customer_id
                    GROUP BY cs.rfm_segment
                    ORDER BY total_revenue_gbp DESC;
                    """
                )
            ).mappings().all()

            risk_rows = conn.execute(
                text(
                    """
                    SELECT
                        cp.risk_tier,
                        COUNT(*) AS customer_count,
                        ROUND(AVG(cp.churn_probability), 4) AS avg_churn_probability,
                        ROUND(SUM(cf.monetary_gross), 2) AS revenue_at_risk_gbp
                    FROM churn_predictions cp
                    JOIN customer_features cf ON cp.customer_id = cf.customer_id
                    GROUP BY cp.risk_tier
                    ORDER BY avg_churn_probability DESC;
                    """
                )
            ).mappings().all()

        kpis = {}
        if EDA_REPORT_JSON_PATH.exists():
            with open(EDA_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
                kpis = json.load(f).get("kpis", {})

        return {
            "kpis": kpis,
            "rfm_segments": [dict(r) for r in seg_rows],
            "churn_risk_breakdown": [dict(r) for r in risk_rows],
        }
    except SQLAlchemyError as exc:
        logger.error(f"Database error in GET /analytics/overview: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics overview from database.",
        )


@app.get("/analytics/revenue", tags=["Analytics"])
def get_revenue_analytics() -> Dict[str, Any]:
    """
    Return monthly gross/net revenue trends, top-selling products, highest-revenue products,
    country performance, and day/hour seasonality.
    """
    if EDA_REPORT_JSON_PATH.exists():
        with open(EDA_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            eda = json.load(f)
        return {
            "monthly_revenue_trend": eda.get("monthly_revenue_trend", []),
            "top_10_highest_revenue_products": eda.get("top_10_highest_revenue_products", []),
            "top_10_selling_products_by_qty": eda.get("top_10_selling_products_by_qty", []),
            "top_10_countries_by_revenue": eda.get("top_10_countries_by_revenue", []),
            "day_of_week_performance": eda.get("day_of_week_performance", []),
            "hourly_performance": eda.get("hourly_performance", []),
            "top_5_cancelled_products": eda.get("top_5_cancelled_products", []),
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="EDA revenue report not found. Run `python -m src.eda` first.",
    )


@app.post("/predict/churn", response_model=ChurnPredictionResponse, tags=["Machine Learning"])
def predict_customer_churn(
    payload: ChurnPredictionRequest,
    request: Request,
) -> ChurnPredictionResponse:
    """
    Score a customer's behavioral feature profile in real time using the cached
    production ML pipeline and return churn probability, binary prediction,
    risk tier, top behavioral indicators, and a recommended retention action.
    """
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        try:
            bundle = load_model_bundle()
            request.app.state.model_bundle = bundle
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained churn model artifact is not available. Run Stage 8 training first.",
            )

    feature_dict = payload.model_dump()
    result = predict_single_customer(feature_dict, bundle=bundle)

    benchmark_medians = {
        "recency_days": 75.0,
        "frequency_orders": 2.0,
        "monetary_gross": 650.0,
    }
    d1, d2, d3 = derive_customer_risk_drivers(pd.Series(feature_dict), benchmark_medians)
    recommendation = get_retention_recommendation(
        risk_tier=result["risk_tier"],
        recency_days=payload.recency_days,
        frequency_orders=payload.frequency_orders,
    )

    return ChurnPredictionResponse(
        churn_probability=result["churn_probability"],
        predicted_churn=result["predicted_churn"],
        decision_threshold=result["decision_threshold"],
        risk_tier=result["risk_tier"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        top_risk_indicators=[d1, d2, d3],
        recommended_action=recommendation,
    )


@app.get("/models/metrics", tags=["Machine Learning"])
def get_model_evaluation_metrics() -> Dict[str, Any]:
    """
    Return all evaluated ML model metrics (Holdout Test Set + 5-Fold OOF CV)
    and top global SHAP feature importance rankings.
    """
    try:
        engine = get_engine(include_db=True)
        with engine.connect() as conn:
            db_metrics = conn.execute(
                text(
                    """
                    SELECT
                        model_name,
                        model_version,
                        dataset_split,
                        decision_threshold,
                        precision_score,
                        recall_score,
                        f1_score,
                        pr_auc,
                        roc_auc,
                        true_positives,
                        false_positives,
                        true_negatives,
                        false_negatives,
                        is_best_model
                    FROM model_metrics
                    ORDER BY dataset_split DESC, pr_auc DESC;
                    """
                )
            ).mappings().all()
        metrics_list = [dict(r) for r in db_metrics]
    except SQLAlchemyError:
        metrics_list = []

    if not metrics_list and MODEL_REPORT_JSON_PATH.exists():
        with open(MODEL_REPORT_JSON_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
        metrics_list = rep.get("holdout_test_metrics", []) + rep.get("cv_oof_metrics", [])

    shap_features = []
    if SHAP_GLOBAL_IMPORTANCE_PATH.exists():
        df_shap = pd.read_csv(SHAP_GLOBAL_IMPORTANCE_PATH)
        shap_features = df_shap.to_dict(orient="records")

    return {
        "model_version": MODEL_VERSION,
        "evaluation_metrics": metrics_list,
        "global_shap_importance": shap_features,
    }

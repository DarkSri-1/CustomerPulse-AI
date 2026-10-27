"""
Central configuration and path management for CustomerPulse AI.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# Raw Dataset Path
RAW_DATASET_FILENAME = "Online Retail.xlsx"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME

# Processed Dataset Paths (Stage 3 outputs)
CLEANED_TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "cleaned_transactions.csv"
CUSTOMER_TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "customer_transactions.csv"
CLEANING_REPORT_JSON_PATH = REPORTS_DIR / "cleaning_summary.json"
CLEANING_REPORT_MD_PATH = REPORTS_DIR / "cleaning_summary.md"

# EDA Output Paths (Stage 4 outputs)
EDA_REPORT_JSON_PATH = REPORTS_DIR / "eda_summary.json"
EDA_REPORT_MD_PATH = REPORTS_DIR / "eda_summary.md"

# Feature & Segmentation Paths (Stage 6 outputs)
CUSTOMER_FEATURES_PATH = PROCESSED_DATA_DIR / "customer_features.csv"
CUSTOMER_SEGMENTS_PATH = PROCESSED_DATA_DIR / "customer_segments.csv"

# Churn Modeling Dataset & Report Paths (Stage 7 outputs)
CHURN_CUTOFF_DATE = "2011-09-10"
CHURN_OUTCOME_DAYS = 90
CHURN_DATASET_PATH = PROCESSED_DATA_DIR / "churn_modeling_dataset.csv"
CHURN_REPORT_JSON_PATH = REPORTS_DIR / "churn_definition_summary.json"
CHURN_REPORT_MD_PATH = REPORTS_DIR / "churn_definition_summary.md"

# Model Training, Evaluation & Predictions Paths (Stage 8 outputs)
MODEL_VERSION = "v1.0.0"
MODEL_ARTIFACT_PATH = MODELS_DIR / "churn_model_v1.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
CHURN_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "churn_predictions.csv"
MODEL_REPORT_JSON_PATH = REPORTS_DIR / "model_evaluation_report.json"
MODEL_REPORT_MD_PATH = REPORTS_DIR / "model_evaluation_report.md"

# Model Explainability Paths (Stage 9 outputs)
SHAP_GLOBAL_IMPORTANCE_PATH = PROCESSED_DATA_DIR / "shap_global_importance.csv"
SHAP_REPORT_JSON_PATH = REPORTS_DIR / "shap_explainability_report.json"
SHAP_REPORT_MD_PATH = REPORTS_DIR / "shap_explainability_report.md"

# Database Configuration (Stage 5+)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "customerpulse_db")

# General Settings
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RANDOM_SEED = 42

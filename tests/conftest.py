"""
Shared Pytest fixtures for CustomerPulse AI unit and integration tests.
Uses small, deterministic synthetic datasets so tests execute in milliseconds.
"""
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.fixture(scope="session")
def raw_transactions_fixture() -> pd.DataFrame:
    """
    Create a 22-row synthetic raw dataset containing:
      - 6 active pre-cutoff customers (C001..C006) with varying RFM profiles
      - 1 right-censored customer (99001) acquired AFTER 2011-09-10 cutoff
      - 1 exact duplicate row
      - 1 valid cancellation invoice ('C536391' with negative Quantity)
      - 1 bad debt adjustment invoice ('A563185')
      - 1 warehouse damage row (negative Quantity WITHOUT 'C' prefix and UnitPrice = 0)
      - 1 non-product postage row ('POST')
      - 1 guest checkout row (CustomerID = NaN)
      - 1 invalid date row ('INVALID_DATE_STRING')
      - Inconsistent lowercase/whitespace product descriptions for StockCode '85123A'
    """
    rows = [
        # Customer 10001: High-frequency Champion, buys before AND after 2011-09-10 cutoff (Retained = 0)
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": " white hanging heart t-light holder ", "Quantity": 12, "InvoiceDate": "2011-01-15 10:00:00", "UnitPrice": 2.55, "CustomerID": 10001.0, "Country": "United Kingdom"},
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": " white hanging heart t-light holder ", "Quantity": 12, "InvoiceDate": "2011-01-15 10:00:00", "UnitPrice": 2.55, "CustomerID": 10001.0, "Country": "United Kingdom"},  # Exact duplicate
        {"InvoiceNo": "545001", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 24, "InvoiceDate": "2011-05-10 11:30:00", "UnitPrice": 2.55, "CustomerID": 10001.0, "Country": "United Kingdom"},
        {"InvoiceNo": "558001", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 10, "InvoiceDate": "2011-08-25 14:15:00", "UnitPrice": 12.75, "CustomerID": 10001.0, "Country": "United Kingdom"},
        {"InvoiceNo": "570001", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 6, "InvoiceDate": "2011-10-12 09:45:00", "UnitPrice": 12.75, "CustomerID": 10001.0, "Country": "United Kingdom"},  # Outcome window purchase

        # Customer 10002: Loyal customer, buys before AND after cutoff + 1 cancellation (Retained = 0)
        {"InvoiceNo": "540010", "StockCode": "47566", "Description": "PARTY BUNTING", "Quantity": 20, "InvoiceDate": "2011-03-01 12:00:00", "UnitPrice": 4.95, "CustomerID": 10002.0, "Country": "Germany"},
        {"InvoiceNo": "C540099", "StockCode": "47566", "Description": "PARTY BUNTING", "Quantity": -2, "InvoiceDate": "2011-03-05 15:00:00", "UnitPrice": 4.95, "CustomerID": 10002.0, "Country": "Germany"},  # Valid cancellation
        {"InvoiceNo": "559010", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 15, "InvoiceDate": "2011-08-18 10:20:00", "UnitPrice": 2.55, "CustomerID": 10002.0, "Country": "Germany"},
        {"InvoiceNo": "572010", "StockCode": "47566", "Description": "PARTY BUNTING", "Quantity": 10, "InvoiceDate": "2011-11-02 16:00:00", "UnitPrice": 4.95, "CustomerID": 10002.0, "Country": "Germany"},  # Outcome window purchase

        # Customer 10003: Single-order early customer, no orders after cutoff (Churned = 1)
        {"InvoiceNo": "538001", "StockCode": "84879", "Description": "ASSORTED COLOUR BIRD ORNAMENT", "Quantity": 8, "InvoiceDate": "2010-12-12 11:00:00", "UnitPrice": 1.69, "CustomerID": 10003.0, "Country": "United Kingdom"},

        # Customer 10004: Two orders prior to cutoff, zero orders in outcome window (Churned = 1)
        {"InvoiceNo": "542001", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 4, "InvoiceDate": "2011-02-10 13:00:00", "UnitPrice": 12.75, "CustomerID": 10004.0, "Country": "France"},
        {"InvoiceNo": "550001", "StockCode": "47566", "Description": "PARTY BUNTING", "Quantity": 6, "InvoiceDate": "2011-04-20 13:00:00", "UnitPrice": 4.95, "CustomerID": 10004.0, "Country": "France"},

        # Customer 10005: Moderate buyer prior to cutoff, zero orders in outcome window (Churned = 1)
        {"InvoiceNo": "552001", "StockCode": "85099B", "Description": "JUMBO BAG RED RETROSPOT", "Quantity": 30, "InvoiceDate": "2011-06-01 09:30:00", "UnitPrice": 2.08, "CustomerID": 10005.0, "Country": "United Kingdom"},

        # Customer 10006: Recent pre-cutoff buyer, also buys in outcome window (Retained = 0)
        {"InvoiceNo": "560001", "StockCode": "85099B", "Description": "JUMBO BAG RED RETROSPOT", "Quantity": 50, "InvoiceDate": "2011-09-01 10:00:00", "UnitPrice": 2.08, "CustomerID": 10006.0, "Country": "EIRE"},
        {"InvoiceNo": "575001", "StockCode": "85099B", "Description": "JUMBO BAG RED RETROSPOT", "Quantity": 40, "InvoiceDate": "2011-11-18 11:00:00", "UnitPrice": 2.08, "CustomerID": 10006.0, "Country": "EIRE"},

        # Customer 99001: Right-Censored Customer (First purchase AFTER 2011-09-10 cutoff!)
        {"InvoiceNo": "576001", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 5, "InvoiceDate": "2011-10-25 14:00:00", "UnitPrice": 12.75, "CustomerID": 99001.0, "Country": "United Kingdom"},

        # Guest checkout row (Missing CustomerID -> kept in store df, excluded from customer df)
        {"InvoiceNo": "555999", "StockCode": "85123A", "Description": None, "Quantity": 3, "InvoiceDate": "2011-07-10 12:00:00", "UnitPrice": 2.55, "CustomerID": np.nan, "Country": "United Kingdom"},

        # Anomalies to be filtered out by Stage 3 cleaning rules:
        # 1) Non-product StockCode ('POST')
        {"InvoiceNo": "540010", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 1, "InvoiceDate": "2011-03-01 12:00:00", "UnitPrice": 18.00, "CustomerID": 10002.0, "Country": "Germany"},
        # 2) Bad debt adjustment ('A' invoice with negative price)
        {"InvoiceNo": "A563185", "StockCode": "B", "Description": "Adjust bad debt", "Quantity": 1, "InvoiceDate": "2011-08-12 14:50:00", "UnitPrice": -11062.06, "CustomerID": np.nan, "Country": "United Kingdom"},
        # 3) Warehouse damage write-off (Negative quantity without 'C' invoice & UnitPrice == 0)
        {"InvoiceNo": "550999", "StockCode": "22423", "Description": "damaged", "Quantity": -15, "InvoiceDate": "2011-05-01 09:00:00", "UnitPrice": 0.0, "CustomerID": np.nan, "Country": "United Kingdom"},
        # 4) Zero UnitPrice row
        {"InvoiceNo": "551000", "StockCode": "84879", "Description": "ASSORTED COLOUR BIRD ORNAMENT", "Quantity": 5, "InvoiceDate": "2011-05-02 10:00:00", "UnitPrice": 0.0, "CustomerID": 10003.0, "Country": "United Kingdom"},
        # 5) Invalid date string
        {"InvoiceNo": "551001", "StockCode": "84879", "Description": "ASSORTED COLOUR BIRD ORNAMENT", "Quantity": 5, "InvoiceDate": "NOT_A_VALID_DATE", "UnitPrice": 1.69, "CustomerID": 10003.0, "Country": "United Kingdom"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def api_client() -> Generator[TestClient, None, None]:
    """
    Create a FastAPI TestClient that executes the lifespan startup and shutdown hooks.
    """
    with TestClient(app) as client:
        yield client

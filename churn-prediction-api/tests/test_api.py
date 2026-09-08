import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 85.5,
    "total_charges": 1026.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_fields(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert "risk_tier" in data
    assert "top_factors" in data


def test_predict_probability_range(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_tier_valid(client):
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    tier = response.json()["risk_tier"]
    assert tier in ["High", "Medium", "Low"]


def test_predict_invalid_gender(client):
    bad_input = {**SAMPLE_CUSTOMER, "gender": "Unknown"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_negative_tenure(client):
    bad_input = {**SAMPLE_CUSTOMER, "tenure_months": -1}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

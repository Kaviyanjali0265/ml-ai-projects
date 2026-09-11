import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

SAMPLE_LOAN = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500.0,
    "LoanAmount": 120.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban",
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
    response = client.post("/predict", json=SAMPLE_LOAN)
    assert response.status_code == 200


def test_predict_response_fields(client):
    response = client.post("/predict", json=SAMPLE_LOAN)
    data = response.json()
    assert "approval_probability" in data
    assert "prediction" in data
    assert "decision" in data


def test_predict_probability_range(client):
    response = client.post("/predict", json=SAMPLE_LOAN)
    prob = response.json()["approval_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_decision_valid(client):
    response = client.post("/predict", json=SAMPLE_LOAN)
    decision = response.json()["decision"]
    assert decision in ["Approved", "Rejected"]


def test_predict_invalid_gender(client):
    bad_input = {**SAMPLE_LOAN, "Gender": "Unknown"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_predict_negative_income(client):
    bad_input = {**SAMPLE_LOAN, "ApplicantIncome": -100}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422

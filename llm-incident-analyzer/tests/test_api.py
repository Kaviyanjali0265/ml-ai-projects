import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

client = TestClient(app)

MOCK_ANALYSIS = {
    "root_cause": "Database connection pool exhausted due to long-running queries",
    "severity": "high",
    "affected_services": ["api-server", "database"],
    "recommended_fix": "Restart the database service and increase connection pool size",
    "estimated_resolution_time": "15 minutes"
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("api.main.analyze_incident", return_value=MOCK_ANALYSIS)
def test_analyze_returns_valid_structure(mock_analyze):
    response = client.post("/analyze", json={
        "description": "API server is returning 500 errors and database connections are timing out"
    })
    assert response.status_code == 200
    data = response.json()
    assert "root_cause" in data
    assert "severity" in data
    assert "affected_services" in data
    assert "recommended_fix" in data
    assert "estimated_resolution_time" in data


@patch("api.main.analyze_incident", return_value=MOCK_ANALYSIS)
def test_severity_is_valid_value(mock_analyze):
    response = client.post("/analyze", json={
        "description": "Critical outage: all services down, users cannot login"
    })
    assert response.status_code == 200
    assert response.json()["severity"] in ["low", "medium", "high", "critical"]


@patch("api.main.analyze_incident", return_value=MOCK_ANALYSIS)
def test_affected_services_is_list(mock_analyze):
    response = client.post("/analyze", json={
        "description": "Memory usage spiked to 95% on the auth service"
    })
    assert response.status_code == 200
    assert isinstance(response.json()["affected_services"], list)


def test_analyze_rejects_short_description():
    response = client.post("/analyze", json={"description": "crash"})
    assert response.status_code == 422


def test_analyze_rejects_empty_description():
    response = client.post("/analyze", json={"description": ""})
    assert response.status_code == 422


@patch("api.main.analyze_incident", return_value={"error": "LLM failed"})
def test_analyze_returns_500_on_llm_error(mock_analyze):
    response = client.post("/analyze", json={
        "description": "Something went wrong with the production database cluster"
    })
    assert response.status_code == 500

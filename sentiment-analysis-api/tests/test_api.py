import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_positive_review(client):
    response = client.post("/predict", json={"text": "This movie was absolutely fantastic! I loved every minute of it."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "POSITIVE"


def test_negative_review(client):
    response = client.post("/predict", json={"text": "Terrible film. Complete waste of time. Worst movie I have ever seen."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "NEGATIVE"


def test_response_fields(client):
    response = client.post("/predict", json={"text": "Great movie!"})
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert "positive_score" in data
    assert "negative_score" in data


def test_confidence_range(client):
    response = client.post("/predict", json={"text": "It was okay I guess."})
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_scores_sum_to_one(client):
    response = client.post("/predict", json={"text": "Amazing experience!"})
    data = response.json()
    total = data["positive_score"] + data["negative_score"]
    assert abs(total - 1.0) < 0.01


def test_empty_text_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

MOCK_RESULTS = [
    {
        "id": "kb001",
        "title": "Database Connection Pooling",
        "tags": "database, performance, backend",
        "content": "Connection pooling maintains a pool of reusable database connections.",
        "similarity_score": 0.92
    },
    {
        "id": "kb010",
        "title": "SQL Query Optimization",
        "tags": "database, sql, performance",
        "content": "Use EXPLAIN ANALYZE to inspect query plans.",
        "similarity_score": 0.87
    }
]


@pytest.fixture
def client():
    with patch("src.indexer.index_knowledge_base"):
        from api.main import app
        return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("api.main.search", return_value=MOCK_RESULTS)
def test_search_returns_results(mock_search, client):
    response = client.post("/search", json={"query": "database connection issue"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "database connection issue"
    assert len(data["results"]) == 2


@patch("api.main.search", return_value=MOCK_RESULTS)
def test_search_result_has_required_fields(mock_search, client):
    response = client.post("/search", json={"query": "how to cache in redis"})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert "id" in result
    assert "title" in result
    assert "content" in result
    assert "similarity_score" in result
    assert "tags" in result


@patch("api.main.search", return_value=MOCK_RESULTS)
def test_similarity_score_between_0_and_1(mock_search, client):
    response = client.post("/search", json={"query": "python memory leak"})
    assert response.status_code == 200
    for r in response.json()["results"]:
        assert 0.0 <= r["similarity_score"] <= 1.0


def test_search_rejects_short_query(client):
    response = client.post("/search", json={"query": "db"})
    assert response.status_code == 422


def test_search_rejects_empty_query(client):
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422


@patch("api.main.search", return_value=MOCK_RESULTS)
def test_top_k_parameter(mock_search, client):
    response = client.post("/search", json={"query": "api rate limiting", "top_k": 5})
    assert response.status_code == 200

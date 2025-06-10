import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.base_models import TextRequest, QueryRequest

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "endpoint,request_data",
    [
        ("/is_ai_generated", {"text": "Test text for AI detection"}),
        ("/text2sql", {"question": "Show me all users"}),
    ],
)
def test_endpoints_with_valid_input(endpoint, request_data, monkeypatch):
    # Mock the actual processing functions
    if endpoint == "/is_ai_generated":
        monkeypatch.setattr("app.main.is_ai_generated", lambda text: True)
    elif endpoint == "/text2sql":
        monkeypatch.setattr("app.main.text2sql", lambda question: "SELECT * FROM users")

    response = client.post(endpoint, json=request_data)
    assert response.status_code == 200

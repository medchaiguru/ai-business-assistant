from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.components.models import MetricsResponse
from app.main import app

client = TestClient(app)

def test_get_metrics() -> None:
    """Test the metrics endpoint."""
    # Mock the metrics manager in app state
    mock_metrics = MagicMock()
    mock_metrics.get = AsyncMock(return_value=MetricsResponse(
        requests_total=10,
        tokens_total=100,
        average_response_time=0.5,
        cost_total=1.25,
    ))

    # Inject the mock
    app.state.metrics = mock_metrics

    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["requests_total"] == 10
    assert data["tokens_total"] == 100
    assert data["average_response_time"] == 0.5
    assert data["cost_total"] == 1.25

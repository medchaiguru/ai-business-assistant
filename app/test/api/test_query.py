from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_query_endpoint() -> None:
    """Test the chat query endpoint."""
    # Mock the RAG Graph
    mock_rag = MagicMock()
    mock_rag.ainvoke = AsyncMock(return_value={
        "answer": "Test Answer",
        "sources": ["doc1"],
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })

    # Inject the mock
    app.state.rag_graph_instance = mock_rag

    payload = {"question": "Hello?"}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test Answer"
    assert data["sources"] == ["doc1"]

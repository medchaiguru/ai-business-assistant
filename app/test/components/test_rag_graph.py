from unittest.mock import AsyncMock

import pytest

from app.components.rag_chain import RAGChain
from app.components.semantic_cache import SemanticCache
from app.components.workflow import RAGGraph


@pytest.fixture
def mock_rag_chain() -> AsyncMock:
    """Creates a mock RAG chain."""
    chain = AsyncMock(spec=RAGChain)
    return chain


@pytest.fixture
def mock_cache() -> AsyncMock:
    """Creates a mock SemanticCache."""
    cache = AsyncMock(spec=SemanticCache)
    return cache


@pytest.fixture
def rag_graph(mock_rag_chain: AsyncMock, mock_cache: AsyncMock) -> RAGGraph:
    """Creates a RAGGraph with mocked dependencies."""
    return RAGGraph(mock_rag_chain, mock_cache)


@pytest.mark.asyncio
async def test_cache_hit(
    rag_graph: RAGGraph, mock_cache: AsyncMock, mock_rag_chain: AsyncMock
) -> None:
    """Test that cache hit returns cached response and skips RAG."""
    # Setup
    question = "What is Muay Thai?"
    cached_response = {
        "answer": "Cached Answer",
        "sources": ["source1"],
        "similarity": 0.95,
    }
    mock_cache.get_cached_response.return_value = cached_response

    # Execute
    result = await rag_graph.ainvoke(question)

    # Verify
    assert result["answer"] == "Cached Answer"
    assert result["sources"] == ["source1"]
    assert result["cached"] is True

    # Verify cache was checked
    mock_cache.get_cached_response.assert_called_once_with(question)

    # Verify RAG chain was NOT called
    mock_rag_chain.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_cache_miss(
    rag_graph: RAGGraph, mock_cache: AsyncMock, mock_rag_chain: AsyncMock
) -> None:
    """Test that cache miss calls RAG chain and updates cache."""
    # Setup
    question = "New Question"
    mock_cache.get_cached_response.return_value = None

    rag_chain_response = {
        "content": "RAG Answer",
        "sources": ["source2"],
        "usage": {"tokens": 100},
    }
    mock_rag_chain.ainvoke.return_value = rag_chain_response

    # Execute
    result = await rag_graph.ainvoke(question)

    # Verify
    assert result["answer"] == "RAG Answer"
    assert result["sources"] == ["source2"]
    assert result["cached"] is False
    assert result["usage"] == {"tokens": 100}

    # Verify cache was checked
    mock_cache.get_cached_response.assert_called_once_with(question)

    # Verify RAG chain WAS called
    mock_rag_chain.ainvoke.assert_called_once_with(question)

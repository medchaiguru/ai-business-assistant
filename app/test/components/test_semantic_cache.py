"""Tests for the SemanticCache component."""
import time
from collections.abc import Generator
from typing import Any

import chromadb
import pytest
from chromadb.api import ClientAPI
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_container_is_ready

from app.components.semantic_cache import SemanticCache


@pytest.fixture(scope="module")
def chroma_container() -> Generator[DockerContainer, None, None]:
    """Spin up a ChromaDB container for testing."""
    container = DockerContainer("ghcr.io/chroma-core/chroma:latest")
    container.with_exposed_ports(8000)
    with container as chroma:
        # Wait for Chroma to be ready by checking logs
        #wait_for_logs(chroma, "Uvicorn running on", timeout=30)
        time.sleep(2)
        yield chroma


@wait_container_is_ready(ConnectionError)
def _get_chroma_client(host: str, port: str) -> ClientAPI:
    """Wait for Chroma container to be ready by checking heartbeat."""
    client = chromadb.HttpClient(host=host, port=int(port))
    client.heartbeat()
    return client


@pytest.fixture
def chroma_client(chroma_container: DockerContainer) -> Any:
    """Fixture to provide a ChromaDB client - function scoped for test isolation."""
    host = chroma_container.get_container_host_ip()
    port = chroma_container.get_exposed_port(8000)
    # Create client directly without using cached get_chroma_client
    client = _get_chroma_client(host=host, port=port)
    return client


@pytest.fixture
def cache(chroma_client: ClientAPI) -> Generator[SemanticCache, None, None]:
    """Fixture to provide a SemanticCache instance - function scoped for isolation."""
    #chroma_client.create_collection(name="test_query_answer_cache")
    semantic_cache_instance = SemanticCache(
        client=chroma_client,
        similarity_threshold=0.8
    )
    yield semantic_cache_instance
    # Cleanup after test
    semantic_cache_instance.vectorstore.delete_collection()


@pytest.mark.asyncio
async def test_exact_match(cache: SemanticCache) -> None:
    """Test storing and retrieving an exact cached response."""
    question = "What is Muay Thai?"
    answer = "Muay Thai is a combat sport..."
    sources = ["www.example.com"]

    await cache.set_cached_response(question, answer, sources)
    result = await cache.get_cached_response(question)

    assert result is not None
    assert result["answer"] == answer


@pytest.mark.asyncio
async def test_semantic_match(cache: SemanticCache) -> None:
    """Test semantic matching for similar questions."""
    # Ensure data is in cache (or use a fresh cache with setup)
    question = "Who is the craziest team member"
    answer = "Muay Thai is a combat sport..."
    sources = ["www.example.com"]
    await cache.set_cached_response(question, answer, sources)

    similar_question = "who is the most crazy member of the team"
    result = await cache.get_cached_response(similar_question)

    assert result is not None
    # We expect high similarity, e.g., > 0.8
    assert result["similarity"] > 0.8


@pytest.mark.asyncio
async def test_cache_miss(cache: SemanticCache) -> None:
    """Test cache miss for unrelated question."""
    different_question = "What is the capital of France?"
    result = await cache.get_cached_response(different_question)

    assert result is None

"""Tests for the SemanticCache component."""

from collections.abc import Generator

import pytest

from app.components.semantic_cache import SemanticCache


@pytest.fixture
def cache() -> Generator[SemanticCache, None, None]:
    """Fixture to provide a SemanticCache instance."""
    semantic_cache_instance = SemanticCache(
        collection_name="test_collection", similarity_threshold=0.8
    )
    yield semantic_cache_instance
    semantic_cache_instance.vectorstore.delete_collection()


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

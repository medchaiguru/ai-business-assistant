"""Semantic cache using ChromaDB."""

import json
import time
from typing import Any

from chromadb.api import ClientAPI
from langchain_core.documents import Document

from app.components.embedding import embeddings_model
from app.components.vector_store import (
    load_vector_store_from_path,
    load_vector_store_from_remote,
)
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """Semantic caching using ChromaDB for similar question matching."""
    def __init__(
        self,
        client: ClientAPI,
        similarity_threshold: float = 0.92,
        collection_name: str = settings.QUERY_ANSWER_CACHE,
    ):
        if settings.CHROMA_SERVER_PERSISTENCE:
            self.vectorstore = load_vector_store_from_path(
                collection_name=collection_name,
                persist_path=settings.SEMANTIC_CACHE_PATH,
                embeddings=embeddings_model
            )
        else:
            self.vectorstore = load_vector_store_from_remote(
                client=client,
                embeddings=embeddings_model,
                collection_name=collection_name,
            )
        self.similarity_threshold = similarity_threshold

    async def get_cached_response(self, question: str) -> dict[str, Any] | None:
        """Get cached response for similar question."""
        try:
            # Use similarity_search_with_score to get distance
            results = await self.vectorstore.asimilarity_search_with_score(
                question,
                k=1
            )
            if results:
                doc, distance = results[0]
                # ChromaDB uses L2 distance (lower = more similar)
                # Convert to similarity score (0-1 range)
                similarity = 1 / (1 + distance)
                logger.info("Question similarity: %.2f", similarity)
                if similarity >= self.similarity_threshold:
                    logger.info("Cache hit with similarity: %.2f", similarity)
                    return {
                        "answer": doc.metadata.get("answer"),
                        "sources": json.loads(doc.metadata.get("sources", "[]")),
                        "similarity": similarity,
                    }

            logger.info("Cache miss for question: %s...", question[:50])
            return None

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error("Error querying cache: %s", e)
            return None

    async def set_cached_response(
        self,
        question: str,
        answer: str,
        sources: list[str]
    ) -> None:
        """Cache a question/answer pair."""
        try:
            doc = Document(
                page_content=question,
                metadata={
                    "answer": answer,
                    "sources": json.dumps(sources),
                    "timestamp": time.time(),
                },
            )

            await self.vectorstore.aadd_documents([doc])
            logger.info("Cached response for question: %s...", question[:50])

        except Exception as e: # pylint: disable=broad-exception-caught
            logger.error(
                "Error caching response for question: %s... : %s", question[:50], e
            )

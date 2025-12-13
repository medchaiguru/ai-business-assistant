"""Semantic cache using ChromaDB."""

import time
from typing import Optional
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.components.embedding import embeddings_model
from app.logger import get_logger


logger = get_logger(__name__)


class SemanticCache:
    """Semantic caching using ChromaDB for similar question matching."""
    def __init__(self, cache_dir: str = None, similarity_threshold: float = 0.92):
        if cache_dir is None:
            cache_dir = f"{settings.PERSIST_PATH}_cache"

        self.vectorstore = Chroma(
            persist_directory=cache_dir,
            embedding_function=embeddings_model,
            collection_name="semantic_cache"
        )
        self.similarity_threshold = similarity_threshold


    def get_cached_response(self, question: str) -> Optional[dict]:
        """Get cached response for similar question."""
        try:
            # Use similarity_search_with_score to get distance
            results = self.vectorstore.similarity_search_with_score(
                question,
                k=1
            )
            if results:
                doc, distance = results[0]
                # ChromaDB uses L2 distance (lower = more similar)
                # Convert to similarity score (0-1 range)
                similarity = 1 / (1 + distance)
                if similarity >= self.similarity_threshold:
                    logger.info("Cache hit with similarity: %.2f", similarity)
                    return {
                        "answer": doc.metadata.get("answer"),
                        "sources": json.loads(doc.metadata.get("sources", "[]")),
                        "similarity": similarity
                    }

            logger.info("Cache miss for question: %s...", question[:50])
            return None

        except Exception as e:
            logger.error("Error querying cache: %s", e)
            return None

    def set_cached_response(
        self,
        question: str,
        answer: str,
        sources: list
    ) -> None:
        """Cache a question/answer pair."""
        try:
            doc = Document(
                page_content=question,
                metadata={
                    "answer": answer,
                    "sources": json.dumps(sources),
                    "timestamp": time.time()
                }
            )

            self.vectorstore.add_documents([doc])
            logger.info("Cached response for question: %s...", question[:50])

        except Exception as e:
            logger.error("Error caching response for question: %s... : %s", question[:50], e)

"""Semantic cache using ChromaDB."""

from typing import Optional
from langchain_chroma import Chroma
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
                    logger.info(f"Cache hit with similarity: {similarity:.2f}")
                    return {
                        "answer": doc.metadata.get("answer"),
                        "sources": eval(doc.metadata.get("sources", "[]")),
                        "similarity": similarity
                    }
            
            logger.info("Cache miss")
            return None
            
        except Exception as e:
            logger.error(f"Error querying cache: {e}")
            return None
    
    def set_cached_response(
        self,
        question: str,
        answer: str,
        sources: list
    ) -> None:
        """Cache a question/answer pair."""
        try:
            import time
            from langchain_core.documents import Document
            
            doc = Document(
                page_content=question,
                metadata={
                    "answer": answer,
                    "sources": str(sources),
                    "timestamp": time.time()
                }
            )
            
            self.vectorstore.add_documents([doc])
            logger.info(f"Cached response for question: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
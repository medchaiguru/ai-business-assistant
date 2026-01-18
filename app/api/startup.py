from fastapi import FastAPI

from app.components.embedding import embeddings_model
from app.components.llm import llm_model
from app.components.metrics import MetricsManager
from app.components.prompt import prompt
from app.components.rag_chain import RAGChain
from app.components.semantic_cache import SemanticCache
from app.components.vector_store import (
    get_chroma_client,
    get_retriever_from_vectorstore,
    load_vector_store_from_remote,
)
from app.components.workflow import RAGGraph
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def load_chroma_client(app: FastAPI) -> None:
    """Initialize and attach ChromaDB client to app state."""
    chroma_client = get_chroma_client()
    app.state.chroma_client = chroma_client


def load_metrics_manager(app: FastAPI) -> None:
    """Initialize and attach MetricsManager to app state."""
    app.state.metrics = MetricsManager(
        host=settings.REDIS_HOST, port=settings.REDIS_PORT
    )


def load_retriever(app: FastAPI) -> None:
    """Load vector store from remote ChromaDB server and
    attach retriever to app state.
    """
    try:

        vector_store = load_vector_store_from_remote(
            client=app.state.chroma_client,
            collection_name=settings.BUSINESS_DATA,
            embeddings=embeddings_model
        )
        retriever = get_retriever_from_vectorstore(vector_store, top_k=1)

    except ValueError:

        logger.warning(
            "Collection %s not found. Starting with empty store.",
            settings.BUSINESS_DATA
        )
        retriever = None

    app.state.retriever = retriever


def load_rag_graph(app: FastAPI) -> None:
    """Initialize and attach RAGGraph to app state."""
    rag_instance = RAGChain(
        retriever=app.state.retriever,
        prompt=prompt,
        llm=llm_model
    )

    semantic_cache = SemanticCache(
        client=app.state.chroma_client,
        similarity_threshold=0.8,
        collection_name=settings.QUERY_ANSWER_CACHE
    )

    app.state.rag_graph_instance = RAGGraph(rag_instance, semantic_cache)
    logger.info("RAG pipeline is ready")

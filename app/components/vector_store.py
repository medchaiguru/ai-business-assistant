from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_chroma_client() -> ClientAPI:
    """
    Get or create a cached ChromaDB client.
    Uses lru_cache to ensure we only create one connection instance.
    """
    try:
        client = chromadb.HttpClient(
            host=settings.CHROMA_SERVER_HOST,
            port=settings.CHROMA_SERVER_HTTP_PORT,
        )
        # Quick health check to fail fast if server is down
        client.heartbeat()
        logger.debug(
            "Connected to ChromaDB at %s:%d",
            settings.CHROMA_SERVER_HOST,
            settings.CHROMA_SERVER_HTTP_PORT,
        )
        return client
    except Exception as e:
        logger.error("Failed to connect to ChromaDB: %s", e)
        raise

def create_chroma_index(
    docs: list[Document],
    embeddings: Embeddings,
    collection_name: str = "langchain",
    persist_path: str = "", # DEV Mode
) -> Chroma:
    """
    Create a Chroma vector store from Documents and save locally.
    """
    if persist_path == "":
        # Create a remote client
        client = get_chroma_client()
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            collection_name=collection_name,
        )
        logger.info(
            "Chroma index created on remote server at %s:%d",
            settings.CHROMA_SERVER_HOST,
            settings.CHROMA_SERVER_HTTP_PORT,
        )
        return vector_store

    Path(persist_path).mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name
    )
    logger.info("Chroma index created at %s", persist_path)
    return vector_store


def load_vector_store(
    embeddings: Embeddings,
    collection_name: str = "langchain",
    persist_path: str = "", # DEV Mode
) -> Chroma:
    """
    Load a previously saved Chroma vector store.
    """
    if persist_path == "":
        # Create a remote client
        client = get_chroma_client()
        # Connect to remote ChromaDB server
        vector_store = Chroma(
            client=client,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        logger.info(
            "Connected to remote ChromaDB server at %s:%d",
            settings.CHROMA_SERVER_HOST,
            settings.CHROMA_SERVER_HTTP_PORT,
        )
        return vector_store

    vector_store = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    logger.info("Chroma index loaded from %s", persist_path)
    return vector_store


def get_retriever_from_vectorstore(
    vector_store: Chroma, top_k: int = 3
) -> VectorStoreRetriever:
    """
    Get a retriever from a Chroma vector store.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    logger.info("Retriever created with top_k=%d", top_k)
    return retriever

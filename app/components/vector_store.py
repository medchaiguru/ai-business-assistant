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


#@lru_cache
def get_chroma_client(
    chroma_host: str = settings.CHROMA_SERVER_HOST,
    chroma_port: int = settings.CHROMA_SERVER_HTTP_PORT
) -> ClientAPI:
    """
    Get or create a cached ChromaDB client.
    Uses lru_cache to ensure we only create one connection instance.
    """
    try:
        client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
        )
        # Quick health check to fail fast if server is down
        client.heartbeat()
        logger.debug(
            "Connected to ChromaDB at %s:%d",
            chroma_host,
            chroma_port,
        )
        # Quick health check to fail fast if server is down
        client.heartbeat()
        logger.debug(
            "Connected to ChromaDB at %s:%d",
            chroma_host,
            chroma_port,
        )
        return client
    except Exception as e:
        logger.error("Failed to connect to ChromaDB: %s", e)
        raise


def create_local_chroma_index(
    collection_name: str,
    docs: list[Document],
    persist_path: str,
    embeddings: Embeddings
) -> Chroma:
    """
    Create a local path for storing ChromaDB vector store.
    """
    if settings.CHROMA_SERVER_PERSISTENCE and persist_path:
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(persist_path),
            collection_name=collection_name
        )
        logger.info("Chroma index created at %s", persist_path)
        return vector_store
    raise ValueError("Local persistence is disabled or persist_path is empty.")


def create_remote_chroma_index(
    client: ClientAPI,
    collection_name: str,
    docs: list[Document],
    embeddings: Embeddings,
) -> Chroma:
    """
    Create a Chroma vector store from Documents and save locally.
    """
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


def load_vector_store_from_path(
    collection_name: str,
    persist_path: str,
    embeddings: Embeddings,
) -> Chroma:
    """
    Load a previously saved Chroma vector store from local path.
    """
    if settings.CHROMA_SERVER_PERSISTENCE and persist_path:
        vector_store = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings,
            collection_name=collection_name,
            create_collection_if_not_exists=False
        )
        logger.info("Chroma index loaded from %s", persist_path)
        return vector_store
    raise ValueError("Local persistence is disabled or persist_path is empty.")


def load_vector_store_from_remote(
    client: ClientAPI,
    collection_name: str,
    embeddings: Embeddings,
) -> Chroma:
    """
    Load a previously saved Chroma vector store.
    """
    # Connect to remote ChromaDB server
    vector_store = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=collection_name,
        create_collection_if_not_exists=True
    )
    logger.info(
        "Connected to remote ChromaDB server at %s:%d",
        settings.CHROMA_SERVER_HOST,
        settings.CHROMA_SERVER_HTTP_PORT,
    )
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

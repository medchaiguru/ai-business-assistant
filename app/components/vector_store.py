from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from app.logger import get_logger

logger = get_logger(__name__)


def create_chroma_index(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    persist_path: str
) -> Chroma:
    """
    Create a Chroma vector store from Documents and save locally.
    """
    Path(persist_path).mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=str(persist_path)
    )
    logger.info("Chroma index created at %s", persist_path)
    return vector_store


def load_vector_store(
    persist_path: str,
    embeddings: OpenAIEmbeddings,
    collection_name: str = "langchain"
) -> Chroma:
    """
    Load a previously saved Chroma vector store.
    """
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

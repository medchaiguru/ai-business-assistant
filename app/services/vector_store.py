from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.logger import logger


def create_chroma_index(
    docs: List[Document],
    embeddings,
    persist_path: str
    ) -> Chroma:
    """
    Create a Chroma vector store from Documents and save locally.
    """
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_path)
    )
    logger.info("Chroma index created at %s", persist_path)
    return vector_store


def load_chroma_index(persist_path: str, embeddings) -> Chroma:
    """
    Load a previously saved Chroma vector store.
    """
    vector_store = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings
    )
    logger.info("Chroma index loaded from %s", persist_path)
    return vector_store

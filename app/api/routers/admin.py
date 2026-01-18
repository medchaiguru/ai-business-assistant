from fastapi import APIRouter, BackgroundTasks, FastAPI, Request

from app.components.data_source import json_data_to_langchain_docs
from app.components.embedding import embeddings_model
from app.components.splitter import text_splitter
from app.components.vector_store import (
    create_remote_chroma_index,
    get_retriever_from_vectorstore,
)
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

def run_indexing_task(app: FastAPI) -> None:
    """Background task to re-index data and update app state."""
    logger.info("Starting background indexing task...")
    chroma_client = app.state.chroma_client
    try:
        # 1. Load and process data
        docs = json_data_to_langchain_docs(settings.DATA_PATH, settings.URL_PAGE_MAP)
        chunked_docs = text_splitter.split_documents(docs)
        logger.info("Split data into %d chunks", len(chunked_docs))

        # 2. Create/Update Index
        vector_store = create_remote_chroma_index(
            client=chroma_client,
            docs=chunked_docs,
            embeddings=embeddings_model,
            collection_name=settings.BUSINESS_DATA
        )

        # 3. Reload retriever
        retriever = get_retriever_from_vectorstore(vector_store, top_k=1)
        rag_graph_instance = app.state.rag_graph_instance

        # 4. Update retriever
        rag_graph_instance.rag_chain.retriever = retriever
        app.state.retriever = retriever

        logger.info("Indexing complete and application state updated.")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Indexing failed: %s", e)

@router.post("/index")
async def trigger_indexing(
    request: Request,
    background_tasks: BackgroundTasks
) -> dict[str, str]:
    """Trigger a background task to re-index the data."""
    background_tasks.add_task(run_indexing_task, request.app)
    return {"status": "accepted", "message": "Indexing started in background"}

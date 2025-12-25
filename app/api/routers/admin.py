from fastapi import APIRouter, BackgroundTasks, FastAPI, Request

from app.components.data_source import json_data_to_langchain_docs
from app.components.embedding import embeddings_model
from app.components.llm import llm_model
from app.components.prompt import prompt
from app.components.rag_chain import RAGChain
from app.components.semantic_cache import SemanticCache
from app.components.splitter import text_splitter
from app.components.vector_store import (
    create_chroma_index,
    get_retriever_from_vectorstore,
)
from app.components.workflow import RAGGraph
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

def run_indexing_task(app: FastAPI) -> None:
    """Background task to re-index data and update app state."""
    logger.info("Starting background indexing task...")
    try:
        # 1. Load and process data
        docs = json_data_to_langchain_docs(settings.DATA_PATH, settings.URL_PAGE_MAP)
        chunked_docs = text_splitter.split_documents(docs)
        logger.info("Split data into %d chunks", len(chunked_docs))

        # 2. Create/Update Index
        vector_store = create_chroma_index(
            chunked_docs,
            embeddings_model,
            persist_path=""
        )

        # 3. Reload components
        retriever = get_retriever_from_vectorstore(vector_store, top_k=1)
        rag_instance = RAGChain(retriever, prompt, llm_model)
        rag_graph_instance = RAGGraph(rag_instance, SemanticCache())

        # 4. Update app state
        app.state.rag_graph_instance = rag_graph_instance

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

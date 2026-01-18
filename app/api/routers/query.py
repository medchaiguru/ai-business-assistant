from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_rag_graph
from app.components.models import QueryRequest, QueryResponse
from app.components.workflow import RAGGraph
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.post("", response_model=QueryResponse)
async def query_endpoint(
    query_request: QueryRequest,
    request: Request,
    rag_graph_instance: RAGGraph = Depends(get_rag_graph), # noqa: B008
) -> QueryResponse | JSONResponse:

    """Handle query requests using RAG pipeline."""

    logger.info("Received query: %s", query_request.question)

    try:
        result = await rag_graph_instance.ainvoke(query_request.question)
        if "usage" in result:
            # Store usage metrics in request state for middleware
            request.state.usage_metrics = result["usage"]

        return QueryResponse(answer=result["answer"], sources=result["sources"])

    except Exception as e: # pylint: disable=broad-exception-caught

        logger.exception("Error handling query")
        return JSONResponse(status_code=500, content={"error": str(e)})

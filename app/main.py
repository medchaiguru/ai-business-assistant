"""Main application file for FastAPI app."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse

from app.config import settings
from app.logger import get_logger
from app.components.retriever import retriever
from app.components.prompt import prompt
from app.components.llm import llm_model
from app.components.rag_chain import RAGChainWithSources
from app.workflow import RAGGraph
from app.semantic_cache import SemanticCache
from app.metrics import MetricsManager
from app.models import (
    QueryRequest,
    QueryResponse,
    MetricsResponse
)


logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup code
    logger.info("Application starting up...")
    # Initialize MetricsManager
    metrics_manager = MetricsManager(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT
    )
    app.state.metrics = metrics_manager
    rag_instance = RAGChainWithSources(retriever, prompt, llm_model)
    rag_graph_instance = RAGGraph(rag_instance, SemanticCache())
    app.state.rag_graph_instance = rag_graph_instance
    logger.info("RAG pipeline is ready")
    yield
    # Shutdown code
    await app.state.metrics.close()
    logger.info("Application shutting down...")


app = FastAPI(lifespan=lifespan)

def get_rag_graph(request: Request) -> RAGGraph:
    """Return RAG chain instance"""
    return request.app.state.rag_graph_instance


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their response times."""
    start_time = time.time()
    logger.info("%s %s", request.method, request.url)

    response = await call_next(request)

    response_time = request.state.response_time
    response.headers["X-Response-Time"] = f"{response_time:.3f}s"
    logger.info("%d for %s", response.status_code, request.url.path)
    logger.info("Completed in %.3f seconds", response_time)

    request.state.start_time = start_time
    return response


@app.middleware("http")
async def cache_metrics(request: Request, call_next):
    """Cache metrics to Redis after each request to /query endpoint."""
    response = await call_next(request)

    start_time = request.state.start_time
    response_time = time.time() - start_time
    # Store response time in request state
    request.state.response_time = response_time

    # Only track metrics for /query endpoint
    if request.url.path == "/query" and hasattr(request.state, "usage_metrics"):
        metrics_manager = request.app.state.metrics
        usage = request.state.usage_metrics

        await metrics_manager.update(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            duration=response_time,
        )
    return response


@app.get("/health")
async def health_check():
    '''Return Health Status Message'''
    logger.info("Health check endpoint called")
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    query_request: QueryRequest,
    request: Request,
    rag_graph_instance: RAGGraph = Depends(get_rag_graph)
) -> QueryRequest | JSONResponse:
    """Handle query requests using RAG pipeline."""
    logger.info("Received query: %s", query_request.question)
    try:
        result = await rag_graph_instance.ainvoke(query_request.question)
        if "usage" in result:
            # Store usage metrics in request state for middleware
            request.state.usage_metrics = result["usage"]

        return QueryResponse(answer=result["content"], sources=result["sources"])

    except Exception as e:
        logger.exception("Error handling query")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/metrics")
async def get_metrics(request: Request) -> MetricsResponse:
    """Get current metrics from Redis"""
    metrics_manager: MetricsManager = request.app.state.metrics
    return await metrics_manager.get()

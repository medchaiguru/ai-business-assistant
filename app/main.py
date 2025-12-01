'''Main'''
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
import time

from app.models import QueryRequest, QueryResponse
from app.config import settings
from app.logger import get_logger
from app.components.retriever import retriever
from app.components.prompt import prompt
from app.components.llm import llm_model
from app.components.rag_chain import RAGChainWithSources
from app.metrics import MetricsManager


logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """LifeSpan"""
    # Startup code
    logger.info("Application starting up...")
    # Initialize MetricsManager
    metrics_manager = MetricsManager(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT
    )
    app.state.metrics = metrics_manager
    rag_instance = RAGChainWithSources(retriever, prompt, llm_model)
    app.state.rag_instance = rag_instance
    logger.info("RAG pipeline is ready")
    yield
    # Shutdown code
    await app.state.metrics.close()
    logger.info("Application shutting down...")


app = FastAPI(lifespan=lifespan)

def get_rag_chain(request: Request) -> RAGChainWithSources:
    """Return RAG chain instance"""
    return request.app.state.rag_instance


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests"""
    start_time = time.time()
    logger.info("%s %s", request.method, request.url)

    response = await call_next(request)

    response_time = time.time() - start_time

    # Store response time in request state
    request.state.response_time = response_time

    response.headers["X-Response-Time"] = f"{response_time:.3f}s"
    logger.info("%d for %s", response.status_code, request.url.path)
    logger.info("Completed in %.3f seconds", response_time)
    return response


@app.middleware("http")
async def cache_metrics(request: Request, call_next):
    """Cache metrics to Redis after each request"""
    response = await call_next(request)

    # Only track metrics for /query endpoint
    if request.url.path == "/query" and hasattr(request.state, "usage_metrics"):
        metrics_manager = request.app.state.metrics
        usage = request.state.usage_metrics
        response_time = getattr(request.state, "response_time", 0.0)

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
    request: QueryRequest,
    rag_instance: RAGChainWithSources = Depends(get_rag_chain)
    ) -> JSONResponse:
    logger.info("Received query: %s", request.question)
    try:
        result = await rag_instance.ainvoke(request.question)

        # Store usage metrics in request state for middleware
        if "usage" in result:
            request.state.usage_metrics = result["usage"]

        return QueryResponse(answer=result["content"], sources=result["sources"])
    except Exception as e:
        logger.exception("Error handling query")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/metrics")
async def get_metrics(request: Request):
    """Get current metrics from Redis"""
    metrics_manager = request.app.state.metrics
    return await metrics_manager.get()
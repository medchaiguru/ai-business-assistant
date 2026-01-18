"""Main application file for FastAPI app."""

import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import admin, health, metrics, query
from app.api.startup import (
    load_chroma_client,
    load_metrics_manager,
    load_rag_graph,
    load_retriever,
)
from app.logger import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Application starting up...")

    # load chroma client to app state
    load_chroma_client(app)

    # initialize MetricsManager into app state
    load_metrics_manager(app)

    # load vector store retriever to app state
    load_retriever(app)

    # load rag graph instance to app state
    load_rag_graph(app)

    yield
    # Shutdown code
    await app.state.metrics.close()
    logger.info("Application shutting down...")


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def cache_metrics(request: Request,
    call_next: Callable[[Request],
    Awaitable[JSONResponse]]
) -> JSONResponse:
    """Cache metrics to Redis after each request to /query endpoint."""
    response = await call_next(request)
    start_time = request.state.start_time
    response_time = time.time() - start_time

    # Only track metrics for /query endpoint
    if request.url.path == "/query" and hasattr(request.state, "usage_metrics"):
        metrics_manager = request.app.state.metrics
        usage = request.state.usage_metrics

        await metrics_manager.update(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            duration=response_time,
        )
    # Store response time in request state
    request.state.response_time = response_time
    return response

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[JSONResponse]]
) -> JSONResponse:
    """Log incoming requests and their response times."""
    start_time = time.time()
    request.state.start_time = start_time

    logger.info("%s %s", request.method, request.url)

    response = await call_next(request)
    response_time = request.state.response_time

    logger.info("%d for %s", response.status_code, request.url.path)
    logger.info("Completed in %.3f seconds", response_time)

    # Add X-Response-Time header for /query endpoint
    if request.url.path == "/query" and hasattr(request.state, "response_time"):
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"

    return response

# === Register Routers ===
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

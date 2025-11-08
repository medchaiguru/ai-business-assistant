'''Main'''

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.models import QueryRequest, QueryResponse
from app.config import settings
from app.logger import get_logger
from app.components.retriever import retriever
from app.components.prompt import prompt
from app.components.llm import llm_model
from app.components.rag_chain import RAGChainWithSources



logger = get_logger(__name__)
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
)


def get_rag_chain(request: Request) -> RAGChainWithSources:
    return request.app.state.rag_instance


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡️ {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"⬅️ {response.status_code} for {request.url.path}")
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Application starting up...")
    rag_instance = RAGChainWithSources(retriever, prompt, llm_model)
    app.state.rag_instance = rag_instance
    logger.info("RAG pipeline is ready")
    yield
    # Shutdown code
    logger.info("Application shutting down...")


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
    logger.info(f"Received query: {request.question}")
    try:
        result = await rag_instance.ainvoke(request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.exception("Error handling query")
        return JSONResponse(status_code=500, content={"error": str(e)})

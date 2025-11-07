'''Main'''

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logger import logger

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Application starting up...")
    yield
    # Shutdown code
    logger.info("Application shutting down...")


@app.get("/health")
async def health_check():
    '''Return Health Status Message'''
    logger.info("Health check endpoint called")
    return {
        "status": "ok",
        "app": settings.app_name,
        "debug": settings.debug,
        "key": settings.OPENAI_API_KEY
    }

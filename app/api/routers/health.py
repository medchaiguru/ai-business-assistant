from fastapi import APIRouter

from app.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("")
async def health_check() -> dict[str, str]:
    """Return Health Status Message"""
    logger.info("Health check endpoint called")
    return {"status": "ok"}

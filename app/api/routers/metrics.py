from fastapi import APIRouter, Request

from app.components.metrics import MetricsManager
from app.components.models import MetricsResponse

router = APIRouter()

@router.get("", response_model=MetricsResponse)
async def get_metrics(request: Request) -> MetricsResponse:
    """Get current metrics from Redis"""
    metrics_manager: MetricsManager = request.app.state.metrics
    return await metrics_manager.get()

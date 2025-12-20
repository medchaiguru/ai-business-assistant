"""Metrics management using Redis."""

import redis.asyncio as redis

from app.components.llm import llm_cost
from app.components.models import MetricsResponse


class MetricsManager:
    """Class to manage and store metrics in Redis."""

    def __init__(self,
        host: str="localhost",
        port: int=6379,
        db: int=0
    ):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)


    async def update(
        self,
        input_tokens: int,
        output_tokens: int,
        response_time: float,
    ) -> None:
        """Update metrics in Redis"""
        pipe = self.r.pipeline()
        pipe.incrby("tokens_total", input_tokens + output_tokens)
        cost = llm_cost(input_tokens, output_tokens)
        pipe.incrbyfloat("cost_total", cost)
        pipe.incr("requests_total")

        # Store response time stats
        pipe.incrbyfloat("time_total", response_time)
        pipe.lpush("response_times", response_time)
        pipe.ltrim("response_times", 0, 99)  # Keep last 100 values
        await pipe.execute()

    async def get(self) -> MetricsResponse:
        """Retrieve current metrics from Redis"""
        tokens = int(await self.r.get("tokens_total") or 0)
        total_time = float(await self.r.get("time_total") or 0.0)
        cost = float(await self.r.get("cost_total") or 0.0)
        requests = int(await self.r.get("requests_total") or 0)

        avg_time = total_time / requests if requests else 0.0

        return MetricsResponse(
            tokens_total=tokens,
            requests_total=requests,
            average_response_time=avg_time,
            cost_total=cost,
        )

    async def close(self) -> None:
        """Close Redis connection"""
        await self.r.aclose()

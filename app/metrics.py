import redis
from app.components.llm import llm_cost


class MetricsManager:
    def __init__(self, host="localhost", port=6379, db=0):
        self.r = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def update(
            self, 
            input_tokens: int,
            output_tokens: int,
            duration: float,
            cost: float = None
        ) -> None:
        pipe = self.r.pipeline()
        pipe.incrby("tokens_total", input_tokens + output_tokens)
        pipe.incrbyfloat("time_total", duration)
        if cost is None:
            cost = llm_cost(input_tokens, output_tokens)
        pipe.incrbyfloat("budget_total", cost)
        pipe.incr("requests_total")
        pipe.execute()


    def get(self) -> dict:
        tokens = int(self.r.get("tokens_total") or 0)
        total_time = float(self.r.get("time_total") or 0.0)
        budget = float(self.r.get("budget_total") or 0.0)
        requests = int(self.r.get("requests_total") or 0)

        avg_time = total_time / requests if requests else 0.0

        return {
            "tokens_total": tokens,
            "requests_total": requests,
            "average_response_time": avg_time,
            "budget_total": budget,
        }
    
    async def close(self):
        """Close Redis connection"""
        await self.r.close()

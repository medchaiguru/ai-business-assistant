"""Model API request and response schemas."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Model API request schema."""

    question: str


class QueryResponse(BaseModel):
    """Model API response schema."""

    answer: str
    sources: list[str]


class MetricsResponse(BaseModel):
    """Model API metrics response schema."""

    tokens_total: int
    requests_total: int
    average_response_time: float
    cost_total: float

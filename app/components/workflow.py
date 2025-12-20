"""RAG workflow with caching using LangGraph."""

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.components.rag_chain import RAGChain
from app.components.semantic_cache import SemanticCache
from app.logger import get_logger

logger = get_logger(__name__)


class RAGState(TypedDict, total=False):
    """State for RAG workflow."""
    question: str
    answer: str
    sources: list[str]
    cached: bool
    usage: dict[str, Any] | None


class RAGGraph:
    """RAG workflow with semantic caching."""
    def __init__(self, rag_chain: RAGChain, cache: SemanticCache):
        self.rag_chain = rag_chain
        self.cache = cache
        self.graph = self._build_graph()

    async def _check_cache(self, state: RAGState) -> RAGState:
        """Step 1: Check semantic cache."""
        logger.info(
            "Step 1: Checking cache for question: %s...", state["question"][:50]
        )

        cached_result = await self.cache.get_cached_response(state["question"])
        if cached_result:
            logger.info("Cache hit! Similarity: %.2f", cached_result["similarity"])
            return RAGState(
                answer=cached_result["answer"],
                sources=cached_result["sources"],
                cached=True,
                usage=None,
            )

        logger.info("Cache miss - proceeding to RAG")
        return RAGState(cached=False)

    async def _call_rag(self, state: RAGState) -> RAGState:
        """Step 2: Call RAG chain if cache missed."""
        if state["cached"]:
            logger.info("Skipping RAG - using cached response")
            return state

        logger.info("Step 2: Calling RAG chain")
        result = await self.rag_chain.ainvoke(state["question"])
        logger.info("RAG response received for question: %s...", state["question"][:50])

        # Cache the new response
        await self.cache.set_cached_response(
            question=state["question"],
            answer=result["content"],
            sources=result["sources"],
        )

        return RAGState(
            answer=result["content"],
            sources=result["sources"],
            usage=result.get("usage"),
        )

    def _build_graph(self) -> Any:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("check_cache", self._check_cache)
        workflow.add_node("call_rag", self._call_rag)

        # Define edges
        workflow.set_entry_point("check_cache")
        workflow.add_edge("check_cache", "call_rag")
        workflow.add_edge("call_rag", END)

        return workflow.compile()

    async def ainvoke(self, question: str) -> RAGState:
        """Execute the RAG workflow."""
        initial_state= RAGState(
            question=question,
            answer="",
            sources=[],
            cached=False,
            usage=None,
        )

        result = await self.graph.ainvoke(initial_state)

        return RAGState(
            answer=result["answer"],
            sources=result["sources"],
            cached=result["cached"],
            usage=result["usage"],
        )

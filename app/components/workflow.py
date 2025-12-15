"""RAG workflow with caching using LangGraph."""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from app.logger import get_logger
from app.semantic_cache import SemanticCache

logger = get_logger(__name__)


class RAGState(TypedDict):
    """State for RAG workflow."""
    question: str
    answer: str | None
    sources: list
    cached: bool
    usage: dict | None


class RAGGraph:
    """RAG workflow with semantic caching."""
    def __init__(self, rag_chain, cache: SemanticCache):
        self.rag_chain = rag_chain
        self.cache = cache
        self.graph = self._build_graph()

    def _check_cache(self, state: RAGState) -> dict:
        """Step 1: Check semantic cache."""
        logger.info("Step 1: Checking cache for question: %s...", state["question"][:50])

        cached_result = self.cache.get_cached_response(state["question"])
        if cached_result:
            logger.info("Cache hit! Similarity: %.2f", cached_result['similarity'])
            return {
                "answer": cached_result["answer"],
                "sources": cached_result["sources"],
                "cached": True,
                "usage": None
            }

        logger.info("Cache miss - proceeding to RAG")
        return {"cached": False}

    async def _call_rag(self, state: RAGState) -> dict:
        """Step 2: Call RAG chain if cache missed."""
        if state["cached"]:
            logger.info("Skipping RAG - using cached response")
            return state

        logger.info("Step 2: Calling RAG chain")
        result = await self.rag_chain.ainvoke(state["question"])
        logger.info("RAG response received for question: %s...", state["question"][:50])
        # Cache the new response
        self.cache.set_cached_response(
            question=state["question"],
            answer=result["content"],
            sources=result["sources"]
        )

        return {
            "answer": result["content"],
            "sources": result["sources"],
            "usage": result.get("usage")
        }

    def _build_graph(self) -> StateGraph:
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

    async def ainvoke(self, question: str) -> dict:
        """Execute the RAG workflow."""
        initial_state = {
            "question": question,
            "answer": None,
            "sources": [],
            "cached": False,
            "usage": None
        }

        result = await self.graph.ainvoke(initial_state)

        return {
            "content": result["answer"],
            "sources": result["sources"],
            "cached": result["cached"],
            "usage": result["usage"]
        }

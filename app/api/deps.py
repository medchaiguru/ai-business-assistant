from typing import cast

from fastapi import Request

from app.components.workflow import RAGGraph


def get_rag_graph(request: Request) -> RAGGraph:
    """Return RAG chain instance from app state."""
    return cast(RAGGraph, request.app.state.rag_graph_instance)

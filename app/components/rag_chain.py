"""
Retrieval-Augmented Generation (RAG) chain component."""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI


class RAGChain(Runnable[str, dict[str, Any]]):
    """
    Retrieval-Augmented Generation (RAG) chain that retrieves
    documents and generates answers with sources.
    """

    def __init__(self,
        retriever:VectorStoreRetriever | None,
        prompt:ChatPromptTemplate,
        llm:ChatOpenAI
    ):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def invoke(
        self,
        input: str, # pylint: disable=redefined-builtin
        config: RunnableConfig | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Sync invoke - required by Runnable abstract base class."""
        raise NotImplementedError("Use ainvoke() for async execution")

    async def ainvoke(
        self,
        input: str, # pylint: disable=redefined-builtin
        config: RunnableConfig | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Override Async invoke method of Runnable"""

        # Allow overriding retriever at runtime
        retriever = kwargs.get("retriever", self.retriever)

        # Retrieve context + sources
        docs = await retriever.ainvoke(input, config=config)
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source_url", "") for d in docs]

        # Format prompt
        prompt_text = self.prompt.format(context=context_text, question=input)

        # Get LLM answer
        model_answer = await self.llm.ainvoke(prompt_text)

        usage_metadata = model_answer.usage_metadata
        usage = dict(usage_metadata) if usage_metadata else {}
        content = model_answer.content or ""
        return {
            "content": content,
            "sources": sources,
            "usage": usage
        }

from langchain_core.runnables.base import Runnable


class RAGChainWithSources(Runnable):
    def __init__(self, retriever, prompt, llm):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def invoke(self, input, config=None, **kwargs) -> dict:  # pylint: disable=abstract-method
        """Override Async invoke method of Runnable"""
        # Retrieve context + sources
        docs = self.retriever._get_relevant_documents(input, run_manager=None)
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source_url", "") for d in docs]

        # Format prompt
        prompt_text = self.prompt.format(context=context_text, question=input)

        # Get LLM answer
        answer = self.llm.invoke(prompt_text)

        # Return both answer and sources
        return {"ai_answer": answer, "sources": sources}
    

    async def ainvoke(self, input, config=None, **kwargs) -> dict:  # pylint: disable=abstract-method
        """Override Async invoke method of Runnable"""
        # Retrieve context + sources
        docs = await self.retriever._aget_relevant_documents(input, run_manager=None)
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source_url", "") for d in docs]

        # Format prompt
        prompt_text = self.prompt.format(context=context_text, question=input)

        # Get LLM answer
        model_answer = await self.llm.ainvoke(prompt_text)

        usage: dict = model_answer.get("usage_metadata", {})
        content : str = model_answer.get("content", "")
        return {
            "content": content,
            "sources": sources,
            "usage": usage
        }

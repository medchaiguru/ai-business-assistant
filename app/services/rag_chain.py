"""from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.services.llm import llm_model
from app.services.retriever import retriever_text
from app.services.prompt import prompt


# Define your chain
rag_chain = (
    {"context": retriever_text, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | StrOutputParser()
)"""

from langchain_core.runnables.base import Runnable


class RAGChainWithSources(Runnable):
    def __init__(self, retriever, prompt, llm):
        self.retriever = retriever
        self.prompt = prompt
        self.llm = llm

    def invoke(self, input, config=None, **kwargs):
        # Retrieve context + sources
        docs = self.retriever._get_relevant_documents(input, run_manager=None)
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("source_url", "") for d in docs]

        # Format prompt
        prompt_text = self.prompt.format(context=context_text, question=input)

        # Get LLM answer
        answer = self.llm.invoke(prompt_text)

        # Return both answer and sources
        return {"answer": answer, "sources": sources}
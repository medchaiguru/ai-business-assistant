from langchain_core.messages.ai import AIMessage

from app.components.rag_chain import RAGChainWithSources
from app.components.retriever import retriever
from app.components.prompt import prompt
from app.components.llm import llm_model

rag_chain = RAGChainWithSources(retriever, prompt, llm_model)
query = "Whos is the craziest team member ?"
print("User query:", query)


answer = rag_chain.invoke(query)
ai_answer: AIMessage = answer["ai_answer"]
sources = answer["sources"]
print("\nAI Answer:\n", ai_answer.content)

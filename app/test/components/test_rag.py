"""Test RAG Chain Functionality"""

from langchain_core.messages.ai import AIMessage

from app.components.embedding import embeddings_model
from app.components.llm import llm_model
from app.components.prompt import prompt
from app.components.rag_chain import RAGChain
from app.components.vector_store import (
    get_retriever_from_vectorstore,
    load_vector_store,
)
from app.config import settings

vector_store = load_vector_store(settings.DATA_VECTOR_PATH, embeddings_model)
retriever = get_retriever_from_vectorstore(vector_store, top_k=1)
rag_chain = RAGChain(retriever, prompt, llm_model)
query = "Whos is the craziest team member ?"
print("User query:", query)

answer = rag_chain.invoke(query)
ai_answer: AIMessage = answer["ai_answer"]
sources = answer["sources"]
print("\nAI Answer:\n", ai_answer.content)

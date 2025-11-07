from app.services.rag_chain import rag_chain

query = "Where is Sinbi Muay Thai gym located ?"
print("User query:", query)

answer = rag_chain.invoke(query)
print("\nAI Answer:\n", answer)

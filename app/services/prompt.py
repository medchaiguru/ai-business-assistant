from langchain_core.prompts import ChatPromptTemplate

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for Sinbi Muay Thai.
Answer the following question based ONLY on the provided context.

Context:
{context}

Question:
{question}
""")

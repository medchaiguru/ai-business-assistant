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

classifier_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Based on the user's question, decide which page from the list below contains the most relevant information.

Pages: {page_names}

Question: {question}

Answer with only one page name from the list.
""")

from langchain_openai import OpenAIEmbeddings

from app.config import settings

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.OPENAI_API_KEY
)

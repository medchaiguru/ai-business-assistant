from langchain_chroma import Chroma

from app.config import settings
from app.components.embedding import embeddings_model


# Load existing Chroma index
vectorstore = Chroma(
    persist_directory=settings.PERSIST_PATH,
    embedding_function=embeddings_model
)

# Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

"""Create chroma index"""

from pathlib import Path

from app.components.data_source import json_data_to_langchain_docs
from app.components.embedding import embeddings_model
from app.components.splitter import text_splitter
from app.components.vector_store import create_local_chroma_index
from app.config import settings

persist_path = Path(settings.DATA_VECTOR_PATH)
persist_path.mkdir(parents=True, exist_ok=True)

# Load documents
docs = json_data_to_langchain_docs(settings.DATA_PATH, settings.URL_PAGE_MAP)

# Split into chunks
chunked_docs = text_splitter.split_documents(docs)
print(f"Split into {len(chunked_docs)} chunks.")

# Create Chroma index
vector_store = create_local_chroma_index(
    collection_name=settings.BUSINESS_DATA,
    docs=chunked_docs,
    embeddings=embeddings_model,
    persist_path=str(persist_path)
)

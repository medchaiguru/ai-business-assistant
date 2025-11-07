'''Create chroma index'''

from pathlib import Path

from app.services.embedding import embeddings_model
from app.services.data_source import json_data_to_langchain_docs
from app.services.vector_store import create_chroma_index
from app.core.config import settings
from app.services.splitter import text_splitter


persist_path = Path(settings.PERSIST_PATH)
persist_path.mkdir(parents=True, exist_ok=True)

# Load documents
docs = json_data_to_langchain_docs(
    settings.DATA_PATH,
    settings.URL_PAGE_MAP
)

# Split into chunks
chunked_docs = text_splitter.split_documents(docs)
print(f"Split into {len(chunked_docs)} chunks.")

# Create Chroma index
vector_store = create_chroma_index(docs, embeddings_model, persist_path)

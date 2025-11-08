'''Load chroma data'''


from app.components.embedding import embeddings_model
from app.components.vector_store import load_chroma_index
from app.config import settings


vector_store = load_chroma_index(settings.PERSIST_PATH, embeddings_model)

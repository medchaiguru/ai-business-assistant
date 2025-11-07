'''Load chroma data'''


from app.services.embedding import embeddings_model
from app.services.vector_store import load_chroma_index
from app.core.config import settings


vector_store = load_chroma_index(settings.PERSIST_PATH, embeddings_model)

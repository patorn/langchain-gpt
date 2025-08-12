from pathlib import Path
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import settings, VECTOR_DIR

_embeddings_cache = None
_chroma_client = None


def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    return _embeddings_cache


def get_vectorstore(collection_name: str | None = None) -> Chroma:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = collection_name or settings.chroma_collection
    return Chroma(client=_chroma_client, collection_name=collection, embedding_function=get_embeddings())

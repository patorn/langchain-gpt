import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DIR = PROJECT_ROOT / "vector_store"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "docs")
    allow_file_types: tuple[str, ...] = (".txt", ".md", ".pdf")
    max_file_size_mb: int = 15

settings = Settings()

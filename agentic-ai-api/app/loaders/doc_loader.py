from pathlib import Path
from typing import Iterable, List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..core.config import settings


def load_paths(paths: Iterable[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        if p.suffix.lower() not in settings.allow_file_types:
            continue
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        else:
            loader = TextLoader(str(p))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)

# Agentic AI API

Production-oriented scaffold featuring:
- FastAPI backend (chat + ingestion)
- LangChain ReAct agent (plan/execute style simplified)
- Chroma persistent vector store
- PDF / text ingestion pipeline
- Simple HTML frontend
- Dockerfile for deployment

## Quick Start

```bash
# Set Google key
export GOOGLE_API_KEY=YOUR_KEY

# Install deps
pip install -r agentic-ai-api/requirements.txt

# (Optional) ingest docs
python -c "from agentic-ai-api.app.core.vectorstore import get_vectorstore; from pathlib import Path; from agentic-ai-api.app.loaders.doc_loader import load_paths; vs=get_vectorstore(); docs=load_paths([Path('README.md')]); vs.add_documents(docs)"

# Run API
uvicorn agentic-ai-api.app.main:app --reload
```

Visit http://localhost:8000/frontend/index.html (serve statically yourself or with nginx) and POST to /api/chat.

## Environment Variables
- GOOGLE_API_KEY (required)
- GEMINI_MODEL (default gemini-2.0-flash-exp)
- EMBEDDING_MODEL (default models/embedding-001)
- CHROMA_COLLECTION (default docs)

## Docker
```bash
docker build -t agentic-ai-api -f agentic-ai-api/Dockerfile .
docker run -p 8000:8000 -e GOOGLE_API_KEY=YOUR_KEY agentic-ai-api
```

## Next Steps
- Add auth / rate limiting
- Add streaming responses
- Replace mock calc with full math / external tools
- Add summarization for long histories

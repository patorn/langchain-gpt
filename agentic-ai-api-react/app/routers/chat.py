from fastapi import APIRouter, Depends
from ..schemas.chat import ChatRequest, ChatResponse
from ..agents.plan_execute import create_agent
from ..core.vectorstore import get_vectorstore
from ..loaders.doc_loader import load_paths
from pathlib import Path

router = APIRouter()
_agent_cache = None


async def get_agent():
    global _agent_cache
    if _agent_cache is None:
        # Creating agent requires an event loop (gRPC async client). Now we are in async context.
        _agent_cache = create_agent()
    return _agent_cache

import asyncio
@router.post("/chat")
async def chat(req: ChatRequest, agent=Depends(get_agent)):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: agent.invoke({"input": req.message}))
    return ChatResponse(session_id=req.session_id, answer=result["output"])

@router.post("/ingest")
async def ingest(paths: list[str]):
    docs = load_paths(Path(p) for p in paths)
    vs = get_vectorstore()
    if docs:
        vs.add_documents(docs)
    return {"ingested": len(docs)}

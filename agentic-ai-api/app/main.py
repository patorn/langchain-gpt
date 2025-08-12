from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.chat import router as chat_router

app = FastAPI(title="Agentic AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")

@app.get("/health")
async def health():
    return {"status": "ok"}

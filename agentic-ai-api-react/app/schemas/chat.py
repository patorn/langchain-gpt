from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: List[str] | None = None

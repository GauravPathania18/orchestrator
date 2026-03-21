from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5


class MemoryRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    is_helpful: bool
    feedback_text: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    data: dict

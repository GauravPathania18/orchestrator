"""
Backend RAG Orchestrator Application.

This is the main application package that coordinates between:
- API routes (FastAPI endpoints)
- Services (business logic and external service integration)
- Schemas (request/response data models)
- Core (configuration and shared utilities)
"""

from app.core.config import EMBEDDING_SERVICE_URL, VECTOR_SERVICE_URL
from app.api.chat import router as chat_router
from app.schemas.chat import ChatRequest, MemoryRequest, ChatResponse

__all__ = [
    "EMBEDDING_SERVICE_URL",
    "VECTOR_SERVICE_URL",
    "chat_router",
    "ChatRequest",
    "MemoryRequest",
    "ChatResponse",
]

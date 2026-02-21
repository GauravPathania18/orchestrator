"""
API Routes module.

This module contains all FastAPI route handlers organized by resource:
- chat: Chat and memory management endpoints
- sessions: Session history and short-term memory endpoints
"""

from .chat import router as chat_router
from .sessions import router as sessions_router

__all__ = [
    "chat_router",
    "sessions_router",
]

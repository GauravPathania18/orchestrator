"""
API Routes module.

This module contains all FastAPI route handlers organized by resource:
- simple: Chat and memory management endpoints
- sessions: Session history and short-term memory endpoints
- raptor: RAPTOR-related endpoints
"""

from .simple import router as chat_router
from .sessions import router as sessions_router

__all__ = [
    "chat_router",
    "sessions_router",
]

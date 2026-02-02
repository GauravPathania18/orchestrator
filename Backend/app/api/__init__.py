"""
API Routes module.

This module contains all FastAPI route handlers organized by resource:
- chat: Chat and memory management endpoints
"""

from .chat import router

__all__ = [
    "router",
]

"""
API Routes module.

This module contains all FastAPI route handlers organized by resource:
- vectors: Vector storage, search, and query endpoints
"""

from .vectors import router

__all__ = [
    "router",
]

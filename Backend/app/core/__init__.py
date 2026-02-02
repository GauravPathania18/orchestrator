"""
Core configuration and settings module.

This module contains application-wide configuration including:
- Environment variables and their defaults
- Service URLs for external integrations
- Application settings
"""

from .config import EMBEDDING_SERVICE_URL, VECTOR_SERVICE_URL

__all__ = [
    "EMBEDDING_SERVICE_URL",
    "VECTOR_SERVICE_URL",
]

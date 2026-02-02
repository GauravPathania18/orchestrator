"""
Vector Storage Service Application.

This is a FastAPI microservice that manages vector storage and retrieval using ChromaDB.

Main components:
- main: FastAPI application setup and router registration
- config: Configuration and environment variables
- models: Pydantic request/response data models
- routes: API endpoint handlers
- services: Core business logic (vector storage, embeddings, metadata management)
"""

from .main import app
from .config import EMBEDDER_URL, OLLAMA_URL, PERSIST_DIR, COLLECTION_NAME
from .models import TextRequest, SearchRequest, QueryRequest

__all__ = [
    "app",
    "EMBEDDER_URL",
    "OLLAMA_URL",
    "PERSIST_DIR",
    "COLLECTION_NAME",
    "TextRequest",
    "SearchRequest",
    "QueryRequest",
]

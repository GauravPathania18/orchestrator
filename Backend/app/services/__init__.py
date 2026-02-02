"""
Services module for Backend application.

This module contains business logic and external service integrations:
- embedding_client: Handles communication with the embedding service
- vector_client: Interfaces with the vector storage service
- rag_pipeline: Implements the RAG (Retrieval Augmented Generation) pipeline
"""

from .embedding_client import get_embedding
from .vector_client import query_text, add_text, add_vector, query_vector
from .rag_pipeline import _extract_top_docs

__all__ = [
    "get_embedding",
    "query_text",
    "add_text",
    "add_vector",
    "query_vector",
    "_extract_top_docs",
]

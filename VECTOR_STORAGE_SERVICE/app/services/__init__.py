"""
Services module for Vector Storage Service.

This module contains core vector storage functionality:
- vector_store: Main vector database management with ChromaDB
- embedder: Handles embedding generation for documents
- metadata: Manages and normalizes metadata for stored vectors
- utils: Utility functions for text and data processing
"""

from .vector_store import VectorStore, vector_store
from .embedder import get_embedding
from .metadata import enrich_metadata, create_metadata_from_text
from .utils import clean_user_text, normalize_metadata

__all__ = [
    "VectorStore",
    "vector_store",
    "get_embedding",
    "enrich_metadata",
    "create_metadata_from_text",
    "clean_user_text",
    "normalize_metadata",
]

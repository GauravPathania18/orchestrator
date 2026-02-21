"""
Services module for Vector Storage Service.

This module contains core vector storage functionality:
- vector_store: Main vector database management with ChromaDB
- embedder: Handles embedding generation for documents
- metadata: Manages and normalizes metadata for stored vectors
- utils: Utility functions for text and data processing
"""

from .vector_store import VectorStore, vector_store, get_vector_store
from .embedder import get_embedding
from .metadata import generate_metadata, validate_metadata, update_metadata_in_chroma
from .utils import clean_user_text, normalize_metadata

__all__ = [
    "VectorStore",
    "vector_store",
    "get_vector_store",
    "get_embedding",
    "generate_metadata",
    "validate_metadata",
    "update_metadata_in_chroma",
    "clean_user_text",
    "normalize_metadata",
]

import httpx
from app.core.config import VECTOR_SERVICE_URL
from typing import Any


async def query_text(query: str, top_k: int = 3) -> Any:
    """Call vector storage service's query_text endpoint with a text query."""
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/query_text", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def add_text(text: str, metadata: dict | None = None) -> Any:
    """Call vector storage service's add_text endpoint to store a document."""
    payload = {"text": text, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/add_text", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def add_vector(vector: list[float], metadata: dict | None = None) -> Any:
    """Send an externally computed vector to the vector service for storage."""
    payload = {"vector": vector, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/add_vector", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def query_vector(vector: list[float], top_k: int = 3) -> Any:
    """Call vector storage service's query_vector endpoint with a raw vector."""
    payload = {"vector": vector, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/query_vector", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def semantic_search(query: str, top_k: int = 5) -> Any:
    """Call vector storage service's semantic_search endpoint with similarity scores."""
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/semantic_search", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


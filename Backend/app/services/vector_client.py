import httpx
from app.core.config import VECTOR_SERVICE_URL
from typing import Any


async def query_text(query: str, top_k: int = 3) -> Any:
    """Call vector storage service's /search endpoint with a text query."""
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/search", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def add_text(text: str, metadata: dict | None = None) -> Any:
    """Call vector storage service's /store endpoint to store a document."""
    payload = {"text": text, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/store", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def add_vector(vector: list[float], metadata: dict | None = None) -> Any:
    """Send an externally computed vector to the vector service's /insert endpoint."""
    payload = {"vector": vector, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/insert", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def query_vector(vector: list[float], top_k: int = 3) -> Any:
    """Call vector storage service's /lookup endpoint with a raw vector."""
    payload = {"vector": vector, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/lookup", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def semantic_search(query: str, top_k: int = 5) -> Any:
    """Call vector storage service's /search/semantic endpoint with similarity scores."""
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/search/semantic", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


import httpx
from app.core.config import VECTOR_SERVICE_URL
from typing import Any


async def query_text(query: str, top_k: int = 3) -> Any:
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/search", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def add_text(text: str, metadata: dict | None = None) -> Any:
    payload = {"text": text, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/store", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def add_vector(vector: list[float], metadata: dict | None = None) -> Any:
    payload = {"vector": vector, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/insert", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def query_vector(vector: list[float], top_k: int = 3) -> Any:
    payload = {"vector": vector, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/lookup", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


async def semantic_search(query: str, top_k: int = 5) -> Any:
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/search/semantic", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


# 🔥 NEW METHOD (IMPORTANT)
async def retrieve(query: str, top_k: int = 5):
    """Unified retrieval wrapper for evaluation"""

    result = await semantic_search(query, top_k)

    docs = result.get("results", [])

    return [
        {"text": d.get("text") or d.get("document")}
        for d in docs
    ]
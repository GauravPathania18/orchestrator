import httpx
from app.core.config import VECTOR_SERVICE_URL
from typing import Any


def _check_response(data: dict, service_name: str = "Vector Service") -> dict:
    """Validate standard contract response and extract data payload."""
    if data.get("status") != "success":
        error = data.get("error", {})
        raise RuntimeError(f"{service_name} error: {error.get('message', 'Unknown error')}")
    return data.get("data", {})


async def query_text(query: str, top_k: int = 3) -> Any:
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/vectors/search", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    return _check_response(data)["results"]


async def add_text(text: str, metadata: dict | None = None) -> Any:
    payload = {"text": text, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/vectors/store", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    return _check_response(data)


async def add_vector(vector: list[float], metadata: dict | None = None) -> Any:
    payload = {"vector": vector, "metadata": metadata or {}}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/vectors/insert", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    return _check_response(data)


async def query_vector(vector: list[float], top_k: int = 3) -> Any:
    payload = {"vector": vector, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/vectors/lookup", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    return _check_response(data)["results"]


async def semantic_search(query: str, top_k: int = 5) -> Any:
    payload = {"query": query, "top_k": top_k}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{VECTOR_SERVICE_URL}/vectors/search/semantic", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    return _check_response(data)["results"]


# NEW METHOD (IMPORTANT)
async def retrieve(query: str, top_k: int = 5):
    """Unified retrieval wrapper for evaluation"""
    result = await semantic_search(query, top_k)
    return [{"text": d.get("text") or d.get("document")} for d in result]
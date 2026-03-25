import httpx
from app.core.config import EMBEDDING_SERVICE_URL
from typing import List


async def get_embedding(text: str) -> List[float]:
    """Call the external embedding service and return a single embedding vector."""
    payload = {"texts": [text], "source": "user_prompt"}

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{EMBEDDING_SERVICE_URL}/embed", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

    # Check standard contract status
    if data.get("status") != "success":
        error = data.get("error", {})
        raise RuntimeError(error.get("message", "Embedding service failed"))

    # Extract from data payload
    items = data.get("data", {}).get("items") or []
    if not items:
        raise RuntimeError("Embedding service returned no items")

    return items[0]["vector"]

import requests
import logging
from fastapi import HTTPException
from ..config import EMBEDDER_URL
from .cache_manager import cache_manager
from .error_handler import CircuitBreaker, ConnectionError as RaptorConnectionError

VECTOR_DIMENSION = 768

# Circuit breaker for embedder service protection
_embedder_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


async def _fetch_embedding(text: str):
    """Internal async wrapper for the HTTP call."""
    payload = {"texts": [text]}
    resp = requests.post(EMBEDDER_URL, json=payload)
    resp.raise_for_status()
    return resp.json()


def get_embedding(text: str):
    # 1. Check cache first
    cached = cache_manager.get_cached_embedding(text)
    if cached:
        logging.debug(f"Cache hit for embedding (len={len(text)})")
        return cached

    # 2. Cache miss — call the embedder service with circuit breaker protection
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        data = loop.run_until_complete(_embedder_circuit.call(_fetch_embedding, text))

        # Check standard contract status
        if data.get("status") != "success":
            error = data.get("error", {})
            raise ValueError(f"Embedder error: {error.get('message', 'Unknown error')}")

        items = data.get("data", {}).get("items") or []
        if not items:
            raise ValueError("No embeddings returned")

        try:
            vector = items[0]["vector"]
        except (KeyError, IndexError) as e:
            logging.error(f"[Embedder] Unexpected response format: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid embedder response format: missing key {e}")

        # Update dimension if needed
        global VECTOR_DIMENSION
        if len(vector) != VECTOR_DIMENSION:
            VECTOR_DIMENSION = len(vector)
            logging.info(f"Updated VECTOR_DIMENSION to {VECTOR_DIMENSION}")

        # 3. Store in cache before returning
        cache_manager.cache_embedding(text, vector)

        return vector

    except RaptorConnectionError as e:
        logging.error(f"[Embedder] Circuit breaker open: {e}")
        raise HTTPException(status_code=503, detail="Embedder service unavailable (circuit open)")
    except KeyError as e:
        logging.error(f"[Embedder] Unexpected response format: {e}")
        raise HTTPException(status_code=500, detail=f"Invalid embedder response format: missing key {e}")
    except requests.exceptions.RequestException as e:
        logging.error(f"[Embedder] HTTP error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedder error: {e}")


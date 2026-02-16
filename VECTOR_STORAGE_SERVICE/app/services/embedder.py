import requests
import logging
from fastapi import HTTPException
from ..config import EMBEDDER_URL

# Default embedding dimension - should match the embedder service output
# This is typically 384 for all-MiniLM-L6-v2 or 768 for all-mpnet-base-v2
VECTOR_DIMENSION = 768

def get_embedding(text: str):
    try:
        payload = {"texts": [text]}
        resp = requests.post(EMBEDDER_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "items" not in data or not data["items"]:
            raise ValueError("No embeddings returned")
        vector = data["items"][0]["vector"]
        # Update VECTOR_DIMENSION if we receive a different size
        global VECTOR_DIMENSION
        if len(vector) != VECTOR_DIMENSION:
            VECTOR_DIMENSION = len(vector)
            logging.info(f"Updated VECTOR_DIMENSION to {VECTOR_DIMENSION}")
        return vector
    except requests.exceptions.RequestException as e:
        logging.error(f"[Embedder] HTTP error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedder error: {e}")


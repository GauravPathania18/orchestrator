import json
import logging
import httpx
from typing import Dict, Any

from .utils import normalize_metadata
from .vector_store import get_vector_store
from ..config import OLLAMA_URL
from .embedder import VECTOR_DIMENSION

# -------------------------
# STRICT CONTRACT
# -------------------------

ALLOWED_DOMAINS = {"movies", "sports", "tech", "general"}
ALLOWED_ENTITY_TYPES = {
    "fictional_character",
    "real_person",
    "organization",
    "concept",
    "unknown"
}
ALLOWED_SOURCES = {"user", "wiki", "pdf", "web", "memory"}

FALLBACK_METADATA = {
    "domain": "general",
    "entity_type": "unknown",
    "entity_name": "unknown",
    "source": "memory",
    "confidence": 0.0
}

METADATA_MODEL = "gemma3:1b"

METADATA_PROMPT = """
Return ONLY valid JSON.

Schema:
{
  "domain": one of ["movies","sports","tech","general"],
  "entity_type": one of ["fictional_character","real_person","organization","concept","unknown"],
  "entity_name": string or "unknown",
  "source": one of ["user","wiki","pdf","web","memory"],
  "confidence": number between 0 and 1
}

TEXT:
"""

# -------------------------
# VALIDATOR
# -------------------------

def validate_metadata(raw: dict) -> dict:
    try:
        if raw.get("domain") not in ALLOWED_DOMAINS:
            return FALLBACK_METADATA

        if raw.get("entity_type") not in ALLOWED_ENTITY_TYPES:
            return FALLBACK_METADATA

        if raw.get("source") not in ALLOWED_SOURCES:
            return FALLBACK_METADATA

        confidence = float(raw.get("confidence", 0.0))
        if not (0.0 <= confidence <= 1.0):
            return FALLBACK_METADATA

        raw["entity_name"] = raw.get("entity_name", "").strip() or "unknown"
        raw["confidence"] = confidence

        return raw

    except Exception:
        return FALLBACK_METADATA


# -------------------------
# GENERATE METADATA
# -------------------------

async def generate_metadata(text: str) -> dict:
    async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
        try:
            resp = await client.post(
                "/api/generate",
                json={
                    "model": METADATA_MODEL,
                    "prompt": METADATA_PROMPT + text[:1000],
                    "stream": False
                },
                timeout=10
            )
            resp.raise_for_status()

            raw_output = resp.json().get("response", "").strip()
            if not raw_output:
                return FALLBACK_METADATA

            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                start = raw_output.find("{")
                end = raw_output.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(raw_output[start:end + 1])
                else:
                    return FALLBACK_METADATA

            return validate_metadata(parsed)

        except Exception as e:
            logging.warning(f"[Metadata] Failed → fallback used: {e}")
            return FALLBACK_METADATA


# -------------------------
# MERGE STRATEGY
# -------------------------

def merge_metadata(old: Any, new: Any) -> Any:
    """
    Simple strategy:
    - Higher confidence wins
    - Otherwise keep old
    """
    if new.get("confidence", 0) >= old.get("confidence", 0):
        return {**old, **new}
    return old


# -------------------------
# BACKGROUND UPDATE
# -------------------------

async def update_metadata_in_chroma(doc_id: str, text: str):
    try:
        vector_store = get_vector_store(VECTOR_DIMENSION)

        metadata = await generate_metadata(text)
        metadata = normalize_metadata(metadata)

        existing = vector_store.get_by_id(doc_id)

        if not existing["ids"]:
            logging.warning(f"Document {doc_id} not found")
            return

        # Get metadatas with proper null check
        metadatas_list = existing.get("metadatas") or []
        old_meta = metadatas_list[0] if metadatas_list else {}
        merged = merge_metadata(old_meta, metadata)
        
        # ✅ Update status to completed
        merged["status"] = "completed"

        vector_store.update_metadata(doc_id, merged)

        logging.info(f"Metadata updated for {doc_id}")

    except Exception as e:
        logging.error(f"Metadata update failed: {e}", exc_info=True)

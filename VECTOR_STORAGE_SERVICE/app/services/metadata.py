import logging, json
import httpx
from .utils import normalize_metadata
from .vector_store import vector_store
from ..config import OLLAMA_URL

# -------------------------
# STRICT METADATA CONTRACT
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

# -------------------------
# LIGHTWEIGHT MODEL (1–2B)
# -------------------------
# Good options:
# - llama3.2:1b
# - qwen2.5:1.5b
# - phi-2

METADATA_MODEL = "llama3.2:1b"

# -------------------------
# STRICT PROMPT
# -------------------------

METADATA_PROMPT = """
You are a metadata classification system.

Return ONLY valid JSON.
Do NOT include explanations.
Do NOT add extra fields.

Schema:
{
  "domain": one of ["movies","sports","tech","general"],
  "entity_type": one of ["fictional_character","real_person","organization","concept","unknown"],
  "entity_name": string or "unknown",
  "source": one of ["user","wiki","pdf","web","memory"],
  "confidence": number between 0 and 1
}

If unsure, use "unknown" and low confidence.

TEXT:
"""

# -------------------------
# VALIDATOR (NON-ML)
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
# ASYNC METADATA GENERATION
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
                timeout=30
            )
            resp.raise_for_status()

            raw_output = resp.json().get("response", "").strip()
            if not raw_output:
                return FALLBACK_METADATA

            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                start, end = raw_output.find("{"), raw_output.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(raw_output[start:end + 1])
                else:
                    return FALLBACK_METADATA

            return validate_metadata(parsed)

        except Exception as e:
            logging.error(f"[Metadata] Ollama failed: {e}", exc_info=True)
            return FALLBACK_METADATA

# -------------------------
# BACKGROUND TASK (NO THREADS)
# -------------------------

async def update_metadata_in_chroma(doc_id: str, text: str):
    try:
        logging.info(f"🧠 Generating metadata for {doc_id}")

        metadata = await generate_metadata(text)
        metadata = normalize_metadata(metadata)

        existing = vector_store.collection.get(ids=[doc_id])
        if not existing["ids"]:
            logging.warning(f"❌ Document {doc_id} not found")
            return

        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        merged_meta = {**old_meta, **metadata}

        vector_store.collection.update(
            ids=[doc_id],
            metadatas=[merged_meta]
        )

        logging.info(f"✅ Metadata updated for {doc_id}")

    except Exception as e:
        logging.error(f"[Metadata Update] Failed for {doc_id}: {e}", exc_info=True)

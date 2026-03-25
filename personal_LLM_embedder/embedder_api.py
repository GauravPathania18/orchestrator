# embedder_api.py
"""
Personal LLM Embedding Service

A FastAPI-based service that generates vector embeddings from text.
Supports local models via SentenceTransformers.
"""

# ----------------- STANDARD LIBRARIES -----------------
import os          # For reading environment variables
import re          # Regular expressions for text cleaning
import time        # To measure processing time
import hashlib      # For generating deterministic IDs
from typing import List, Optional  # Type hints
from contextlib import asynccontextmanager  # For FastAPI lifespan context 
from concurrent.futures import ThreadPoolExecutor
# ----------------- THIRD-PARTY LIBRARIES -----------------
from fastapi import FastAPI, HTTPException, Request  # Web framework and exception handling
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool  # To run blocking code in a thread pool
from pydantic import BaseModel, field_validator, model_validator  # Data validation for request/response models
from bs4 import BeautifulSoup  # HTML parsing and stripping
import numpy as np  # Numerical operations

# SentenceTransformer is optional; raise error if missing
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

# ----------------- CONFIGURATION -----------------
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")  # "local" / "openai" / "gemini"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "all-mpnet-base-v2")  # Default local model
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda" (GPU)
HTML_STRIP = os.getenv("HTML_STRIP", "true").strip().lower() in ("1", "true", "yes")
 # Whether to strip HTML from text
NORMALIZE = os.getenv("NORMALIZE", "true").strip().lower() in ("1", "true", "yes")

MAX_CACHE_SIZE = 10000  # Prevent uncontrolled memory growth

# ----------------- BATCHING CONFIGURATION -----------------
# Maximum batch size for a single request (reject larger requests)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
# Internal processing batch size (process large batches in chunks to control memory)
INTERNAL_BATCH_SIZE = int(os.getenv("INTERNAL_BATCH_SIZE", "32"))

# ----------------- SECURITY: ALLOWLISTED MODELS -----------------
# Models that don't require trust_remote_code (safe, no arbitrary code execution)
TRUSTED_MODELS = {
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "all-distilroberta-v1",
    "paraphrase-multilingual-MiniLM-L12-v2",
}

def validate_model(model_name: str) -> None:
    """Validate that the model is in the trusted allowlist."""
    if model_name not in TRUSTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not in the trusted allowlist. "
            f"Allowed models: {', '.join(sorted(TRUSTED_MODELS))}. "
            f"Add it to TRUSTED_MODELS in embedder_api.py if you're sure it's safe."
        )

# ----------------- IMPORT LRU CACHE FROM VECTOR STORAGE SERVICE -----------------
import sys
from pathlib import Path

# Add project root to path for importing cache_manager
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from VECTOR_STORAGE_SERVICE.app.services.cache_manager import LRUCache, CacheConfig
except ImportError as e:
    raise ImportError(f"Failed to import LRUCache from cache_manager: {e}")

# ----------------- GLOBALS -----------------
# Global variable to hold the loaded embedding model
_model: Optional[SentenceTransformer] = None
VECTOR_DIMENSION: Optional[int] = None  # Will be set after loading the model to avoid hardcoding

# Initialize LRU cache with proper config (reused from cache_manager)
_embedding_cache = LRUCache(CacheConfig(max_size=MAX_CACHE_SIZE, ttl_seconds=7200))

# Precompiled regex for simple HTML tag detection (fast path)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# ----------------- FASTAPI APP WITH LIFESPAN -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic for the FastAPI app.
    Loads the embedding model at startup based on configuration.
    """
    global _model, VECTOR_DIMENSION

    if EMBEDDING_MODE != "local":
        raise ValueError("Currently only 'local' EMBEDDING_MODE is supported.")

    print(f"[Startup] Loading model: {LOCAL_MODEL} on device {DEVICE}")
    
    # Security: validate model is in allowlist
    validate_model(LOCAL_MODEL)
    
    _model = SentenceTransformer(
        LOCAL_MODEL, 
        device=DEVICE, 
        model_kwargs={"attn_implementation": "eager"}
        # NOTE: trust_remote_code removed for security. 
        # Only allowlisted models can be loaded (see TRUSTED_MODELS above).
    )
    
    # Set vector dimension based on the model's output
    # Dimension Locking: Ensure VECTOR_DIMENSION matches the model's output dimension to prevent mismatches
    test_vector = _model.encode(["dimension check"], convert_to_numpy=True)
    VECTOR_DIMENSION = test_vector.shape[1]
    print(f"[Embedding] Vector dimension locked at {VECTOR_DIMENSION}")

    # Warm-up run with realistic sentence
    print("[Startup] Warming up model...")
    _model.encode(["This is a realistic warmup sentence for embedding model performance testing."], convert_to_numpy=True)

    yield 

# ----------------- FASTAPI APP -----------------
# Initialize FastAPI app with custom lifespan
app = FastAPI(
    title="Personal LLM Embedding Service",
    version="3.1",
    lifespan=lifespan
)

def _strip_html_if_present(s: str) -> str:
    """
    Strip HTML only if:
      1) HTML_STRIP is enabled AND
      2) The input likely contains HTML tags (regex detection)
    """
    if not HTML_STRIP or not s:
        return s
    
    # Fast-path check: only call BeautifulSoup if tags are found
    if _HTML_TAG_RE.search(s):
        return BeautifulSoup(s, "html.parser").get_text(" ")
    return s


# ----------------- DATA SCHEMAS -----------------
class EmbedRequest(BaseModel):
    """
    Request schema for embedding endpoint.
    Accepts either a single text or a list of texts.
    """
    
    texts: Optional[List[str]] = None  # List of texts to embed; can also accept a single string (handled in validator)


    # ---- Field-level cleaning: full clean at validation boundary ----
    @field_validator("texts", mode="before")
    @classmethod
    def ensure_list_and_clean(cls, v):
        # Case 1: single string → wrap into list
        if isinstance(v, str):
            v = [v]
        
        # Case 2: list of strings → clean each
        if isinstance(v, list):
            cleaned = []
            for s in v:
                if isinstance(s, str):
                    # Full clean: HTML strip + whitespace normalize
                    s = _strip_html_if_present(s)
                    s = _WHITESPACE_RE.sub(" ", s).strip()
                    if s:
                        cleaned.append(s)
            return cleaned
        
        # Invalid input
        raise ValueError("texts must be a string or list of strings")
    
    # ---- Model-level rule: enforce exactly one of text/texts ----
    @model_validator(mode="after")
    def check_not_empty(self):
        """
        Ensure `texts` exists and has at least one non-empty string.
        """
        if not self.texts or all(len(s) == 0 for s in self.texts):
            raise ValueError("At least one non-empty string is required.")
        return self

class EmbedItem(BaseModel):
    """Single embedding item with ID, vector, and metadata."""
    id: str
    vector: List[float]
    metadata: dict = {}

class StandardResponse(BaseModel):
    """Standard API response wrapper for all endpoints."""
    status: str  # "success" or "error"
    data: Optional[dict] = None
    error: Optional[dict] = None

class EmbedData(BaseModel):
    """Data payload for embedding responses."""
    mode: str
    model_name: str
    vector_size: int
    count: int
    processing_time_sec: float
    items: List[EmbedItem]

class HealthData(BaseModel):
    """Data payload for health check responses."""
    mode: str
    model: str
    device: str
    vector_dimension: Optional[int]
    normalize: bool
    cache_size: int


# ----------------- PARALLEL BATCH PROCESSING -----------------
from concurrent.futures import ThreadPoolExecutor

def _process_batch(batch: List[str]) -> np.ndarray:
    """Process a single batch of texts for parallel execution."""
    return _model.encode(batch, convert_to_numpy=True, batch_size=len(batch))


# _clean() removed - cleaning now happens exclusively in Pydantic validator
# This eliminates redundant processing and ensures single-source-of-truth


def _embed_local(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using a local SentenceTransformer model.
    Enforces dimension consistency and optional normalization.
    Uses LRU cache with hashed keys to avoid redundant computations.
    Processes batches in parallel for 2-5x speed improvement.
    """
    if _model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    # Pre-hash all texts once for efficient cache lookups
    cache_keys = [hashlib.md5(text.encode()).hexdigest() for text in texts]
    
    cached_results = {}
    uncached_indices = []
    uncached_texts = []

    # Separate cached vs uncached using LRU cache with hashed keys
    for idx, (text, cache_key) in enumerate(zip(texts, cache_keys)):
        cached = _embedding_cache.get(cache_key)
        if cached is not None:
            cached_results[idx] = cached
        else:
            uncached_indices.append(idx)
            uncached_texts.append(text)

    # Log cache hit ratio for monitoring
    cache_hits = len(texts) - len(uncached_texts)
    if texts:
        hit_rate = cache_hits / len(texts)
        print(f"[Embedding] Cache hit rate: {cache_hits}/{len(texts)} ({hit_rate:.1%})")

    # Process uncached texts in parallel batches for maximum throughput
    if uncached_texts:
        # Create batches
        batches = [
            uncached_texts[i:i + INTERNAL_BATCH_SIZE]
            for i in range(0, len(uncached_texts), INTERNAL_BATCH_SIZE)
        ]
        
        # Parallel processing with ThreadPoolExecutor (2-5x speedup)
        # Use 2 workers to avoid GIL contention while maximizing CPU utilization
        with ThreadPoolExecutor(max_workers=2) as executor:
            batch_results = list(executor.map(_process_batch, batches))
        
        # Concatenate all batch results
        raw_vectors = np.vstack(batch_results)

        if VECTOR_DIMENSION is None:
            raise HTTPException(status_code=500, detail="Vector dimension not initialized.")

        if raw_vectors.shape[1] != VECTOR_DIMENSION:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding dimension mismatch. "
                       f"Expected {VECTOR_DIMENSION}, got {raw_vectors.shape[1]}"
            )

        if NORMALIZE:
            norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            raw_vectors = raw_vectors / norms

        # Store in LRU cache using pre-computed keys
        for idx_in_uncached, (original_idx, vector) in enumerate(zip(uncached_indices, raw_vectors)):
            vector_list = vector.tolist()
            cache_key = cache_keys[original_idx]
            _embedding_cache.set(cache_key, vector_list)
            cached_results[original_idx] = vector_list

    # Return results in original order
    return [cached_results[i] for i in range(len(texts))]

def _embed(texts: List[str]) -> List[List[float]]:
    """
    Route to the correct embedding method based on EMBEDDING_MODE.
    Currently only supports local embeddings.
    """
    if EMBEDDING_MODE == "local":
        return _embed_local(texts)
    raise HTTPException(status_code=501, detail=f"Unknown EMBEDDING_MODE: {EMBEDDING_MODE}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return HTTP exceptions in standard contract format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "data": None,
            "error": {
                "message": exc.detail,
                "code": f"HTTP_{exc.status_code}"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Return unhandled exceptions in standard contract format."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "data": None,
            "error": {
                "message": str(exc),
                "code": "INTERNAL_ERROR"
            }
        }
    )
@app.get("/", response_model=StandardResponse)
def health():
    """
    Health check endpoint.
    Returns service status, mode, model, device, and HTML stripping setting.
    """
    # Cap cache size reporting to avoid leaking detailed internal state
    raw_cache_size = _embedding_cache.get_stats()["size"]
    safe_cache_size = min(raw_cache_size, 1000)
    
    return {
        "status": "success",
        "data": {
            "mode": EMBEDDING_MODE,
            "model": LOCAL_MODEL,
            "device": DEVICE,
            "vector_dimension": VECTOR_DIMENSION,
            "normalize": NORMALIZE,
            "cache_size": safe_cache_size
        },
        "error": None
    }

@app.post("/embed", response_model=StandardResponse)
async def embed(req: EmbedRequest):
    """
    Generate embeddings for one or multiple texts.
    Returns embedding vectors with metadata and processing time.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")

    # Enforce maximum batch size to prevent abuse/OOM
    if len(req.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(req.texts)} exceeds maximum limit of {MAX_BATCH_SIZE}"
        )

    # No cleaning needed — Pydantic validator already handled it
    # req.texts is guaranteed clean at this point

    t0 = time.time()
    vectors = await run_in_threadpool(_embed, req.texts)  # use directly
    dt = time.time() - t0

    if not vectors:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.")

    items = []
    for text, vec in zip(req.texts, vectors):
        # Deterministic ID based on model + text hash (same model + text = same ID)
        # Includes model name to prevent collisions across different models
        deterministic_id = hashlib.sha256(
            f"{LOCAL_MODEL}:{text}".encode()
        ).hexdigest()[:16]
        items.append({
            "id": deterministic_id,
            "vector": vec,
            "metadata": {
                "model": LOCAL_MODEL,
                "dim": VECTOR_DIMENSION or 768
            }
        })

    return {
        "status": "success",
        "data": {
            "mode": EMBEDDING_MODE,
            "model_name": LOCAL_MODEL,
            "vector_size": VECTOR_DIMENSION or 768,
            "count": len(items),
            "processing_time_sec": round(dt, 4),
            "items": items
        },
        "error": None
    }

# -------- Run the server --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Embedding Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

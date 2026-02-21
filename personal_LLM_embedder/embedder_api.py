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
from typing import List, Optional  # Type hints
from contextlib import asynccontextmanager  # For FastAPI lifespan context
from threading import Lock  # For thread-safe model loading 

# ----------------- THIRD-PARTY LIBRARIES -----------------
from fastapi import FastAPI, HTTPException  # Web framework and exception handling
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

# ----------------- GLOBALS -----------------
# Global variable to hold the loaded embedding model
_model: Optional[SentenceTransformer] = None
VECTOR_DIMENSION: Optional[int] = None  # Will be set after loading the model to avoid hardcoding

_embedding_cache = {}
_cache_lock = Lock()

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
    _model = SentenceTransformer(
        LOCAL_MODEL, 
        device=DEVICE, 
        model_kwargs={"attn_implementation": "eager"},
        trust_remote_code=True
    )
    
    # Set vector dimension based on the model's output
    # Dimension Locking: Ensure VECTOR_DIMENSION matches the model's output dimension to prevent mismatches
    test_vector = _model.encode(["dimension check"], convert_to_numpy=True)
    VECTOR_DIMENSION = test_vector.shape[1]
    print(f"[Embedding] Vector dimension locked at {VECTOR_DIMENSION}")

    # Warm-up run
    print("[Startup] Warming up model...")
    _model.encode(["warm up run"], convert_to_numpy=True)

    yield 

# ----------------- FASTAPI APP -----------------
# Initialize FastAPI app with custom lifespan
app = FastAPI(
    title="Personal LLM Embedding Service",
    version="3.0",
    lifespan=lifespan
)

# ----------------- DATA SCHEMAS -----------------
class EmbedRequest(BaseModel):
    """
    Request schema for embedding endpoint.
    Accepts either a single text or a list of texts.
    """
    
    texts: Optional[List[str]] = None  # List of texts to embed; can also accept a single string (handled in validator)


    # ---- Field-level cleaning : trim strings early ----
    @field_validator("texts", mode="before")
    @classmethod
    def ensure_list_and_trim(cls, v):
        # Case 1: single string → wrap into list
        if isinstance(v, str):
            return [v.strip()]
        # Case 2: list of strings → trim each
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str)]
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

class EmbedBatchResponse(BaseModel):
    """
    Full response returned by the embedding endpoint.
    Contains metadata, processing time, and embedding vectors.
    """
    model_config = {"protected_namespaces": ()}
    
    mode: str
    model_name: str
    vector_size: int
    count: int
    processing_time_sec: float
    vectors: List[List[float]]

# ----------------- HELPER FUNCTIONS -----------------


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

def _clean(s: str) -> str:
    """
    Clean text by stripping HTML and normalizing whitespace.
    """
    s = _strip_html_if_present(s)
    return _WHITESPACE_RE.sub(" ", s.strip())


def _embed_local(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using a local SentenceTransformer model.
    Enforces dimension consistency and optional normalization.
    """
    if _model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")

    cached_results = {}
    uncached_texts = []

    # Separate cached vs uncached
    with _cache_lock:
        for text in texts:
            if text in _embedding_cache:
                cached_results[text] = _embedding_cache[text]
            else:
                uncached_texts.append(text)

    # Batch encode uncached texts
    if uncached_texts:
        raw_vectors = _model.encode(uncached_texts, convert_to_numpy=True)

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

        with _cache_lock:
            for text, vector in zip(uncached_texts, raw_vectors):
                if len(_embedding_cache) >= MAX_CACHE_SIZE:
                    _embedding_cache.clear()  # Simple protection strategy
                vector_list = vector.tolist()
                _embedding_cache[text] = vector_list
                cached_results[text] = vector_list

    return [cached_results[text] for text in texts]

def _embed(texts: List[str]) -> List[List[float]]:
    """
    Route to the correct embedding method based on EMBEDDING_MODE.
    Currently only supports local embeddings.
    """
    if EMBEDDING_MODE == "local":
        return _embed_local(texts)
    raise HTTPException(status_code=501, detail=f"Unknown EMBEDDING_MODE: {EMBEDDING_MODE}")

# ----------------- ROUTES -----------------
@app.get("/")
def health():
    """
    Health check endpoint.
    Returns service status, mode, model, device, and HTML stripping setting.
    """
    return {
        "status": "ok",
        "mode": EMBEDDING_MODE,
        "model": LOCAL_MODEL,
        "device": DEVICE,
        "vector_dimension": VECTOR_DIMENSION,
        "normalize": NORMALIZE,
        "cache_size": len(_embedding_cache)
    }

@app.post("/embed", response_model=EmbedBatchResponse)
async def embed(req: EmbedRequest):
    """
    Generate embeddings for one or multiple texts.
    Returns embedding vectors with metadata and processing time.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")

    # Clean once (efficient)
    cleaned_texts = []
    for x in req.texts:
        cleaned = _clean(x)
        if cleaned:
            cleaned_texts.append(cleaned)

    if not cleaned_texts:
        raise HTTPException(status_code=400, detail="All inputs empty after cleaning.")

    t0 = time.time()
    vectors = await run_in_threadpool(_embed, cleaned_texts)
    dt = time.time() - t0

    if not vectors:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.")

    return EmbedBatchResponse(
        mode=EMBEDDING_MODE,
        model_name=LOCAL_MODEL,
        vector_size=VECTOR_DIMENSION or 768,
        count=len(vectors),
        processing_time_sec=round(dt, 4),
        vectors=vectors,
    )

# -------- Run the server --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Embedding Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

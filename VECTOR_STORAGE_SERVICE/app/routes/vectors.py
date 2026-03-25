import logging
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from ..models import TextRequest, SearchRequest, QueryRequest
from ..services.vector_store import get_vector_store
from ..services.embedder import get_embedding
from ..services.metadata import update_metadata_in_chroma
from ..services.utils import clean_user_text, normalize_metadata
from ..services.error_handler import handle_exception

vector_store = get_vector_store()

router = APIRouter()

async def process_text_background(doc_id: str, text: str, clean_text: str):
    """Background task to handle embedding and metadata generation."""
    try:
        # 1. Generate embedding
        vector = get_embedding(clean_text)
        
        # 2. Update vector and initial metadata in Chroma
        vector_store.collection.update(
            ids=[doc_id],
            embeddings=[vector]
        )
        
        # 3. Generate and update advanced metadata (LLM based)
        await update_metadata_in_chroma(doc_id, clean_text)
        
        logging.info(f"✅ Background processing complete for {doc_id}")
    except Exception as e:
        logging.error(f"❌ Background processing failed for {doc_id}: {e}", exc_info=True)

# -------------------------
# ADD TEXT
# -------------------------
@router.post("/store")
@handle_exception
async def add_text(req: TextRequest, background_tasks: BackgroundTasks):
    try:
        clean_text = clean_user_text(req.text)
        if not clean_text:
            raise HTTPException(status_code=400, detail="Input text empty after cleaning.")

        # initial safe metadata
        metadata = req.metadata or {}
        metadata["text"] = req.text
        metadata["status"] = "processing"  # Changed from pending
        metadata = normalize_metadata(metadata)

        # Store with dummy vector (zeros) initially to get the ID
        from ..services.embedder import VECTOR_DIMENSION
        dummy_vector = [0.0] * VECTOR_DIMENSION
        
        doc_id = vector_store.store_vector(dummy_vector, metadata)

        # ✅ async background task handles BOTH embedding and metadata
        background_tasks.add_task(process_text_background, doc_id, req.text, clean_text)

        return {
            "status": "success",
            "data": {
                "id": doc_id,
                "text": req.text,
                "message": "Text accepted for processing",
                "metadata_status": "processing"
            },
            "error": None
        }

    except Exception as e:
        logging.error(f"[VectorStore] Failed: {e}", exc_info=True)
        return {
            "status": "error",
            "data": None,
            "error": {
                "message": str(e),
                "code": "STORE_FAILED"
            }
        }

# -------------------------
# LIST VECTORS
# -------------------------
@router.get("/list")
@handle_exception
async def list_vectors():
    items = vector_store.collection.get()
    ids = items.get("ids") or []
    documents = items.get("documents") or []
    metadatas = items.get("metadatas") or []
    
    return {
        "status": "success",
        "data": {
            "results": [
                {
                    "id": ids[i] if i < len(ids) else None,
                    "document": documents[i] if i < len(documents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                }
                for i in range(len(ids))
            ]
        },
        "error": None
    }


# -------------------------
# QUERY TEXT (EMBED + SEARCH)
# -------------------------
@router.post("/search")
@handle_exception
async def query_text(req: QueryRequest):
    try:
        query_vec = get_embedding(req.query)

        # temporary safe defaults (until intent router)
        results = vector_store.search(
            query_vector=query_vec,
            top_k=req.top_k or 3,
            domain=None,  # Don't filter by domain for RAPTOR compatibility
            entity_type=None
        )

        return {
            "status": "success",
            "data": {
                "query": req.query,
                "results": results
            },
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "error": {
                "message": str(e),
                "code": "SEARCH_FAILED"
            }
        }


# -------------------------
# ADD VECTOR (EXTERNAL EMBEDDING)
# -------------------------
@router.post("/insert")
@handle_exception
async def add_vector(req: SearchRequest):
    """Accepts an external vector with optional metadata and stores it."""
    try:
        if not req.vector or not isinstance(req.vector, list):
            raise HTTPException(status_code=400, detail="`vector` must be a non-empty list of floats")

        metadata = getattr(req, "metadata", {}) if hasattr(req, "metadata") else {}
        metadata = normalize_metadata(metadata)
        
        # Ensure metadata is not empty (ChromaDB requirement)
        if not metadata:
            metadata = {"source": "external_vector", "timestamp": str(__import__('datetime').datetime.now())}

        # allow optional text in metadata
        doc_id = vector_store.store_vector(req.vector, metadata)

        return {
            "status": "success",
            "data": {"id": doc_id},
            "error": None
        }

    except Exception as e:
        logging.error(f"[VectorStore:add_vector] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# SEMANTIC SEARCH
# -------------------------
@router.post("/search/semantic")
@handle_exception
async def semantic_search(req: QueryRequest):
    """
    Semantic search endpoint that returns results with similarity scores.
    Converts cosine similarity (distance) to a 0-100% similarity score.
    """
    try:
        query_vec = get_embedding(req.query)

        results = vector_store.search(
            query_vector=query_vec,
            top_k=req.top_k or 5,
            domain=None,  # Don't filter by domain for RAPTOR compatibility
            entity_type=None,
            min_confidence=0.0,
            max_distance=2.0
        )

        # Transform results with similarity scores
        formatted_results = []
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        for i in range(len(ids)):
            distance = distances[i] if i < len(distances) else 1.0
            similarity_score = round((1 - distance) * 100, 2)
            
            formatted_results.append({
                "id": ids[i],
                "text": documents[i] if i < len(documents) else None,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distance,
                "similarity_score": similarity_score
            })

        return {
            "status": "success",
            "data": {
                "query": req.query,
                "results_count": len(formatted_results),
                "results": formatted_results
            },
            "error": None
        }

    except Exception as e:
        logging.error(f"[VectorStore:semantic_search] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# QUERY BY VECTOR
# -------------------------
@router.post("/lookup")
@handle_exception
async def query_vector(req: SearchRequest):
    """Accepts a raw vector and returns nearest neighbours."""
    try:
        if not req.vector or not isinstance(req.vector, list):
            raise HTTPException(status_code=400, detail="`vector` must be a non-empty list of floats")

        results = vector_store.search(query_vector=req.vector, top_k=req.top_k or 5)
        return {
            "status": "success",
            "data": {"query": None, "results": results},
            "error": None
        }

    except Exception as e:
        logging.error(f"[VectorStore:query_vector] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

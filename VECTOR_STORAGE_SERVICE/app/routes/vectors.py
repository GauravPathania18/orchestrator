import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from ..models import TextRequest, SearchRequest, QueryRequest
from ..services.vector_store import vector_store
from ..services.embedder import get_embedding
from ..services.metadata import update_metadata_in_chroma
from ..services.utils import clean_user_text, normalize_metadata

router = APIRouter()

# -------------------------
# ADD TEXT
# -------------------------
@router.post("/add_text")
async def add_text(req: TextRequest, background_tasks: BackgroundTasks):
    try:
        clean_text = clean_user_text(req.text)
        if not clean_text:
            raise HTTPException(status_code=400, detail="Input text empty after cleaning.")

        vector = get_embedding(clean_text)

        # initial safe metadata
        metadata = req.metadata or {}
        metadata["text"] = req.text
        metadata["status"] = "pending"
        metadata = normalize_metadata(metadata)

        doc_id = vector_store.store_vector(vector, metadata)

        # ✅ async background task (NO THREADS)
        background_tasks.add_task(update_metadata_in_chroma, doc_id, clean_text)

        return {
            "status": "success",
            "id": doc_id,
            "text": req.text,
            "metadata_status": "pending"
        }

    except Exception as e:
        logging.error(f"[VectorStore] Failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# -------------------------
# LIST VECTORS
# -------------------------
@router.get("/vectors")
async def list_vectors():
    items = vector_store.collection.get()
    return {
        "status": "success",
        "results": [
            {
                "id": items["ids"][i],
                "document": items["documents"][i],
                "metadata": items["metadatas"][i]
            }
            for i in range(len(items["ids"]))
        ]
    }


# -------------------------
# QUERY TEXT (EMBED + SEARCH)
# -------------------------
@router.post("/query_text")
def query_text(req: QueryRequest):
    try:
        query_vec = get_embedding(req.query)

        # temporary safe defaults (until intent router)
        results = vector_store.search(
            query_vector=query_vec,
            top_k=req.top_k or 5,
            domain="general",
            entity_type=None
        )

        return {
            "status": "success",
            "query": req.query,
            "results": results
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -------------------------
# ADD VECTOR (EXTERNAL EMBEDDING)
# -------------------------
@router.post("/add_vector")
async def add_vector(req: SearchRequest):
    """Accepts an external vector with optional metadata and stores it."""
    try:
        if not req.vector or not isinstance(req.vector, list):
            raise HTTPException(status_code=400, detail="`vector` must be a non-empty list of floats")

        metadata = getattr(req, "metadata", {}) if hasattr(req, "metadata") else {}
        metadata = normalize_metadata(metadata)

        # allow optional text in metadata
        doc_id = vector_store.store_vector(req.vector, metadata)

        return {"status": "success", "id": doc_id}

    except Exception as e:
        logging.error(f"[VectorStore:add_vector] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# QUERY BY VECTOR
# -------------------------
@router.post("/query_vector")
async def query_vector(req: SearchRequest):
    """Accepts a raw vector and returns nearest neighbours."""
    try:
        if not req.vector or not isinstance(req.vector, list):
            raise HTTPException(status_code=400, detail="`vector` must be a non-empty list of floats")

        results = vector_store.search(query_vector=req.vector, top_k=req.top_k or 5)
        return {"status": "success", "query": None, "results": results}

    except Exception as e:
        logging.error(f"[VectorStore:query_vector] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

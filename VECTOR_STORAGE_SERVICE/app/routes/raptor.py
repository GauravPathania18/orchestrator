import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from ..services.raptor_builder import RaptorBuilder
from ..services.raptor_retriever import RaptorRetriever
from ..services.reranker import Reranker
from ..services.pipeline import RetrievalPipeline
from ..services.vector_store import get_vector_store
from ..services.cache_manager import cache_manager
from ..services.error_handler import handle_exception
from ..config_manager import config_manager

vector_store = get_vector_store()

router = APIRouter()

# Request/Response Models
class IngestRequest(BaseModel):
    documents: List[str]
    cluster_size: Optional[int] = None
    chunk_size: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    k_summary: Optional[int] = 3
    k_chunks: Optional[int] = 10
    top_k_final: Optional[int] = 5
    return_intermediate: Optional[bool] = False
    return_scores: Optional[bool] = False

# Global pipeline instance (lazy initialization)
_raptor_pipeline: Optional[RetrievalPipeline] = None

def get_raptor_pipeline() -> RetrievalPipeline:
    """Get or create RAPTOR pipeline instance"""
    global _raptor_pipeline
    if _raptor_pipeline is None:
        _raptor_pipeline = RetrievalPipeline(vector_store)
        logging.info("🚀 RAPTOR Pipeline initialized")
    return _raptor_pipeline

@router.post("/index")
@handle_exception
async def ingest_documents(req: IngestRequest):
    """
    Ingest documents using RAPTOR hierarchical clustering
    
    - Chunks documents
    - Creates embeddings
    - Clusters similar chunks
    - Generates summaries per cluster
    - Stores hierarchical structure
    """
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    try:
        # Get config values (from request or fallback to config manager)
        raptor_config = config_manager.get_raptor_config()
        cluster_size = req.cluster_size or raptor_config.cluster_size
        chunk_size = req.chunk_size or raptor_config.chunk_size
        
        builder = RaptorBuilder(vector_store, cluster_size=cluster_size, chunk_size=chunk_size)
        result = await builder.ingest(req.documents)
        
        logging.info(f"✅ RAPTOR ingestion completed: {result}")
        return {
            "status": "success",
            "data": result,
            "error": None
        }
    except Exception as e:
        logging.error(f"❌ RAPTOR ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
@handle_exception
async def raptor_query(req: QueryRequest):
    """
    Query using RAPTOR hierarchical retrieval pipeline

    Flow:
    1. Check RAPTOR cache first
    2. Get summaries relevant to query
    3. Extract cluster IDs from summaries
    4. Get chunks from those clusters
    5. Rerank combined results
    6. Return final context
    """
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Check RAPTOR cache first
        cached = cache_manager.get_cached_raptor_result(
            req.query,
            k_summary=req.k_summary or 3,
            k_chunks=req.k_chunks or 10,
            top_k_final=req.top_k_final or 5
        )
        if cached:
            logging.info("RAPTOR cache hit")
            return {"status": "success", "data": cached, "error": None}

        pipeline = get_raptor_pipeline()

        # Update pipeline config if provided
        if any([req.k_summary, req.k_chunks, req.top_k_final]):
            pipeline.update_config(
                k_summary=req.k_summary,
                k_chunks=req.k_chunks,
                top_k_final=req.top_k_final
            )

        result = pipeline.run(
            query=req.query,
            return_intermediate=req.return_intermediate or False,
            return_scores=req.return_scores or False
        )

        # Store result in cache
        cache_manager.cache_raptor_result(req.query, result)

        logging.info(f"✅ RAPTOR query completed: {len(result.get('final_documents', []))} docs")
        return {
            "status": "success",
            "data": result,
            "error": None
        }
    except Exception as e:
        logging.error(f"❌ RAPTOR query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
@handle_exception
async def get_pipeline_stats():
    """Get RAPTOR pipeline statistics and configuration"""
    try:
        pipeline = get_raptor_pipeline()
        stats = pipeline.get_pipeline_stats()
        
        return {
            "status": "success",
            "data": stats,
            "error": None
        }
    except Exception as e:
        logging.error(f"❌ Failed to get pipeline stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
@handle_exception
async def reset_raptor():
    """Reset RAPTOR pipeline (clear all RAPTOR data)"""
    try:
        # Delete all RAPTOR-related data
        vector_store.collection.delete(
            where={"type": "summary"}
        )
        vector_store.collection.delete(
            where={"type": "chunk"}
        )
        
        # Reset pipeline instance
        global _raptor_pipeline
        _raptor_pipeline = None
        
        logging.info("🗑️ RAPTOR data reset completed")
        return {
            "status": "success",
            "data": {"message": "RAPTOR data cleared successfully"},
            "error": None
        }
    except Exception as e:
        logging.error(f"❌ Failed to reset RAPTOR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

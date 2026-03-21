"""
RAPTOR-specific API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.chat import ChatRequest, MemoryRequest
from app.services.enhanced_rag import run_enhanced_rag, store_memory_with_raptor, get_raptor_stats
from app.api.auth import get_current_user
import logging

router = APIRouter()

@router.post("/chat")
async def raptor_chat(req: ChatRequest, current_user: str = Depends(get_current_user)):
    """
    Enhanced chat using RAPTOR hierarchical retrieval
    
    - Uses RAPTOR for better context organization
    - Falls back to simple RAG if RAPTOR fails
    - Provides detailed retrieval information
    """
    try:
        resp = await run_enhanced_rag(
            req.message, 
            session_id=req.session_id, 
            top_k=req.top_k or 5,
            use_raptor=True  # Force RAPTOR usage
        )
        
        return {
            "status": "success",
            "data": resp
        }
    except Exception as e:
        logging.error(f"RAPTOR chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def raptor_ingest(req: MemoryRequest, current_user: str = Depends(get_current_user)):
    """
    Ingest documents using RAPTOR hierarchical clustering
    
    Better organization through semantic clustering and summarization
    """
    try:
        res = await store_memory_with_raptor(req.text, metadata={"session_id": req.session_id} if req.session_id else {})
        return {"status": "success", "data": res}
    except Exception as e:
        logging.error(f"RAPTOR ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def raptor_statistics(current_user: str = Depends(get_current_user)):
    """Get RAPTOR pipeline statistics and health"""
    try:
        stats = await get_raptor_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logging.error(f"Failed to get RAPTOR stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

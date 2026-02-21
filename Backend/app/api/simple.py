"""
Simplified API with only essential endpoints.
"""
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, MemoryRequest
from app.services.rag_pipeline import run_rag_with_scores, store_memory
from app.services.short_term_memory import session_manager

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Essential chat endpoint with automatic session management.
    
    - Auto-manages sessions (60min timeout)
    - Uses gemma3:1b for responses
    - Searches both short-term and long-term memory
    """
    # Get or create session (automatic time-based management)
    active_session_id = await session_manager.add_message(
        role="user",
        message=req.message,
        session_id=req.session_id  # If None, auto-manages
    )
    
    # Get response with RAG (searches both memories)
    resp = await run_rag_with_scores(
        req.message, 
        session_id=active_session_id, 
        top_k=req.top_k or 5
    )
    
    # Store assistant response
    await session_manager.add_message(
        role="assistant",
        message=resp.get("answer", ""),
        session_id=active_session_id
    )
    
    return {
        "status": "success", 
        "data": resp,
        "session_id": active_session_id
    }


@router.post("/memory")
async def memory(req: MemoryRequest):
    """Store a memory/document into vector DB."""
    # Only include session_id in metadata if it's provided
    metadata = {}
    if req.session_id:
        metadata["session_id"] = req.session_id
    
    res = await store_memory(req.text, metadata=metadata)
    return {"status": "success", "data": res}


@router.get("/session/current")
async def get_current_session():
    """Get current active session."""
    current = session_manager.get_current_session()
    if current:
        return {"status": "success", "session_id": current}
    return {"status": "success", "session_id": None}

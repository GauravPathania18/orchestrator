from fastapi import APIRouter
from app.schemas.chat import ChatRequest, MemoryRequest
from app.services.rag_pipeline import run_rag, run_rag_with_scores, store_memory
from app.services.vector_client import semantic_search
from app.services.short_term_memory import session_manager

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    Orchestrator chat endpoint with automatic session management.
    
    Session behavior:
    - If session_id provided: uses that specific session
    - If no session_id: automatically manages sessions based on time
      - Reuses current active session if < 60min since last message
      - Creates new session if previous expired (old one gets summarized & stored)
    """
    # Get or create session (automatic time-based management)
    active_session_id = await session_manager.add_message(
        role="user",
        message=req.message,
        session_id=req.session_id  # If None, auto-manages; if provided, uses that session
    )
    
    # Get response with RAG (searches both short-term and long-term memory)
    resp = await run_rag_with_scores(
        req.message, 
        session_id=active_session_id, 
        top_k=req.top_k or 5
    )
    
    # Store assistant response in short-term memory
    await session_manager.add_message(
        role="assistant",
        message=resp.get("answer", ""),
        session_id=active_session_id
    )
    
    # Include session info in response
    session_info = session_manager.get_session_info(active_session_id)
    
    return {
        "status": "success", 
        "data": resp,
        "session": {
            "session_id": active_session_id,
            "is_new": req.session_id is None and session_info and session_info.get("message_count", 0) <= 2,
            "message_count": session_info.get("message_count", 0) if session_info else 0
        }
    }


@router.post("/semantic_search")
async def semantic_search_endpoint(req: ChatRequest):
    """Semantic search endpoint that returns results with similarity scores."""
    results = await semantic_search(req.message, top_k=req.top_k or 5)
    return {"status": "success", "data": results}


@router.post("/memory")
async def memory(req: MemoryRequest):
    """Store a memory/document into the vector DB tied to an optional session_id."""
    # Only include session_id in metadata if it's provided
    metadata = {}
    if req.session_id:
        metadata["session_id"] = req.session_id
    
    res = await store_memory(req.text, metadata=metadata)
    return {"status": "success", "data": res}

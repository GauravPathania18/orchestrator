"""
Simplified API with only essential endpoints.
"""
from fastapi import APIRouter, Depends, Query
from app.schemas.chat import ChatRequest, MemoryRequest, FeedbackRequest
from app.services.enhanced_rag import run_enhanced_rag, store_memory_with_raptor
from app.services.short_term_memory import session_manager
from app.api.auth import get_current_user
from app.services.memory.profile_builder import extract_user_profile
from app.services.vector_client import add_text
import logging
from datetime import datetime
import uuid

router = APIRouter()

@router.post("/feedback")
async def feedback(req: FeedbackRequest, current_user: str = Depends(get_current_user)):
    """
    Store user feedback on a RAG response for evaluation.
    """
    logging.info(f"Feedback received from {current_user}: {req.dict()}")
    # In a real app, store this in a database for RAGAS evaluation
    return {"status": "success", "data": {"message": "Feedback received"}, "error": None}


@router.post("/chat")
async def chat(req: ChatRequest, current_user: str = Depends(get_current_user)):
    """
    Essential chat endpoint with automatic session management.
    
    - Auto-manages sessions (60min timeout)
    - Uses configured Ollama model for responses
    - Searches both short-term and long-term memory
    """
    # Get or create session (automatic time-based management)
    active_session_id = await session_manager.add_message(
        role="user",
        message=req.message,
        session_id=req.session_id  # If None, auto-manages
    )
    # -----------------------------
    # USER PROFILE EXTRACTION
    # -----------------------------
    try:
        profile_memories = extract_user_profile(req.message)

        for mem in profile_memories:
            # Optional: filter very small / useless entries
            if len(mem["value"].split()) < 5:
                continue

            await add_text(
                text=mem["value"],
                metadata={
                    "id": str(uuid.uuid4()),
                    "type": "profile",
                    "category": mem["category"],
                    "value": mem["value"],
                    "confidence": 0.9,
                    "importance": 0.9,
                    "source": "user",
                    "created_at": datetime.now().isoformat(),
                    "last_accessed": datetime.now().isoformat(),
                    "session_id": active_session_id
                }
            )

    except Exception as e:
        logging.warning(f"Profile extraction failed: {e}")

    # Get response with Enhanced RAG (RAPTOR + fallback)
    resp = await run_enhanced_rag(
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
        "data": {
            "response": resp,
            "session_id": active_session_id
        },
        "error": None
    }


@router.post("/memory")
async def memory(req: MemoryRequest, current_user: str = Depends(get_current_user)):
    """Store a memory/document into vector DB."""
    # Only include session_id in metadata if it's provided
    metadata = {}
    if req.session_id:
        metadata["session_id"] = req.session_id
    
    res = await store_memory_with_raptor(req.text, metadata=metadata)
    return {"status": "success", "data": res, "error": None}


@router.get("/session/current")
async def get_current_session():
    """Get current active session."""
    current = session_manager.get_current_session()
    return {
        "status": "success", 
        "data": {"session_id": current},
        "error": None
    }

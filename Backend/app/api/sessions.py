"""
Session management endpoints for time-based automatic session management.
"""
from fastapi import APIRouter
from typing import Optional
from app.services.short_term_memory import session_manager

router = APIRouter()


@router.get("/session/current")
async def get_current_session():
    """Get the currently active session (auto-managed)."""
    current = session_manager.get_current_session()
    if current:
        info = session_manager.get_session_info(current)
        return {
            "status": "success",
            "session_id": current,
            "session_info": info
        }
    return {
        "status": "success",
        "session_id": None,
        "message": "No active session. Start chatting to create one."
    }


@router.post("/session/new")
async def force_new_session():
    """Force end current session (summarizes & stores it) and start a new one."""
    new_session_id = await session_manager.force_summarize_current()
    return {
        "status": "success",
        "message": "Previous session summarized and stored. New session started.",
        "new_session_id": new_session_id
    }


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get all conversation history for a session from short-term memory."""
    history = session_manager.get_session_history(session_id)
    info = session_manager.get_session_info(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "session_info": info,
        "message_count": len(history),
        "history": history
    }


@router.get("/session/{session_id}/context")
async def get_context_window(session_id: str):
    """Get the current context window (messages within limits) for a session."""
    context = session_manager.get_context_window(session_id)
    stats = session_manager.get_context_stats(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "context": context,
        "stats": stats
    }


@router.get("/session/{session_id}/context/stats")
async def get_context_stats(session_id: str):
    """Get context window statistics for a session (like LLM context usage)."""
    stats = session_manager.get_context_stats(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "context_stats": stats
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear all short-term memory for a session."""
    session_manager.clear_session(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Session short-term memory cleared"
    }


@router.get("/sessions")
async def list_sessions():
    """List all active sessions with short-term memory."""
    sessions = session_manager.get_all_sessions()
    current = session_manager.get_current_session()
    return {
        "status": "success",
        "session_count": len(sessions),
        "current_session_id": current,
        "sessions": sessions
    }


@router.get("/session/{session_id}/search")
async def search_session(session_id: str, query: str):
    """Search within a specific session's history."""
    results = session_manager.search_session(session_id, query)
    return {
        "status": "success",
        "session_id": session_id,
        "query": query,
        "match_count": len(results),
        "results": results
    }


@router.post("/context/config")
async def set_context_config(max_messages: Optional[int] = None, max_chars: Optional[int] = None):
    """
    Dynamically update context window limits (like adjusting LLM context window).
    
    Args:
        max_messages: Maximum messages to retain (default: 20)
        max_chars: Maximum characters in context (default: 4000)
    """
    session_manager.set_context_limits(max_messages, max_chars)
    return {
        "status": "success",
        "message": "Context window limits updated",
        "new_limits": {
            "max_messages": session_manager.max_messages,
            "max_context_chars": session_manager.max_context_chars
        }
    }


@router.get("/context/config")
async def get_context_config():
    """Get current context window configuration."""
    current = session_manager.get_current_session()
    return {
        "status": "success",
        "context_config": {
            "max_messages": session_manager.max_messages,
            "max_context_chars": session_manager.max_context_chars,
            "session_timeout_minutes": 60,
            "current_session_id": current,
            "description": "Auto-managed sessions: expire after 60min inactivity, get summarized & stored in vector DB"
        }
    }


@router.post("/cleanup/expired")
async def cleanup_expired_sessions():
    """Manually trigger cleanup of expired sessions (summarizes and stores them)."""
    await session_manager.cleanup_all_expired()
    return {
        "status": "success",
        "message": "Expired sessions cleaned up and summarized to vector DB"
    }

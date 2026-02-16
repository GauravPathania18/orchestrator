from fastapi import APIRouter
from app.schemas.chat import ChatRequest, MemoryRequest
from app.services.rag_pipeline import run_rag, store_memory

router = APIRouter()


@router.post("/chat")
async def chat(req: ChatRequest):
    """Orchestrator chat endpoint: forwards the query to the RAG pipeline."""
    resp = await run_rag(req.message, session_id=req.session_id, top_k=req.top_k or 5)
    return {"status": "success", "data": resp}


@router.post("/memory")
async def memory(req: MemoryRequest):
    """Store a memory/document into the vector DB tied to an optional session_id."""
    res = await store_memory(req.text, metadata={"session_id": req.session_id})
    return {"status": "success", "data": res}

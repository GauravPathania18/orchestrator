from app.services.vector_client import query_text, add_text, add_vector, query_vector
from app.services.embedding_client import get_embedding
from app.services.rag_pipeline import __name__
import asyncio


def _extract_top_docs(vector_response: dict, top_k: int = 5) -> list:
    """Normalize vector service response into a list of top documents with scores."""
    # vector_response expected shape: {"status":"success","results": {...chroma query result...}}
    payload = vector_response.get("results") if isinstance(vector_response, dict) and "results" in vector_response else vector_response
    if not payload:
        return []

    # Chroma query returns lists-of-lists for single query: ids=[[...]], documents=[[...]], metadatas=[[...]], distances=[[...]]
    ids = payload.get("ids", [[]])[0] if isinstance(payload.get("ids"), list) and payload.get("ids") else []
    docs = payload.get("documents", [[]])[0] if isinstance(payload.get("documents"), list) and payload.get("documents") else []
    metadatas = payload.get("metadatas", [[]])[0] if isinstance(payload.get("metadatas"), list) and payload.get("metadatas") else []
    distances = payload.get("distances", [[]])[0] if isinstance(payload.get("distances"), list) and payload.get("distances") else []

    results = []
    for i in range(min(len(ids), top_k)):
        results.append({
            "id": ids[i],
            "document": docs[i] if i < len(docs) else None,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "score": distances[i] if i < len(distances) else None,
        })
    return results


async def _compose_answer(query: str, retrieved: list) -> str:
    """Simple LLM composition stub: synthesizes a short answer from retrieved docs.

    Replace this with real LLM calls later.
    """
    if not retrieved:
        return f"I don't have relevant memories for: {query}"
    # create a short summary by concatenating top 3 docs
    snippets = [r.get("document") or r.get("metadata", {}).get("text") for r in retrieved[:3]]
    snippets = [s for s in snippets if s]
    summary = " \n---\n ".join(snippets)
    return f"Relevant memories:\n{summary}\n\nAnswer (stub): I found these memories related to your question." 


async def run_rag(message: str, session_id: str | None = None, top_k: int = 5) -> dict:
    """Run a simple RAG flow: query the vector DB and return results.

    The vector service currently accepts a text `query` and performs embedding
    on its side; we forward the user message to it and return results.
    """
    # 1) compute embedding in orchestrator
    embedding = await get_embedding(message)

    # 2) store embedding in vector DB as an external vector (preferred)
    try:
        store_resp = await add_vector(embedding, metadata={"text": message, "source": "orchestrator", "session_id": session_id})
    except Exception:
        # fallback: send raw text to vector service which will embed itself
        try:
            store_resp = await add_text(message, metadata={"source": "orchestrator", "session_id": session_id})
        except Exception:
            store_resp = None

    # 3) query by vector to retrieve relevant context
    try:
        raw_results = await query_vector(embedding, top_k=top_k)
    except Exception:
        # fallback: query by text if vector query fails
        raw_results = await query_text(message, top_k=top_k)

    # 4) normalize top documents
    top_docs = _extract_top_docs(raw_results, top_k=top_k)

    # 5) optional LLM composition (stub)
    answer = await _compose_answer(message, top_docs)

    return {
        "query": message,
        "session_id": session_id,
        "embedding_dim": len(embedding) if embedding else None,
        "stored": store_resp,
        "retrieved": top_docs,
        "answer": answer,
    }


async def store_memory(text: str, metadata: dict | None = None) -> dict:
    """Store a memory/document into the vector service.

    We currently forward the raw text and metadata to the vector service's
    `/add_text` endpoint (which performs its own embedding). If you prefer
    the orchestrator to compute embeddings and send vectors, we can extend
    the vector service API to accept external vectors.
    """
    return await add_text(text, metadata)

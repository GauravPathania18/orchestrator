from app.services.vector_client import query_text, add_text, add_vector, query_vector, semantic_search
from app.services.embedding_client import get_embedding
from app.services.short_term_memory import session_manager
from app.services.ollama_client import generate_response, create_rag_prompt


def _extract_top_docs(vector_response: dict, top_k: int = 3) -> list:
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


def _extract_semantic_results(semantic_response: dict, top_k: int = 5) -> list:
    """Extract results from semantic search response with similarity scores."""
    if not isinstance(semantic_response, dict):
        return []
    
    results = semantic_response.get("results", [])
    if not results:
        return []
    
    formatted = []
    for item in results[:top_k]:
        formatted.append({
            "id": item.get("id"),
            "document": item.get("text"),
            "metadata": item.get("metadata", {}),
            "distance": item.get("distance"),
            "similarity_score": item.get("similarity_score"),
        })
    return formatted


def _search_short_term_memory(session_id: str, query: str, top_k: int = 5) -> list:
    """Search short-term memory (session history) for matches."""
    if not session_id:
        return []
    
    # Get session history
    history = session_manager.get_session_history(session_id)
    if not history:
        return []
    
    # Simple text search within the session
    query_lower = query.lower()
    matches = []
    
    for msg in history:
        message_text = msg.get("message", "").lower()
        # Check if query is contained in the message
        if query_lower in message_text:
            matches.append({
                "id": f"stm_{msg.get('timestamp', 'unknown')}",
                "document": msg.get("message"),
                "metadata": {
                    "role": msg.get("role"),
                    "timestamp": msg.get("timestamp"),
                    "source": "short_term_memory"
                },
                "distance": 0.0,  # Exact match has 0 distance
                "similarity_score": 100.0,  # Exact text match = 100%
            })
    
    return matches[:top_k]


async def _compose_answer(query: str, retrieved: list) -> str:
    """
    Generate an answer using Ollama LLM with retrieved context.
    
    Args:
        query: User query
        retrieved: List of retrieved documents
        
    Returns:
        Generated answer from Ollama
    """
    if not retrieved:
        return f"I don't have relevant memories for: {query}"
    
    # Create RAG prompt with context
    prompt = create_rag_prompt(query, retrieved)
    
    try:
        # Generate response from Ollama
        answer = await generate_response(
            prompt=prompt,
            temperature=0.7
        )
        return answer
    except ConnectionError as e:
        # Fallback if Ollama is not available
        print(f"[RAG] Ollama not available: {e}")
        snippets = [r.get("document") or r.get("metadata", {}).get("text") for r in retrieved[:3]]
        snippets = [s for s in snippets if s]
        summary = " \n---\n ".join(snippets)
        return f"Relevant memories:\n{summary}\n\n[Ollama not available - showing raw context]"
    except Exception as e:
        print(f"[RAG] Error generating answer: {e}")
        return f"Error generating answer: {e}"


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
    "retrieval": {
        "top_k": top_k,
        "results": top_docs,
    },
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


async def run_rag_with_scores(message: str, session_id: str | None = None, top_k: int = 5) -> dict:
    """Run a RAG flow with semantic similarity scores.

    Searches both short-term memory (session history) and long-term memory (vector DB).
    """
    all_results = []
    
    # 1) Search short-term memory (session history) if session_id provided
    if session_id:
        stm_results = _search_short_term_memory(session_id, message, top_k=top_k)
        all_results.extend(stm_results)
    
    # 2) Search long-term memory (vector DB) via semantic search
    raw_results = await semantic_search(message, top_k=top_k)
    long_term_results = _extract_semantic_results(raw_results, top_k=top_k)
    all_results.extend(long_term_results)
    
    # 3) Sort by similarity score (descending) - short-term matches (100%) will be first
    all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    
    # 4) Limit to top_k total results
    top_docs = all_results[:top_k]

    # 5) compose answer
    answer = await _compose_answer(message, top_docs)
    
    return {
        "query": message,
        "session_id": session_id,
        "retrieval": {
            "top_k": top_k,
            "results": top_docs,
            "sources": {
                "short_term": len([r for r in top_docs if r.get("metadata", {}).get("source") == "short_term_memory"]),
                "long_term": len([r for r in top_docs if r.get("metadata", {}).get("source") != "short_term_memory"])
            }
        },
        "answer": answer,
    }

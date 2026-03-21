import logging
from typing import Dict, Any, List
from datetime import datetime
import uuid
import numpy as np

from .raptor_client import raptor_client
from .short_term_memory import session_manager
from .ollama_client import generate_response
from app.services.embedding_client import get_embedding

from app.services.memory.intent_classifier import classify_intent
from app.services.memory.memory_scorer import compute_memory_score
from app.services.memory.memory_selector import select_top_memories
from app.services.memory.context_builder import build_structured_context
from app.services.memory.memory_reinforcement import reinforce_memory
from app.services.memory.memory_decay import apply_decay


# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------
# INTENT FILTERING
# -----------------------------
def filter_memories_by_intent(memories, intent):

    if intent == "memory_read":
        return [
            m for m in memories
            if m.get("metadata", {}).get("type") == "profile"
            or m.get("metadata", {}).get("source") == "short_term"
        ]

    elif intent == "memory_write":
        return [
            m for m in memories
            if m.get("metadata", {}).get("type") == "profile"
        ]

    elif intent == "knowledge":
        return [
            m for m in memories
            if m.get("metadata", {}).get("type") == "knowledge"
            or m.get("metadata", {}).get("source") == "short_term"
        ]

    return memories


# -----------------------------
# MAIN PIPELINE
# -----------------------------
async def run_enhanced_rag(
    message: str,
    session_id: str | None = None,
    top_k: int = 5,
    use_raptor: bool = True
) -> dict:

    logging.info(f"🎯 Running enhanced RAG for: '{message[:50]}...'")

    # 1. INTENT
    intent = classify_intent(message)

    # 2. QUERY EMBEDDING
    try:
        query_embedding = await get_embedding(message)
    except Exception:
        query_embedding = None

    # 3. SHORT TERM MEMORY
    stm_docs = []

    if session_id:
        history = session_manager.get_session_history(session_id)

        if history:
            for msg in history[-5:]:
                text = f"{msg.get('role')}: {msg.get('message')}"

                try:
                    emb = await get_embedding(text)
                except Exception:
                    emb = None

                sim = 0.7
                if emb and query_embedding:
                    sim = cosine_similarity(query_embedding, emb)

                stm_docs.append({
                    "document": text,
                    "metadata": {
                        "id": str(uuid.uuid4()),
                        "type": "conversation",
                        "category": "recent",
                        "value": text,
                        "confidence": 1.0,
                        "importance": 0.7,
                        "source": "short_term",
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat()
                    },
                    "similarity": sim,
                    "embedding": emb
                })

    # 4. RAPTOR RETRIEVAL
    raptor_docs = []
    retrieval_info = {}

    if use_raptor:
        try:
            raptor_result = await raptor_client.raptor_query(
                query=message,
                k_summary=2,
                k_chunks=8,
                top_k_final=top_k
            )

            if raptor_result.get("status") == "success":
                data = raptor_result["data"]

                chunks = data.get("chunks", [])
                summaries = data.get("summaries", [])

                for doc in summaries + chunks:
                    try:
                        emb = await get_embedding(doc)
                    except Exception:
                        emb = None

                    sim = 0.5
                    if emb and query_embedding:
                        sim = cosine_similarity(query_embedding, emb)

                    raptor_docs.append({
                        "document": doc,
                        "metadata": {
                            "id": str(uuid.uuid4()),
                            "type": "knowledge",
                            "category": "general",
                            "value": doc,
                            "confidence": 0.8,
                            "importance": 0.8,
                            "source": "raptor",
                            "created_at": datetime.now().isoformat(),
                            "last_accessed": datetime.now().isoformat()
                        },
                        "similarity": sim,
                        "embedding": emb
                    })

                retrieval_info = {
                    "type": "raptor",
                    "num_retrieved": len(raptor_docs)
                }

            else:
                use_raptor = False

        except Exception as e:
            logging.error(f"RAPTOR error: {e}")
            use_raptor = False

    # 5. FALLBACK
    if not use_raptor:
        from .rag_pipeline import run_rag_with_scores

        simple_result = await run_rag_with_scores(message, session_id, top_k)

        docs = simple_result.get("retrieval", {}).get("results", [])

        for doc in docs:
            raptor_docs.append({
                "document": doc.get("document"),
                "metadata": doc.get("metadata", {}),
                "similarity": doc.get("similarity_score", 0.7),
                "embedding": None
            })

        retrieval_info = {
            "type": "fallback",
            "num_retrieved": len(raptor_docs)
        }

    # 6. MERGE + FILTER
    all_memories = stm_docs + raptor_docs
    filtered_memories = filter_memories_by_intent(all_memories, intent)

    if not filtered_memories:
        filtered_memories = all_memories

    # 7. DECAY
    for m in filtered_memories:
        m["metadata"] = apply_decay(m.get("metadata", {}))

    # 8. SCORING
    for m in filtered_memories:
        meta = m.get("metadata", {})
        sim = m.get("similarity", 0.5)

        score = compute_memory_score(meta, sim)

        if meta.get("type") == "profile":
            score *= 1.5

        if meta.get("source") == "short_term":
            score *= 1.2

        score *= sim
        m["score"] = score

    # 9. SELECT
    selected = select_top_memories(filtered_memories, top_k=top_k)

    # 10. REINFORCE
    for m in selected:
        m["metadata"] = reinforce_memory(m.get("metadata", {}))

    # 11. CONTEXT
    user_profile, short_term, knowledge = [], [], []

    for m in selected:
        meta = m.get("metadata", {})
        doc = m.get("document", "")

        if meta.get("type") == "profile":
            user_profile.append(doc)
        elif meta.get("source") == "short_term":
            short_term.append(doc)
        else:
            knowledge.append(doc)

    final_context = build_structured_context(
        user_profile=user_profile,
        short_term=short_term,
        knowledge=knowledge,
        query=message
    )

    # 12. LLM
    prompt = f"""
You are an intelligent personal AI assistant.

Use the structured context below:

{final_context}
"""

    try:
        answer = await generate_response(prompt=prompt, temperature=0.7)
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        answer = "Error generating response."

    return {
        "query": message,
        "session_id": session_id,
        "answer": answer,
        "intent": intent,
        "retrieval": retrieval_info,
        "selected_memories": len(selected),
        "enhanced_rag": use_raptor
    }


# -----------------------------
# RAPTOR INGESTION (🔥 INCLUDED)
# -----------------------------
async def store_memory_with_raptor(text: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}

    try:
        result = await raptor_client.ingest_documents([text])

        if result.get("status") == "success":
            data = result["data"]
            logging.info(f"🌲 RAPTOR ingestion: {data}")
            return {
                "status": "success",
                "type": "raptor",
                "data": data
            }
        else:
            raise Exception("RAPTOR ingestion failed")

    except Exception as e:
        logging.warning(f"RAPTOR ingestion failed: {e}, using fallback")
        from .rag_pipeline import store_memory
        return await store_memory(text, metadata)


# -----------------------------
# RAPTOR STATS
# -----------------------------
async def get_raptor_stats() -> dict:
    try:
        return await raptor_client.get_pipeline_stats()
    except Exception as e:
        logging.error(f"Failed to get RAPTOR stats: {e}")
        return {"error": str(e)}
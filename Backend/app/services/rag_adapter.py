from app.services.raptor_client import raptor_client
from app.services.vector_client import retrieve as vector_retrieve
from app.services.enhanced_rag import run_enhanced_rag
from app.services.ollama_client import generate_response


# -----------------------------
# RAPTOR RETRIEVAL
# -----------------------------
async def raptor_retrieve(query: str):
    return await raptor_client.retrieve(query)


# -----------------------------
# BASELINE RETRIEVAL
# -----------------------------
async def baseline_retrieve(query: str):
    return await vector_retrieve(query)


# -----------------------------
# GENERATION
# -----------------------------
async def generate_answer(question: str, context: str):

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    return await generate_response(prompt=prompt)


# -----------------------------
# OPTIONAL: FULL PIPELINE (if needed)
# -----------------------------
async def full_pipeline(query: str):
    return await run_enhanced_rag(query)
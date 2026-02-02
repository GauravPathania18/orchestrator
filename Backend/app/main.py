from fastapi import FastAPI
from app.api.chat import router as chat_router

app = FastAPI(title="RAG Orchestrator")

app.include_router(chat_router)

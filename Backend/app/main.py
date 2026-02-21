from fastapi import FastAPI
from app.api.simple import router as simple_router

app = FastAPI(
    title="RAG Chat API",
    description="Simplified RAG chat with automatic session management"
)

app.include_router(simple_router, prefix="/api")

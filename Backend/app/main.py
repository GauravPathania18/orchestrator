from fastapi import FastAPI
from app.api.simple import router as simple_router
from app.api.auth import router as auth_router
from app.api.raptor import router as raptor_router
from app.api.sessions import router as sessions_router

app = FastAPI(
    title="RAG Chat API",
    description="Simplified RAG chat with automatic session management"
)

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(simple_router, prefix="/api")
app.include_router(raptor_router, prefix="/api/raptor", tags=["raptor"])
app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])

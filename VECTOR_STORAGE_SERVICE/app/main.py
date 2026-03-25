import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .routes import vectors, raptor
from .config import OLLAMA_URL, DEFAULT_MODEL

# Required models for metadata and processing
REQUIRED_MODELS = [DEFAULT_MODEL]

async def verify_ollama_models():
    """Check if required models are present in Ollama, pull if missing."""
    try:
        async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=30.0) as client:
            # Check tags
            resp = await client.get("/api/tags")
            if resp.status_code != 200:
                logging.warning(f"⚠️ Could not connect to Ollama at {OLLAMA_URL} to verify models.")
                return

            models_data = resp.json().get("models", [])
            installed_models = [m.get("name") for m in models_data]
            
            for model in REQUIRED_MODELS:
                if model not in installed_models:
                    logging.info(f"📥 Pulling required model: {model}...")
                    # Non-blocking pull (we don't wait for the whole pull to finish here to avoid blocking startup too long, 
                    # but in production you might want to wait or handle it as a background task)
                    await client.post("/api/pull", json={"name": model}, timeout=None)
                else:
                    logging.info(f"✅ Model {model} is already installed.")
                    
    except Exception as e:
        logging.error(f"❌ Error verifying Ollama models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    await verify_ollama_models()
    yield
    # Shutdown (if needed)

app = FastAPI(
    title="Vector Store Service",
    version="1.2",
    description="A microservice to store, retrieve, and search vectors in ChromaDB",
    lifespan=lifespan
)

# Register routers
app.include_router(vectors.router, prefix="/vectors", tags=["vectors"])
app.include_router(raptor.router, prefix="/raptor", tags=["raptor"])

@app.get("/")
def root():
    return {"message": "Vector Storage Service is running 🚀"}

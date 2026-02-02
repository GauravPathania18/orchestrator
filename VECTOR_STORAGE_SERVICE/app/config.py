import os

# URL of the embedding service (can be overridden via env var)
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://127.0.0.1:8000/embed")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_llm")

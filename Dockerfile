# Multi-stage Dockerfile for Personal LLM RAG System
# This builds all services: Backend, Vector Storage, and Embedder

FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy consolidated requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 1: Embedder Service
FROM base AS embedder
WORKDIR /app/embedder
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY personal_LLM_embedder/embedder_api.py ./
COPY personal_LLM_embedder/requirements.txt ./
ENV PYTHONPATH=/app/embedder
EXPOSE 8000
CMD ["uvicorn", "embedder_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 2: Vector Storage Service  
FROM base AS vector-storage
WORKDIR /app/vector-storage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY VECTOR_STORAGE_SERVICE/app ./app
COPY VECTOR_STORAGE_SERVICE/requirements.txt ./
COPY VECTOR_STORAGE_SERVICE/run.py ./
ENV PYTHONPATH=/app/vector-storage
ENV PERSIST_DIR=./chroma_store
ENV COLLECTION_NAME=personal_llm
ENV OLLAMA_URL=http://ollama:11434
EXPOSE 8001
CMD ["python", "run.py"]

# Stage 3: Backend Service
FROM base AS backend
WORKDIR /app/backend
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY Backend/app ./app
COPY Backend/requirements.txt ./
ENV PYTHONPATH=/app/backend
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

# Final stage (for building individual services)
FROM embedder       


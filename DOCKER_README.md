# Personal LLM RAG System - Docker Deployment

## Quick Start

```bash
# Build and start all services
docker-compose up --build

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Start specific service
docker-compose up -d embedder
```

## Services

- **Embedder** (Port 8000): Text embedding service with caching
- **Vector Storage** (Port 8001): ChromaDB + RAPTOR hierarchical clustering
- **Backend** (Port 8002): Main API with enhanced RAG
- **Ollama** (Port 11434): LLM inference
- **Nginx** (Port 80): Reverse proxy

## API Endpoints

### Direct Service Access
- Embedder: `http://localhost:8000/embed`
- Vector Storage: `http://localhost:8001/raptor/*` and `http://localhost:8001/vectors/*`
- Backend: `http://localhost:8002/api/*`

### Via Nginx Proxy
- Embeddings: `http://localhost/embed`
- Vectors: `http://localhost/vectors`
- Backend API: `http://localhost/api`

### RAPTOR-Specific Endpoints
- `POST /raptor/ingest` - Document clustering and summarization
- `POST /raptor/query` - Hierarchical retrieval with reranking
- `GET /raptor/stats` - Pipeline statistics and configuration
- `POST /raptor/reset` - Clear RAPTOR data

## Environment Variables

Key environment variables configured in docker-compose:

- `OLLAMA_URL`: Ollama service endpoint
- `EMBEDDER_URL`: Embedder service endpoint
- `VECTOR_STORE_URL`: Vector storage endpoint
- `PERSIST_DIR`: ChromaDB storage location
- `COLLECTION_NAME`: Vector collection name

## Volumes

- `ollama_data`: Persistent Ollama models
- `chroma_data`: Persistent vector database

## Development

For development with hot reload:
```bash
docker-compose up --build --watch
```

## Production

For production deployment:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```



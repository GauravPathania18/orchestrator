# Vector Storage Service

A FastAPI-based microservice that stores, retrieves, and searches vector embeddings using ChromaDB. This service is the persistence layer for semantic search in your Personal LLM system.

## Overview

The Vector Storage Service manages a vector database (ChromaDB) that stores embeddings and their metadata. It supports:
- Storing embeddings with metadata (session_id, source, timestamps, etc.)
- Semantic search using vector similarity
- Query by text (delegates embedding to Embedder Service)
- Query by pre-computed vectors (from Backend)

### Data Flow

```
Embedding Service → Vector (768-dim)
                         ↓
Backend Orchestrator → Store in ChromaDB
                         ↓
Backend Queries → Similarity Search → Top-K Results
```

## Features

✅ **ChromaDB Backend**: Persistent vector storage with metadata filtering  
✅ **Semantic Search**: Find similar texts using vector similarity  
✅ **Dual Query Modes**: Query by text OR by pre-computed vector  
✅ **Metadata Filtering**: Support for session_id, domain, entity_type, confidence scores  
✅ **Batch Operations**: Add/query multiple vectors efficiently  
✅ **Swagger UI**: Interactive API documentation at `/docs`

## Setup

### 1. Prerequisites

- Python 3.10+
- ChromaDB (automatically installed via requirements.txt)
- Embedder Service running (http://127.0.0.1:8000)

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

Key dependencies:
- `fastapi>=0.95.0` — Web framework
- `uvicorn[standard]>=0.22.0` — ASGI server
- `chromadb>=0.3.21` — Vector database
- `pydantic>=1.10.0` — Request/response validation
- `requests` — HTTP client for Embedder Service
- `beautifulsoup4` — Text cleaning

### 3. Environment Variables (Optional)

Create a `.env` file in this folder:

```env
# Embedding Service Configuration
EMBEDDER_URL=http://127.0.0.1:8000/embed

# ChromaDB Configuration
PERSIST_DIR=./chroma_store
COLLECTION_NAME=personal_llm

# Optional: LLM Service (for future use)
OLLAMA_URL=http://localhost:11434
```

**Default Values**:
- `EMBEDDER_URL`: http://127.0.0.1:8000/embed
- `PERSIST_DIR`: ./chroma_store
- `COLLECTION_NAME`: personal_llm

## Running the Service

### Start the Vector Storage Service

```powershell
python run.py
```

Or use uvicorn directly:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8003 --reload
```

**Output** (when successful):
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8003 (Press CTRL+C to quit)
```

### Access Interactive Docs

Open your browser:
- **Swagger UI** (recommended): http://127.0.0.1:8003/docs
- **ReDoc**: http://127.0.0.1:8003/redoc

## API Endpoints

### 1. `GET /` — Health Check

**Purpose**: Verify the service is running.

**Response**:
```json
{
  "message": "Vector Storage Service is running 🚀"
}
```

### 2. `POST /add_text` — Store Text with Auto-Embedding

**Purpose**: Submit raw text; service calls Embedder Service to compute embedding and stores it.

**Request**:
```json
{
  "text": "I love hiking in the mountains",
  "metadata": {
    "session_id": "user_session_123",
    "source": "user_input"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "I love hiking in the mountains",
  "metadata_status": "pending"
}
```

**Parameters**:
- `text` (string, required): Text to embed and store
- `metadata` (dict, optional): Additional metadata (session_id, source, etc.)

### 3. `POST /add_vector` — Store Pre-Computed Vector

**Purpose**: Store an externally computed embedding (from Backend) with metadata.

**Request**:
```json
{
  "vector": [0.123, -0.456, ..., 0.789],
  "metadata": {
    "text": "Sample text",
    "session_id": "user_session_123",
    "source": "orchestrator"
  }
}
```

**Response**:
```json
{
  "status": "success",
  "id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Parameters**:
- `vector` (list[float], required): Pre-computed embedding vector
- `metadata` (dict, optional): Metadata to associate with vector

### 4. `POST /query_text` — Search by Text

**Purpose**: Submit a query text; service embeds it and searches for similar vectors.

**Request**:
```json
{
  "query": "What did I say about hiking?",
  "top_k": 5
}
```

**Response**:
```json
{
  "status": "success",
  "query": "What did I say about hiking?",
  "results": {
    "ids": [["id1", "id2", "id3"]],
    "documents": [["I love hiking in the mountains", "Hiking is my favorite...", "]],
    "metadatas": [[{"session_id": "user_session_123", "source": "user_input"}, ...]],
    "distances": [[0.05, 0.15, 0.22]],
    "embeddings": null,
    "uris": null,
    "included": ["metadatas", "documents", "distances"]
  }
}
```

**Parameters**:
- `query` (string, required): Query text
- `top_k` (int, optional, default=5): Number of results to return

### 5. `POST /query_vector` — Search by Vector

**Purpose**: Submit a pre-computed query vector and retrieve similar stored vectors.

**Request**:
```json
{
  "vector": [0.123, -0.456, ..., 0.789],
  "top_k": 5
}
```

**Response**:
```json
{
  "status": "success",
  "query": null,
  "results": {
    "ids": [["id1", "id2", "id3"]],
    "documents": [["I love hiking...", "Hiking is...", ...]],
    "metadatas": [[{"session_id": "...", "source": "..."}, ...]],
    "distances": [[0.05, 0.15, 0.22]]
  }
}
```

**Parameters**:
- `vector` (list[float], required): Query vector
- `top_k` (int, optional, default=5): Number of results

### 6. `GET /vectors` — List All Stored Vectors

**Purpose**: Retrieve metadata about all stored vectors (useful for debugging).

**Response**:
```json
{
  "status": "success",
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "document": "I love hiking in the mountains",
      "metadata": {
        "session_id": "user_session_123",
        "source": "user_input",
        "timestamp": "2026-02-02T12:34:56Z"
      }
    }
  ]
}
```

## Usage Examples

### Store a Memory

```bash
curl -X POST http://127.0.0.1:8003/add_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My favorite food is pizza",
    "metadata": {
      "session_id": "sess1",
      "source": "user_input"
    }
  }'
```

### Query by Text

```bash
curl -X POST http://127.0.0.1:8003/query_text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is my favorite food?",
    "top_k": 3
  }'
```

### Query by Vector

```python
import httpx

async def search_by_vector():
    vector = [0.123, -0.456, ...]  # Pre-computed embedding
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:8003/query_vector",
            json={"vector": vector, "top_k": 5},
            timeout=30.0
        )
        data = resp.json()
        return data["results"]
```

### List All Memories

```bash
curl http://127.0.0.1:8003/vectors
```

## Project Structure

```
VECTOR_STORAGE_SERVICE/
├── app/
│   ├── main.py               # FastAPI app entry point
│   ├── models.py             # Pydantic request/response models
│   ├── config.py             # Configuration & environment vars
│   ├── routes/
│   │   └── vectors.py        # Vector endpoints
│   └── services/
│       ├── embedder.py       # Client to Embedder Service
│       ├── vector_store.py   # ChromaDB wrapper
│       ├── metadata.py       # Metadata management
│       └── utils.py          # Text cleaning utilities
├── chroma_store/             # Persistent ChromaDB storage
├── run.py                    # Entry point (python run.py)
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Data Storage

### ChromaDB Persistence

Vectors are stored in `./chroma_store/` (persistent):

```
chroma_store/
├── chroma.sqlite3           # Metadata DB
└── <collection_uuid>/       # Data directory
```

To **clear all data**, delete the `chroma_store/` folder and restart.

### Metadata Storage

Each stored vector includes metadata:

```json
{
  "session_id": "user_session_123",
  "source": "user_input",
  "timestamp": "2026-02-02T12:34:56Z",
  "domain": "general",
  "entity_type": "memory",
  "confidence": 0.95
}
```

## Performance

### Benchmarks (on CPU, Intel i7)

| Operation | Count | Time | Throughput |
|-----------|-------|------|-----------|
| Add texts | 100 | 2.5s | 40 texts/sec |
| Query search | 1 query | 0.05s | ~20 queries/sec |
| List all | 1000 vectors | 0.3s | — |

## Architecture

### Integration with Personal LLM

```
┌──────────────────────────────────┐
│    Backend (Orchestrator)        │
│    http://127.0.0.1:8001        │
└────────┬───────────────┬────────┘
         │               │
         ▼               ▼
┌──────────────────┐  ┌──────────────────────┐
│  Embedder Svc    │  │ Vector Storage Svc   │
│  (port 8000)     │  │  (port 8003)         │
│                  │  │                      │
│  POST /embed     │  │  POST /add_text      │
│  GET /           │  │  POST /add_vector    │
└──────────────────┘  │  POST /query_text    │
                      │  POST /query_vector  │
                      │  GET /vectors        │
                      └──────────────────────┘
```

## Troubleshooting

### Error: `Connection refused (Embedder)`

Embedder Service not running. Start it:
```powershell
cd ../personal_LLM_embedder
python embedder_api.py
```

### Error: `No module named chromadb`

```powershell
pip install chromadb
```

### Error: `Vector length mismatch`

All vectors in the collection must have the same dimension (768 for all-mpnet-base-v2). Ensure Embedder Service uses the same model.

### Error: `Database locked`

ChromaDB database is locked. Restart the service.

### Slow Queries

- Try reducing `top_k` parameter
- Limit metadata filters
- Consider using GPU Embedder for faster embedding computation

## Future Enhancements

- [ ] **Batch Add**: Optimize bulk insertion
- [ ] **Similarity Threshold**: Filter results by minimum score
- [ ] **Session History**: List all memories for a session_id
- [ ] **Vector Compression**: Store compressed vectors for space efficiency

## Development Tips

- **Watch logs**: Monitor `uvicorn` terminal for errors
- **Test endpoints**: Use Swagger UI at `/docs`
- **Inspect storage**: Use `GET /vectors` to see stored data
- **Debug queries**: Add print statements in `vector_store.py`

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for your Personal LLM Project**

For questions or updates, see the main project repository.


Response:

{
  "status": "success",
  "id": "uuid",
  "message": "Vector stored successfully."
}

GET /vectors – Fetch stored vectors

Optional Query Param: limit

Response:

{
  "status": "success",
  "data": {
    "ids": ["uuid1", "uuid2"],
    "embeddings": [[...], [...]],
    "metadatas": [{"doc": "my note"}, {"doc": "another"}]
  }
}

POST /search – Search vectors by similarity

Request:

{
  "vector": [0.1, 0.9, -0.3],
  "top_k": 3
}


Response:

{
  "status": "success",
  "results": {
    "ids": [["uuid1", "uuid2"]],
    "distances": [[0.12, 0.34]],
    "metadatas": [[{"doc": "closest match"}, {"doc": "second match"}]]
  }
}

⚙️ Environment Variables

You can configure storage:

export PERSIST_DIR=./chroma_store
export COLLECTION_NAME=personal_llm

📖 About

A lightweight vector storage API for AI pipelines.
Pairs well with the Personal LLM Embedding Service.

📜 License

MIT License

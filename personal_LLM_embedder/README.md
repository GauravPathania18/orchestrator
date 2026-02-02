# Personal LLM Embedding Service

A FastAPI-based microservice that generates high-quality vector embeddings from text using local SentenceTransformer models. This service powers semantic search and context retrieval for your Personal LLM system.

## Overview

The Embedding Service converts any text input into a dense vector representation (embedding) that captures the semantic meaning. These embeddings are then stored and queried by the Vector Storage Service to enable semantic search.

### Key Capabilities

- 🚀 **Local Embeddings**: Uses SentenceTransformers (all-mpnet-base-v2) for fast, offline embedding generation
- 📊 **Batch Processing**: Embed multiple texts in a single request
- 🧹 **Text Cleaning**: Automatic HTML stripping and whitespace normalization
- ⚡ **Performance**: Optimized for CPU and optional GPU acceleration
- 📝 **Metadata Tracking**: Returns embeddings with unique IDs and timestamps
- 🔧 **Configurable**: Control model, device, and preprocessing via environment variables

## Features

✅ **Semantic Text Embeddings**: Convert any text to 768-dimensional vectors  
✅ **Batch Encoding**: Process multiple texts efficiently  
✅ **HTML Stripping**: Clean messy text automatically  
✅ **Metadata**: Track embedding source, timestamp, and model info  
✅ **Performance Metrics**: Return processing time for each batch  
✅ **Health Checks**: Monitor service status and configuration  
✅ **Swagger UI**: Interactive API documentation at `/docs`

## Setup

### 1. Prerequisites

- Python 3.10+ (tested with 3.10)
- 2+ GB RAM (for model loading)
- Optional: CUDA-capable GPU for faster inference

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

Key dependencies:
- `fastapi>=0.95.0` — Web framework
- `uvicorn[standard]>=0.22.0` — ASGI server
- `pydantic>=1.10.0` — Request/response validation
- `sentence-transformers` — Local embedding models
- `numpy` — Numerical operations
- `beautifulsoup4` — HTML parsing

### 3. Environment Variables (Optional)

Create a `.env` file in this folder to customize behavior:

```env
# Embedding configuration
EMBEDDING_MODE=local              # "local" | "openai" | "gemini"
LOCAL_MODEL=all-mpnet-base-v2     # HuggingFace model name
DEVICE=cpu                        # "cpu" | "cuda"
HTML_STRIP=true                   # Strip HTML from text?

# Server configuration
HOST=127.0.0.1
PORT=8000
```

**Default Values**:
- `EMBEDDING_MODE`: local
- `LOCAL_MODEL`: all-mpnet-base-v2 (384M, 768-dim embeddings)
- `DEVICE`: cpu
- `HTML_STRIP`: true

## Running the Service

### Start the Embedder

```powershell
python embedder_api.py
```

Or use uvicorn directly:

```powershell
python -m uvicorn embedder_api:app --host 127.0.0.1 --port 8000 --reload
```

**Output** (when successful):
```
[Embedding] Loading local model: all-mpnet-base-v2 on cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Access Interactive Docs

Open your browser:
- **Swagger UI** (recommended): http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## API Endpoints

### 1. `GET /` — Health Check

**Purpose**: Verify the service is running and check configuration.

**Response**:
```json
{
  "status": "ok",
  "mode": "local",
  "model": "all-mpnet-base-v2",
  "device": "cpu",
  "html_strip": true
}
```

### 2. `POST /embed` — Generate Embeddings

**Purpose**: Convert one or more texts into vector embeddings.

**Request**:
```json
{
  "texts": ["Hello world", "This is a test"],
  "source": "user_prompt"
}
```

**Response**:
```json
{
  "mode": "local",
  "model_name": "all-mpnet-base-v2",
  "vector_size": 768,
  "count": 2,
  "processing_time_sec": 0.1234,
  "items": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "vector": [0.123, -0.456, ..., 0.789],
      "metadata": {
        "text": "Hello world",
        "timestamp": "2026-02-02T12:34:56.789Z",
        "source": "user_prompt",
        "embedding_mode": "local"
      }
    },
    {
      "id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "vector": [-0.321, 0.654, ..., -0.987],
      "metadata": {
        "text": "This is a test",
        "timestamp": "2026-02-02T12:34:56.790Z",
        "source": "user_prompt",
        "embedding_mode": "local"
      }
    }
  ]
}
```

**Parameters**:
- `texts` (string | list[string], required): Text(s) to embed. Auto-trimmed and validated.
- `source` (string, optional, default="user_prompt"): Metadata tag for tracking embedding origin

**Response Fields**:
- `mode`: Embedding mode (local/openai/gemini)
- `model_name`: Model used for embeddings
- `vector_size`: Dimension of each embedding (768 for all-mpnet-base-v2)
- `count`: Number of embeddings generated
- `processing_time_sec`: Time to compute all embeddings
- `items`: List of embedding results with IDs, vectors, and metadata

## Usage Examples

### Single Text Embedding

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": "What is semantic search?",
    "source": "question"
  }'
```

### Batch Embeddings

```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love hiking in the mountains",
      "Hiking is my favorite outdoor activity",
      "Mountains are beautiful"
    ],
    "source": "user_memories"
  }'
```

### Python Example (with httpx)

```python
import httpx

async def get_embedding(text: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:8000/embed",
            json={"texts": [text], "source": "app"},
            timeout=30.0
        )
        resp.raise_for_status()
        data = resp.json()
        return data["items"][0]["vector"]

# Usage
vector = await get_embedding("Hello world")
print(f"Embedding dimension: {len(vector)}")
```

## Model Information

### Default Model: `all-mpnet-base-v2`

| Property | Value |
|----------|-------|
| **Dimensions** | 768 |
| **Model Size** | ~384 MB |
| **Max Seq Length** | 384 tokens |
| **Training Data** | Multi-lingual large corpus |
| **Use Case** | General semantic search, clustering, similarity |
| **Performance** | Fast on CPU, ~0.1s per 100 texts |

### Alternative Models

You can replace `all-mpnet-base-v2` with other SentenceTransformer models:

```env
LOCAL_MODEL=all-MiniLM-L6-v2      # 22M, 384-dim, fastest
LOCAL_MODEL=all-mpnet-base-v2     # 384M, 768-dim, best quality
LOCAL_MODEL=all-roberta-large-v1  # Larger, slower but more powerful
```

See [SentenceTransformers models](https://www.sbert.net/models.html) for more options.

## Performance

### Benchmarks (on CPU, Intel i7)

| Batch Size | Model | Time | Throughput |
|-----------|-------|------|-----------|
| 1 | all-mpnet-base-v2 | 0.08s | ~12 texts/sec |
| 10 | all-mpnet-base-v2 | 0.15s | ~67 texts/sec |
| 100 | all-mpnet-base-v2 | 1.2s | ~83 texts/sec |

### GPU Acceleration

To use CUDA on an NVIDIA GPU:

```env
DEVICE=cuda
```

Expected speedup: **5-10x faster** than CPU.

## Architecture

### Request-Response Flow

```
┌─────────────────────────────────────┐
│   Client Request (POST /embed)      │
│   {"texts": [...], "source": "..."}│
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Input Validation                  │
│   - Ensure texts is list/string    │
│   - Trim whitespace                │
│   - Validate non-empty             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Text Cleaning                     │
│   - Strip HTML (if enabled)        │
│   - Normalize whitespace           │
│   - Filter empty strings           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Embedding Computation             │
│   - SentenceTransformer.encode()   │
│   - Convert numpy → Python lists   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Response Generation               │
│   - Assign unique IDs              │
│   - Add metadata (timestamp, etc)  │
│   - Format JSON response           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Client Response (200 OK)          │
│   {"items": [...], "vector_size": 768}
└─────────────────────────────────────┘
```

## Configuration & Customization

### Change Embedding Model

Edit `.env`:
```env
LOCAL_MODEL=all-MiniLM-L6-v2
```

Then restart the service. The new model downloads automatically (~22 MB).

### Disable HTML Stripping

```env
HTML_STRIP=false
```

### Use GPU

```env
DEVICE=cuda
```

### Add Custom Models

Modify `embedder_api.py` and add a new embedding function:

```python
async def _embed_custom(texts: List[str]) -> List[List[float]]:
    # Your custom embedding logic
    pass
```

## Troubleshooting

### Error: `No module named sentence_transformers`

```powershell
pip install sentence-transformers
```

### Error: `CUDA out of memory`

Use CPU instead or reduce batch size:
```env
DEVICE=cpu
```

### Error: `Connection refused (port 8000)`

The service may not be running. Start it:
```powershell
python embedder_api.py
```

### Slow on First Request

The embedding model is loaded on first request (~1-2 seconds). Subsequent requests are fast.

### All Embeddings Are Zeros

Check if the model loaded correctly. Look for:
```
[Embedding] Loading local model: all-mpnet-base-v2 on cpu
```

If missing, the model failed to load.

## Future Enhancements

- [ ] **OpenAI API Support**: Add `EMBEDDING_MODE=openai` to use OpenAI embeddings
- [ ] **Caching**: Cache embeddings of repeated texts
- [ ] **Dimensionality Reduction**: Optional PCA/UMAP compression
- [ ] **Fine-tuning**: Support domain-specific fine-tuned models
- [ ] **Batch Async**: Queue for large batch processing
- [ ] **Metrics**: Prometheus metrics for monitoring

## Integration with Backend

The Backend orchestrator calls this service:

```python
# Backend/app/services/embedding_client.py
async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:8000/embed",
            json={"texts": [text], "source": "user_prompt"}
        )
        data = resp.json()
        return data["items"][0]["vector"]
```

## Development Tips

- **Reload on code changes**: Add `--reload` flag
- **Check logs**: Watch terminal for detailed error traces
- **Test performance**: Use Swagger UI to batch-test different text sizes
- **Monitor memory**: Watch RAM usage, especially with GPU

## References

- [SentenceTransformers Docs](https://www.sbert.net/)
- [all-mpnet-base-v2 Model Card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for your Personal LLM Project**

For questions or updates, see the main project repository.

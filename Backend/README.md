# Backend Service

A FastAPI-based orchestrator service that coordinates the Personal LLM RAG system. This service provides the main API for chat, memory management, and RAPTOR-enhanced retrieval.

## Overview

The Backend Service acts as the central hub, orchestrating:
- **Chat Interface**: Conversational AI with context awareness
- **Memory Management**: Store and retrieve user memories
- **Session Handling**: Automatic session lifecycle management
- **RAPTOR Integration**: Hierarchical document clustering and retrieval
- **User Profiles**: Extract and maintain user preference profiles

## Features

- **RAG Pipeline**: Retrieve-Augment-Generate with semantic search
- **Session Management**: 60-minute timeout, automatic cleanup
- **Short-term Memory**: Conversation context within sessions
- **Long-term Memory**: Persistent storage via Vector Storage Service
- **RAPTOR Support**: Hierarchical clustering for better retrieval
- **User Profiles**: Automatic profile extraction from conversations
- **Feedback Loop**: Collect user feedback for RAG evaluation

## Architecture

```
User Request → Backend Orchestrator
    ├── Session Manager (short-term memory)
    ├── Vector Client (long-term retrieval)
    ├── RAPTOR Client (hierarchical retrieval)
    ├── Ollama Client (LLM generation)
    └── Response → User
```

## API Endpoints

### Chat & Memory

#### POST /api/chat

Main chat endpoint with automatic session management.

**Request:**
```json
{
  "message": "What did I say about hiking?",
  "session_id": "sess_123",
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "response": "You mentioned that you love hiking in the mountains...",
    "context_used": [...],
    "session_id": "sess_123"
  }
}
```

#### POST /api/memory

Store a memory for later retrieval.

**Request:**
```json
{
  "text": "My favorite color is blue",
  "session_id": "sess_123"
}
```

**Response:**
```json
{
  "status": "success",
  "id": "uuid",
  "message": "Memory stored"
}
```

### RAPTOR Endpoints

#### POST /api/raptor/chat

Chat with RAPTOR-enhanced retrieval (hierarchical clustering).

**Request:**
```json
{
  "message": "Tell me about my hiking preferences",
  "session_id": "sess_123"
}
```

#### POST /api/raptor/ingest

Ingest documents using RAPTOR clustering.

**Request:**
```json
{
  "text": "Long document text here...",
  "session_id": "sess_123"
}
```

#### GET /api/raptor/stats

Get RAPTOR pipeline statistics.

**Response:**
```json
{
  "status": "success",
  "data": {
    "cluster_count": 5,
    "document_count": 42
  }
}
```

### Session Management

#### GET /api/sessions/{session_id}

Get session details and history.

#### DELETE /api/sessions/{session_id}

Delete a session and its short-term memory.

### Feedback

#### POST /api/feedback

Submit feedback on RAG responses.

**Request:**
```json
{
  "response_id": "resp_123",
  "rating": 5,
  "comment": "Very helpful!"
}
```

## Configuration

Environment variables (set in root `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_SERVICE_URL` | `http://localhost:8000` | Embedding Service endpoint |
| `VECTOR_SERVICE_URL` | `http://localhost:8001` | Vector Storage Service endpoint |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama LLM endpoint |
| `SESSION_TIMEOUT_MINUTES` | `60` | Session timeout |
| `FASTAPI_ENV` | `development` | Environment mode |
| `LOG_LEVEL` | `INFO` | Logging level |

## Running Locally

```bash
cd Backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8002
```

Service will be available at `http://localhost:8002`

## Running with Docker

```bash
docker-compose up backend
```

## Testing

```bash
# Store a memory
curl -X POST http://localhost:8002/api/memory \
  -H "Content-Type: application/json" \
  -d '{"text":"I love hiking","session_id":"test1"}'

# Chat
curl -X POST http://localhost:8002/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What do I love?","session_id":"test1"}'

# RAPTOR chat
curl -X POST http://localhost:8002/api/raptor/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me about my interests"}'
```

## API Documentation

Interactive documentation available at:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

## Project Structure

```
Backend/
├── app/
│   ├── api/                    # API route handlers
│   │   ├── auth.py            # Authentication
│   │   ├── raptor.py          # RAPTOR endpoints
│   │   ├── sessions.py        # Session management
│   │   └── simple.py          # Core chat/memory endpoints
│   ├── core/                   # Core configuration
│   │   └── config.py          # Environment & settings
│   ├── schemas/                # Pydantic models
│   │   └── chat.py            # Request/response schemas
│   └── services/               # Business logic
│       ├── embedding_client.py    # Embedding Service client
│       ├── enhanced_rag.py        # RAG pipeline
│       ├── ollama_client.py       # LLM client
│       ├── raptor_client.py       # RAPTOR client
│       ├── short_term_memory.py   # Session memory
│       └── vector_client.py     # Vector Service client
├── Dockerfile
└── requirements.txt
```

## Service Dependencies

The Backend depends on:
- **Embedding Service** (port 8000): For text embeddings
- **Vector Storage Service** (port 8001): For vector search
- **Ollama** (port 11434): For LLM responses

Ensure all services are running before starting the Backend.

## Session Management

Sessions automatically expire after 60 minutes of inactivity:

```python
# Session lifecycle
1. User sends message with session_id
2. Session loaded or created
3. Message added to short-term memory
4. Response generated with context
5. Session saved with updated timestamp
6. Auto-cleanup removes expired sessions
```

## RAG Pipeline Flow

```
1. Receive user message
2. Load session context (short-term memory)
3. Query vector storage (long-term memory)
4. Rerank results (if RAPTOR enabled)
5. Build prompt with context
6. Call Ollama LLM
7. Return response
8. Store interaction in session
```

## Troubleshooting

### Connection Errors

Verify all services are running:
```bash
curl http://localhost:8000/      # Embedder
curl http://localhost:8001/      # Vector Storage
curl http://localhost:11434/     # Ollama
```

### Import Errors

Install dependencies:
```bash
pip install -r requirements.txt
```

### Session Not Found

Sessions expire after 60 minutes. Create a new session or increase `SESSION_TIMEOUT_MINUTES`.

## Development

### Adding New Endpoints

1. Create route handler in `app/api/`
2. Add Pydantic schema in `app/schemas/`
3. Implement business logic in `app/services/`
4. Register router in `app/main.py`

### Testing

```bash
pytest
```

## See Also

- [Project Root README](../README.md)
- [Embedding Service](../personal_LLM_embedder/README.md)
- [Vector Storage Service](../VECTOR_STORAGE_SERVICE/README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)

---

**Built with ❤️ for the Personal LLM RAG System**

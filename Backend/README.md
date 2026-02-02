# Backend RAG Orchestrator Service

A FastAPI-based microservice that orchestrates embeddings and vector storage to build a conversational RAG (Retrieval-Augmented Generation) pipeline for your Personal LLM system.

## Overview

The Backend acts as the **central orchestrator** for your Personal LLM. It coordinates between:
- **Embedder Service** (port 8000): Generates vector embeddings from text using SentenceTransformers.
- **Vector Storage Service** (port 8003): Stores and retrieves vector embeddings using ChromaDB.
- **LLM Service** (future): Will generate context-aware responses.

### Data Flow

```
User Input
    ↓
POST /chat (Backend)
    ├→ Call Embedder Service → Get embedding vector
    ├→ Store embedding in Vector DB (with session context)
    ├→ Query Vector DB → Retrieve similar past messages
    ├→ Extract top documents & scores
    ├→ Compose LLM answer (stub for now)
    └→ Return query, retrieved context, and answer
    ↓
Response with conversational context
```

## Features

✅ **Session-based Conversations**: Keep related messages grouped by `session_id`  
✅ **Vector Embeddings**: Compute embeddings for every user message  
✅ **Context Retrieval**: Automatically recall similar past messages  
✅ **RAG Pipeline**: Compose answers using retrieved documents (stub LLM ready for real LLM integration)  
✅ **Memory Store**: Store arbitrary memories with `POST /memory`  
✅ **Swagger UI**: Interactive API documentation at `/docs`

## Setup

### 1. Prerequisites

- Python 3.10+ (tested with 3.10)
- Git (optional, for cloning)
- Three running services:
  - **Embedder** (port 8000)
  - **Vector Storage** (port 8003)
  - **Backend** (port 8001)

### 2. Activate Virtual Environment

On **Windows PowerShell**:
```powershell
.venv\Scripts\Activate.ps1
```

On **Windows CMD**:
```cmd
.venv\Scripts\activate.bat
```

On **macOS/Linux**:
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

Key dependencies:
- `fastapi>=0.95.0` — Web framework
- `uvicorn[standard]>=0.22.0` — ASGI server
- `pydantic>=1.10.0` — Request/response validation
- `httpx>=0.24.0` — Async HTTP client for calling other services
- `python-dotenv>=1.0.0` — Load environment variables

### 4. Environment Variables (Optional)

Create a `.env` file in the `Backend` folder to override defaults:

```env
EMBEDDING_SERVICE_URL=http://localhost:8000
VECTOR_SERVICE_URL=http://localhost:8003
```

## Running the Service

### Start the Backend

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

**Output** (when successful):
```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Access Interactive Docs

Open your browser and go to:
- **Swagger UI** (recommended): http://127.0.0.1:8001/docs
- **ReDoc**: http://127.0.0.1:8001/redoc

## API Endpoints

### 1. `POST /chat` — Query with Conversation Context

**Purpose**: Process a user query, store it, retrieve similar past messages, and compose a context-aware answer.

**Request**:
```json
{
  "message": "What did I say about hiking?",
  "session_id": "user_session_123",
  "top_k": 3
}
```

**Response**:
```json
{
  "status": "ok",
  "data": {
    "query": "What did I say about hiking?",
    "session_id": "user_session_123",
    "embedding_dim": 768,
    "stored": {
      "status": "success",
      "id": "05d88822-2aa1-4bf3-859f-c96b810235ef",
      "text": "What did I say about hiking?"
    },
    "retrieved": [
      {
        "id": "doc_1",
        "document": "I love hiking in the mountains.",
        "metadata": {"session_id": "user_session_123"},
        "score": 0.95
      },
      {
        "id": "doc_2",
        "document": "Hiking is my favorite outdoor activity.",
        "metadata": {"session_id": "user_session_123"},
        "score": 0.88
      }
    ],
    "answer": "Relevant memories:\nI love hiking in the mountains.\n---\nHiking is my favorite outdoor activity.\n\nAnswer (stub): I found these memories related to your question."
  }
}
```

**Parameters**:
- `message` (string, required): User query/message
- `session_id` (string, optional): Group related messages. If not provided, defaults to anonymous.
- `top_k` (int, optional, default=5): Number of similar past messages to retrieve

### 2. `POST /memory` — Store a Memory

**Purpose**: Store arbitrary text/memories for a session without querying.

**Request**:
```json
{
  "text": "My favorite food is pizza.",
  "session_id": "user_session_123"
}
```

**Response**:
```json
{
  "status": "ok",
  "data": {
    "status": "success",
    "id": "mem_12345",
    "text": "My favorite food is pizza.",
    "metadata_status": "pending"
  }
}
```

**Parameters**:
- `text` (string, required): Memory/document to store
- `session_id` (string, optional): Group memories by session

### 3. `GET /` — Health Check

**Purpose**: Verify the Backend is running.

**Response**:
```json
{
  "message": "Backend is running"
}
```

## Usage Example

### Step 1: Store a Memory
```bash
curl -X POST http://127.0.0.1:8001/memory \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My name is Gaurav and I love AI.",
    "session_id": "sess1"
  }'
```

### Step 2: Query with Context
```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Who am I?",
    "session_id": "sess1",
    "top_k": 5
  }'
```

### Step 3: View Response
The response includes:
- Your query
- Retrieved similar past messages (with scores)
- An AI-composed answer using those retrieved messages

## Project Structure

```
Backend/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   └── chat.py             # Chat & memory endpoints
│   ├── services/
│   │   ├── embedding_client.py # Calls embedder service
│   │   ├── vector_client.py    # Calls vector storage service
│   │   └── rag_pipeline.py     # Orchestrates embeddings, storage, retrieval, LLM
│   ├── schemas/
│   │   └── chat.py             # Pydantic models (ChatRequest, MemoryRequest)
│   └── core/
│       └── config.py           # Configuration (URLs, env vars)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Optional Docker setup
└── README.md                   # This file
```

## Architecture

### Service Dependencies

```
┌─────────────────────────────────┐
│   Backend (Orchestrator)        │
│   http://127.0.0.1:8001        │
│                                 │
│  - POST /chat                  │
│  - POST /memory                │
│  - GET /docs (Swagger UI)      │
└────────┬──────────────┬────────┘
         │              │
         ▼              ▼
┌──────────────────┐  ┌──────────────────┐
│  Embedder Svc    │  │ Vector Store Svc │
│  (port 8000)     │  │  (port 8003)     │
│                  │  │                  │
│  POST /embed     │  │  POST /add_text  │
│  GET /           │  │  POST /query_text│
└──────────────────┘  │  POST /add_vector│
                      │  POST /query_vect│
                      │  GET /vectors    │
                      └──────────────────┘
```

### Request Flow

1. **User sends message** → `POST /chat`
2. **Backend calls Embedder** → Compute vector embedding
3. **Backend stores vector** → Call Vector Service `/add_vector`
4. **Backend queries vectors** → Retrieve similar past messages
5. **Backend extracts & normalizes** → Format top documents
6. **Backend composes answer** → Call LLM (stub for now)
7. **Backend responds** → Return query, context, and answer

## Future Enhancements

- [ ] **Real LLM Integration**: Replace `_compose_answer()` stub with OpenAI/Gemini/local LLM calls
- [ ] **Web Search**: Integrate web search to supplement vector retrieval
- [ ] **Authentication**: Add JWT or API key authentication
- [ ] **Rate Limiting**: Prevent abuse
- [ ] **Web Frontend**: Build a React/Vue UI to consume these APIs
- [ ] **Conversation History**: Add `GET /history/{session_id}` to list all memories in a session
- [ ] **Docker**: Deploy services as containers

## Troubleshooting

### Error: `No module named uvicorn`
→ Make sure venv is activated: `.venv\Scripts\Activate.ps1`
→ Then reinstall: `pip install -r requirements.txt`

### Error: `Connection refused (port 8001)`
→ Backend may not be running. Start it: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8001`

### Error: `Connection to Embedder/Vector Service failed`
→ Make sure embedder (port 8000) and vector storage (port 8003) are running
→ Check their health endpoints: http://127.0.0.1:8000/ and http://127.0.0.1:8003/

### Error: `WinError 10013: Socket access denied`
→ Port 8001 is already in use. Kill it:
```powershell
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

## Development Tips

- **Reload on file changes**: Add `--reload` flag:
  ```powershell
  python -m uvicorn app.main:app --reload --port 8001
  ```
- **Check logs**: Watch the terminal running uvicorn for detailed error traces
- **Test endpoints**: Use Swagger UI at `/docs` to test without writing curl commands

## Collaboration Notes

When merging with your friends' work:
1. Keep this Backend as the **central orchestrator**
2. Integrate new services by adding client functions in `app/services/`
3. Add new routes in `app/api/`
4. Update `app/services/rag_pipeline.py` to call new services
5. Document new endpoints in this README

---

**Built with ❤️ for your Personal LLM Project**

For questions or updates, see the main project repository.

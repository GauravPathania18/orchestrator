# Quick Start Guide

## 🚀 Run the System (3 Steps)

### Step 1: Terminal 1 - Embedding Service
```bash
cd personal_LLM_embedder
python embedder_api.py
```
✅ Running on http://localhost:8000

### Step 2: Terminal 2 - Vector Storage  
```bash
cd VECTOR_STORAGE_SERVICE
python run.py
```
✅ Running on http://localhost:8001

### Step 3: Terminal 3 - Backend Service
```bash
cd Backend
python -m uvicorn app.main:app --reload --port 8002
```
✅ Running on http://localhost:8002

---

## 🌲 Test RAPTOR Features

### 1. Ingest Documents with RAPTOR
```bash
curl -X POST http://localhost:8001/raptor/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Machine learning is a method of data analysis that automates analytical model building.",
      "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
      "Natural language processing helps computers understand human language.",
      "Computer vision enables machines to interpret visual information from the world."
    ],
    "cluster_size": 2
  }'
```

### 2. Query with RAPTOR
```bash
curl -X POST http://localhost:8001/raptor/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "k_summary": 2,
    "k_chunks": 5,
    "top_k_final": 3
  }'
```

### 3. Enhanced Chat with Backend
```bash
curl -X POST http://localhost:8002/api/raptor/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "top_k": 5
  }'
```

---

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up --build

# Stop all services  
docker-compose down

# View logs
docker-compose logs -f
```

Services will be available at:
- Embedder: http://localhost:8000
- Vector Storage: http://localhost:8001  
- Backend: http://localhost:8002
- Nginx Proxy: http://localhost:80

---

## 📊 API Endpoints

### RAPTOR Endpoints
- `POST /raptor/ingest` - Ingest documents with clustering
- `POST /raptor/query` - Query with hierarchical retrieval
- `GET /raptor/stats` - Pipeline statistics
- `POST /raptor/reset` - Clear RAPTOR data

### Backend Endpoints  
- `POST /api/chat` - Standard chat (legacy RAG)
- `POST /api/raptor/chat` - Enhanced chat with RAPTOR
- `POST /api/memory` - Store documents
- `POST /api/raptor/ingest` - RAPTOR ingestion via backend

### Vector Endpoints
- `POST /vectors/add_text` - Add text to vector DB
- `POST /vectors/query_text` - Query by text
- `GET /vectors/` - Get all vectors
```bash
curl -X POST http://localhost:8002/memory \
  -H "Content-Type: application/json" \
  -d '{"text":"Python is a great programming language","session_id":"user1"}'
```

### 2. Ask a Question
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Why is Python great?","session_id":"user1","top_k":5}'
```

### 3. View All Vectors
```bash
curl http://localhost:8001/vectors
```

### 4. Test Direct Embedding
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["hello world"]}'
```

---

## 🔗 API Documentation

Click these links to explore:
- **Embedding API:** http://localhost:8000/docs
- **Vector Storage API:** http://localhost:8001/docs
- **Backend API:** http://localhost:8002/docs

---

## ⚙️ Configuration

Edit `.env` to change:
- Service ports
- Embedding model
- Vector storage path
- Ollama LLM URL

---

## 📁 Project Files You Need

```
✅ CRITICAL:
   .env              - Settings
   .venv/            - Python packages
   Backend/          - Main service
   personal_LLM_embedder/   - Embedding service
   VECTOR_STORAGE_SERVICE/  - Vector DB service

✅ IMPORTANT:
   requirements.txt  - Dependencies
   .vscode/          - IDE config
   README.md         - Full documentation

⚠️ Optional:
   pyproject.toml    - Python project config
```

---

## 🆘 Troubleshooting

**Port already in use?**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Python not found?**
```bash
.venv\Scripts\python.exe --version
```

**Missing packages?**
```bash
pip install -r requirements.txt
```

**Reset everything?**
```bash
rmdir /s chroma_store
pip install -r requirements.txt
# Then restart all 3 services
```

---

## 📞 Service Endpoints

| Service | Port | Base URL | Docs |
|---------|------|----------|------|
| Embedding | 8000 | http://localhost:8000 | /docs |
| Vector Storage | 8001 | http://localhost:8001 | /docs |
| Backend | 8002 | http://localhost:8002 | /docs |

---

**That's it! Your RAG system is ready.** 🎉

Check README.md for full documentation.

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
✅ Running on http://localhost:8003

### Step 3: Terminal 3 - Backend Service
```bash
cd Backend
python -m uvicorn app.main:app --reload --port 8002
```
✅ Running on http://localhost:8002

---

## 🧪 Test Commands

### 1. Store a Memory
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
curl http://localhost:8003/vectors
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
- **Vector Storage API:** http://localhost:8003/docs
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
| Vector Storage | 8003 | http://localhost:8003 | /docs |
| Backend | 8002 | http://localhost:8002 | /docs |

---

**That's it! Your RAG system is ready.** 🎉

Check README.md for full documentation.

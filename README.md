# Personal LLM RAG System

A modular, production-ready **Retrieval-Augmented Generation (RAG)** system with three microservices.

## 🎯 Overview

This project implements a personal knowledge management system that:
- **Embeds** text documents into high-dimensional vectors
- **Stores** vectors in ChromaDB for semantic search
- **Retrieves** relevant documents based on queries
- **Generates** responses using LLMs (via Ollama)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Backend Orchestrator (8002)         │
│  • /chat - Query with context               │
│  • /memory - Store documents                │
└────────────┬────────────────────────────────┘
             │
    ┌────────┼────────┐
    ↓        ↓        ↓
┌───────┐ ┌────────┐ ┌──────────┐
│Embed  │ │Vector  │ │ Ollama   │
│(8000) │ │Storage │ │  LLM     │
│       │ │(8003)  │ │(11434)   │
└───────┘ └────────┘ └──────────┘
```

### **Three Microservices**

1. **Embedding Service** (Port 8000)
   - Converts text → 768-dim vectors
   - Uses `sentence-transformers`
   - Endpoint: `/embed`

2. **Vector Storage** (Port 8003)
   - Persistent vector database
   - Built on ChromaDB
   - Endpoints: `/add_text`, `/query_text`, `/vectors`

3. **Backend Orchestrator** (Port 8002)
   - Coordinates all services
   - Implements RAG pipeline
   - Endpoints: `/chat`, `/memory`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment: `.venv/`
- Dependencies installed: `pip install -r requirements.txt`

### Start Services (3 Terminals)

**Terminal 1 - Embedding Service:**
```bash
cd personal_LLM_embedder
python embedder_api.py
# Runs on http://localhost:8000
```

**Terminal 2 - Vector Storage:**
```bash
cd VECTOR_STORAGE_SERVICE
python run.py
# Runs on http://localhost:8003
```

**Terminal 3 - Backend:**
```bash
cd Backend
python -m uvicorn app.main:app --reload --port 8002
# Runs on http://localhost:8002
```

### Test the System

**Store a memory:**
```bash
curl -X POST http://localhost:8002/memory \
  -H "Content-Type: application/json" \
  -d '{"text":"Paris is famous for the Eiffel Tower","session_id":"user1"}'
```

**Ask a question:**
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Paris famous for?","session_id":"user1"}'
```

**View stored vectors:**
```bash
curl http://localhost:8003/vectors
```

---

## 📁 Project Structure

```
PROJECT/
├── .env                      # Configuration (ports, URLs, settings)
├── .venv/                    # Virtual environment (11 packages)
├── .vscode/                  # VS Code settings & debug config
├── Backend/                  # Main orchestrator service
│   ├── app/main.py
│   ├── app/api/
│   ├── app/core/
│   ├── app/schemas/
│   └── app/services/
├── personal_LLM_embedder/    # Embedding service
│   └── embedder_api.py
├── VECTOR_STORAGE_SERVICE/   # Vector database service
│   └── app/
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## 📦 Dependencies

**11 total packages:**
```
fastapi, uvicorn, pydantic, chromadb, httpx, requests,
python-dotenv, beautifulsoup4, sentence-transformers, numpy, huggingface-hub
```

Install: `pip install -r requirements.txt`

---

## 🔧 Configuration

Edit `.env`:
```bash
EMBEDDER_URL=http://127.0.0.1:8000/embed
VECTOR_STORAGE_URL=http://localhost:8003
OLLAMA_URL=http://localhost:11434
PERSIST_DIR=./chroma_store
```

---

## 📚 API Documentation

- **Embedding:** http://localhost:8000/docs
- **Vector Storage:** http://localhost:8003/docs
- **Backend:** http://localhost:8002/docs

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Services won't start | Activate venv: `.venv\Scripts\Activate.ps1` |
| Import errors | Reinstall: `pip install -r requirements.txt` |
| Port conflicts | Check: `netstat -ano \| findstr :8000` |
| ChromaDB errors | Clear cache: `rmdir /s chroma_store` |

---

## 📄 License

See [LICENSE](LICENSE)

---

**Status:** ✅ Production Ready | **Python:** 3.10+ | **Last Updated:** February 2026

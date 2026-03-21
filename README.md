# Personal LLM RAG System with RAPTOR

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.95+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/ChromaDB-0.3+-orange.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker Ready">
</p>

<p align="center">
  <b>A modular, production-ready Retrieval-Augmented Generation (RAG) system with advanced hierarchical clustering and microservices architecture</b>
</p>

## Overview

This project implements a **personal knowledge management system** that combines state-of-the-art RAG techniques with RAPTOR hierarchical clustering for superior document retrieval and question answering.

### Key Capabilities

- **Embeds** text documents into high-dimensional vectors
- **Clusters** documents using RAPTOR hierarchical approach  
- **Stores** vectors and summaries in ChromaDB for semantic search
- **Retrieves** relevant documents with cluster-based filtering
- **Reranks** results using CrossEncoder for quality boost
- **Generates** responses using LLMs (via Ollama)

## � RAPTOR Architecture

```
Documents → Chunking → Embedding → Clustering → Summarization → Storage
                                                    ↓
Query → Embedding → Summary Retrieval → Cluster Filter → Chunk Retrieval → Rerank → LLM
```

### Advanced Features

- **RAPTOR Hierarchical Clustering** - Groups similar documents automatically into clusters with summaries
- **Summary-based Retrieval** - Fast filtering via cluster summaries before chunk-level search  
- **IVF-like Optimization** - Cluster-based document filtering for efficient retrieval
- **CrossEncoder Reranking** - Quality boost for final results using semantic similarity
- **HNSW Indexing** - Fast ANN search via ChromaDB
- **Microservices Architecture** - Scalable, modular design with independent services

## Microservices Architecture

```
┌─────────────────────────────────────────────┐
│         Backend Orchestrator (8002)         │
│  • /api/chat - Enhanced RAG with RAPTOR     │
│  • /api/memory - Store documents            │
│  • /api/raptor/* - RAPTOR operations       │
└────────────┬────────────────────────────────┘
             │
    ┌────────┼────────┐
    ↓        ↓        ↓
┌───────┐ ┌────────┐ ┌──────────┐
│Embed  │ │Vector  │ │ Ollama   │
│(8000) │ │Storage │ │  LLM     │
│       │ │(8001)  │ │(11434)   │
└───────┘ └────────┘ └──────────┘
```

| Service | Port | Description | Key Endpoints | Documentation |
|---------|------|-------------|---------------|---------------|
| **Embedding** | 8000 | Text → 768-dim vectors using sentence-transformers | `/embed` | [README](personal_LLM_embedder/README.md) |
| **Vector Storage** | 8001 | ChromaDB with RAPTOR clustering & reranking | `/raptor/index`, `/raptor/retrieve`, `/store`, `/search`, `/list` | [README](VECTOR_STORAGE_SERVICE/README.md) |
| **Backend** | 8002 | Main API orchestrator with session management | `/api/chat`, `/api/memory`, `/api/raptor/*` | [README](Backend/README.md) |
| **Ollama** | 11434 | Local LLM inference server | Ollama API | [Ollama Docs](https://ollama.ai/) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional, for containerized deployment)
- [Ollama](https://ollama.ai/) (for local LLM inference)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/GauravPathania18/personal-llm-rag.git
cd personal-llm-rag
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\Activate.ps1  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running Locally

Start each service in separate terminals:

**Terminal 1 - Embedding Service:**
```bash
cd personal_LLM_embedder
python embedder_api.py
# http://localhost:8000
```

**Terminal 2 - Vector Storage:**
```bash
cd VECTOR_STORAGE_SERVICE
python run.py
# http://localhost:8001
```

**Terminal 3 - Backend:**
```bash
cd Backend
python -m uvicorn app.main:app --reload --port 8002
# http://localhost:8002
```

### Docker Deployment (Recommended)

```bash
docker-compose up -d
```

This starts all services including Ollama LLM. Access:
- Backend API: http://localhost:8002
- Vector Storage: http://localhost:8001
- Embedding: http://localhost:8000

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
curl http://localhost:8001/list
```

---

## Project Structure

```
personal-llm-rag/
├── Backend/                  # Main orchestrator service
│   ├── app/
│   │   ├── api/             # API routes (chat, memory, raptor)
│   │   ├── core/            # Core configuration
│   │   ├── schemas/         # Pydantic models
│   │   └── services/        # Business logic & clients
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md            # Service documentation
├── personal_LLM_embedder/    # Embedding microservice
│   ├── embedder_api.py      # FastAPI embedding service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md            # Service documentation
├── VECTOR_STORAGE_SERVICE/   # Vector database with RAPTOR
│   ├── app/
│   │   ├── routes/          # API endpoints
│   │   └── services/        # RAPTOR implementation, reranker
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md            # Service documentation
├── docker-compose.yml       # Full stack deployment
├── nginx.conf              # Reverse proxy config
├── pyproject.toml          # Project metadata & dependencies
├── requirements.txt        # Root dependencies
├── .env.example           # Environment template
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md               # This file
```

Each service has its own README with detailed documentation. See the services table above for links.

## Configuration

Environment variables (`.env`):

```bash
EMBEDDER_URL=http://127.0.0.1:8000/embed
VECTOR_STORAGE_URL=http://localhost:8001
OLLAMA_URL=http://localhost:11434
PERSIST_DIR=./chroma_store
COLLECTION_NAME=personal_llm
```

## API Documentation

Interactive API documentation is available at:

- **Embedding:** http://localhost:8000/docs
- **Vector Storage:** http://localhost:8001/docs
- **Backend:** http://localhost:8002/docs

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
# Format with black
black .

# Lint with ruff
ruff check .
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Services won't start | Activate venv: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\Activate.ps1` (Windows) |
| Import errors | Reinstall: `pip install -r requirements.txt` |
| Port conflicts | Check: `netstat -ano \| findstr :8000` (Windows) or `lsof -i :8000` (Linux/Mac) |
| ChromaDB errors | Clear cache: `rm -rf chroma_store/` |
| Ollama not responding | Verify Ollama is running: `ollama list` |

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to submit issues, feature requests, and pull requests.

## Roadmap

- [x] RAPTOR hierarchical clustering
- [x] CrossEncoder reranking
- [x] Session management
- [x] Docker deployment
- [ ] Web UI
- [ ] Multi-modal support (images, audio)
- [ ] Distributed deployment
- [ ] Model fine-tuning pipeline
- [ ] Vector store sharding

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## Support

For <strong>issues</strong> and feature requests, please use [GitHub Issues](https://github.com/GauravPathania18/personal-llm-rag/issues).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/GauravPathania18">GauravPathania18</a>
</p>

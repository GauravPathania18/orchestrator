# Personal LLM RAG System - Quick Start Guide

## Project Structure

```
PROJECT/
├── .venv/                     # Consolidated virtual environment (NEW!)
├── personal_LLM_embedder/     # Embedding Service (Port 8000)
├── VECTOR_STORAGE_SERVICE/    # Vector Storage Service (Port 8001)
├── Backend/                   # Orchestrator Service (Port 8002)
├── requirements.txt           # All dependencies (consolidated)
├── setup.bat                  # One-click setup (NEW!)
└── start_services.bat         # Start all services (UPDATED!)
```

## Quick Start

### 1. Setup (One-time only)

```cmd
setup.bat
```

This will:
- Create a virtual environment at `.venv/`
- Install all dependencies (FastAPI, transformers, ChromaDB, etc.)

### 2. Start Services

```cmd
start_services.bat
```

This will open 3 terminal windows:
- **Embedder Service** on port 8000
- **Vector Storage Service** on port 8001
- **Backend Orchestrator** on port 8002

Wait 5 seconds between each service startup.

## Service URLs

| Service | URL | API Docs |
|---------|-----|----------|
| Embedder | http://localhost:8000 | http://localhost:8000/docs |
| Vector Storage | http://localhost:8001 | http://localhost:8001/docs |
| Backend | http://localhost:8002 | http://localhost:8002/docs |

## Manual Commands

If you prefer to run services manually or individually:

### Activate Virtual Environment

```cmd
.venv\Scripts\activate.bat
```

### Start Embedder Service

```cmd
.venv\Scripts\activate.bat && cd personal_LLM_embedder && python embedder_api.py
```

### Start Vector Storage Service

```cmd
.venv\Scripts\activate.bat && cd VECTOR_STORAGE_SERVICE && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Start Backend Orchestrator

```cmd
.venv\Scripts\activate.bat && cd Backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## VS Code Configuration

To resolve import warnings in VS Code:

1. **Open Command Palette** (`Ctrl+Shift+P`)
2. Type **"Python: Select Interpreter"**
3. Choose: `e:\OneDrive\Desktop\PROJECT\.venv\Scripts\python.exe`

Or create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
```

## Troubleshooting

### Port Already in Use

If you get "Address already in use" errors:

```cmd
# Find and kill processes using ports 8000-8002
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %a
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8001 ^| findstr LISTENING') do taskkill /F /PID %a
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8002 ^| findstr LISTENING') do taskkill /F /PID %a
```

### Dependencies Missing

If imports fail:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Clean Reset

To remove and recreate everything:

```cmd
# Remove virtual environment
rmdir /s /q .venv

# Re-run setup
setup.bat
```

## Environment Variables

Each service can be configured via environment variables. See `.env.example` files in each service directory:

- `personal_LLM_embedder/.env.example`
- `VECTOR_STORAGE_SERVICE/.env.example`
- `Backend/.env.example`

## Requirements

- Python 3.10 or higher
- Windows (CMD or PowerShell)
- ~2GB disk space (for ML models and dependencies)

## Next Steps

Once services are running:

1. Visit http://localhost:8002/docs to explore the API
2. Use the API to upload documents
3. Query your documents using natural language

## Support

For issues:
1. Check that all 3 services are running
2. Verify ports 8000-8002 are available
3. Check service logs in each terminal window

# Virtual Environment Usage Guide

This project uses separate Python virtual environments for each microservice to avoid dependency conflicts.

## Project Structure

```
PROJECT/
├── personal_LLM_embedder/     # Embedding Service (Port 8000)
│   ├── .venv/                 # Virtual environment
│   └── requirements.txt
├── VECTOR_STORAGE_SERVICE/    # Vector Storage Service (Port 8001)
│   ├── .venv/                 # Virtual environment
│   └── requirements.txt
├── Backend/                   # Orchestrator Service (Port 8002)
│   ├── .venv/                 # Virtual environment
│   └── requirements.txt
├── setup_environments.bat     # Setup script
└── start_services.bat         # Start all services
```

## Quick Start

### Option 1: Using Batch Scripts (Recommended for Windows)

1. **Setup (One-time only):**
   ```cmd
   setup_environments.bat
   ```

2. **Start all services:**
   ```cmd
   start_services.bat
   ```

### Option 2: Manual Setup

#### 1. Personal LLM Embedder Service (Port 8000)

```cmd
cd personal_LLM_embedder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python embedder_api.py
```

#### 2. Vector Storage Service (Port 8001)

```cmd
cd VECTOR_STORAGE_SERVICE
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

#### 3. Backend Orchestrator (Port 8002)

```cmd
cd Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## Activating Virtual Environments

### Windows (Command Prompt)
```cmd
<service_folder>\.venv\Scripts\activate.bat
```

### Windows (PowerShell)
```powershell
<service_folder>\.venv\Scripts\Activate.ps1
```

### Deactivate
```cmd
deactivate
```

## Installing New Dependencies

When adding new packages to a service:

1. Activate the virtual environment
2. Install the package: `pip install <package_name>`
3. Update requirements.txt: `pip freeze > requirements.txt`

## Troubleshooting

### "pip is not recognized"
Activate the virtual environment first, then use pip.

### "Module not found" errors
Ensure you:
1. Activated the correct virtual environment
2. Installed requirements: `pip install -r requirements.txt`

### Port conflicts
Make sure no other services are using ports 8000, 8001, or 8002.

## Service URLs

Once running, access the services at:

| Service | URL | API Docs |
|---------|-----|----------|
| Embedder | http://localhost:8000 | http://localhost:8000/docs |
| Vector Storage | http://localhost:8001 | http://localhost:8001/docs |
| Backend | http://localhost:8002 | http://localhost:8002/docs |

## Environment Variables

Each service can be configured via environment variables. See `.env.example` files in each service directory.

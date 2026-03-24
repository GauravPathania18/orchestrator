@echo off
REM Start all services for Personal LLM RAG System

echo ==========================================
echo Personal LLM RAG System - Start Services
echo ==========================================
echo.
echo Services will start in order:
echo   1. Embedder Service (Port 8000)
echo   2. Vector Storage Service (Port 8001)
echo   3. Backend Orchestrator (Port 8002)
echo.
echo Press Ctrl+C to stop a service, or close the terminal window
echo.
pause

start "Embedder Service (8000)" cmd /k "cd personal_LLM_embedder && .venv\Scripts\activate.bat && python embedder_api.py"
timeout /t 5 /nobreak >nul

start "Vector Storage (8001)" cmd /k "cd VECTOR_STORAGE_SERVICE && .venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001"
timeout /t 5 /nobreak >nul

start "Backend Orchestrator (8002)" cmd /k "cd Backend && .venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002"

echo.
echo ==========================================
echo All services started!
echo ==========================================
echo.
echo Service URLs:
echo   - Embedder:    http://localhost:8000
echo   - Vector Store: http://localhost:8001
echo   - Backend:     http://localhost:8002
echo.

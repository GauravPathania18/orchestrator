@echo off
REM Setup script for Personal LLM RAG System (Consolidated Virtual Environment)

echo ==========================================
echo Personal LLM RAG System - Setup
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('python --version') do set PYTHON_VERSION=%%a
echo Found: %PYTHON_VERSION%
echo.

REM Create virtual environment
if exist ".venv" (
    echo Virtual environment already exists.
    echo.
    choice /C YN /M "Re-create it (will delete existing .venv)"
    if errorlevel 2 goto :skip_venv
    if errorlevel 1 (
        echo Removing existing virtual environment...
        rmdir /s /q .venv 2>nul
        if exist ".venv" (
            echo Failed to remove .venv. Please delete it manually.
            pause
            exit /b 1
        )
    )
)

echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

:skip_venv
REM Upgrade pip
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing...
)

REM Install dependencies
echo.
echo Installing dependencies (this may take a few minutes)...
echo   - FastAPI, Uvicorn
echo   - Sentence Transformers
echo   - ChromaDB, scikit-learn
echo   - NumPy, httpx, and more...
echo.

.venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Try running manually: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Project is ready to run.
echo.
echo To start all services:
echo   start_services.bat
echo.
echo Or start services individually:
echo   .venv\Scripts\activate.bat
echo   cd personal_LLM_embedder ^&^& python embedder_api.py
echo.
echo Service URLs when running:
echo   - Embedder:     http://localhost:8000
echo   - Vector Store: http://localhost:8001
echo   - Backend:      http://localhost:8002
echo.
pause

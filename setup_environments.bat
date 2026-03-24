@echo off
REM Setup script for Personal LLM RAG System - Creates virtual environments and installs dependencies

echo ==========================================
echo Personal LLM RAG System - Setup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    exit /b 1
)

echo Python detected:
python --version
echo.

REM ==========================================
REM Setup personal_LLM_embedder (Port 8000)
REM ==========================================
echo [1/3] Setting up personal_LLM_embedder...
if exist "personal_LLM_embedder\.venv" (
    echo    Virtual environment already exists
) else (
    echo    Creating virtual environment...
    python -m venv personal_LLM_embedder\.venv
)
echo    Installing dependencies...
call personal_LLM_embedder\.venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r personal_LLM_embedder\requirements.txt
call deactivate
echo    [OK] personal_LLM_embedder ready!
echo.

REM ==========================================
REM Setup VECTOR_STORAGE_SERVICE (Port 8001)
REM ==========================================
echo [2/3] Setting up VECTOR_STORAGE_SERVICE...
if exist "VECTOR_STORAGE_SERVICE\.venv" (
    echo    Virtual environment already exists
) else (
    echo    Creating virtual environment...
    python -m venv VECTOR_STORAGE_SERVICE\.venv
)
echo    Installing dependencies...
call VECTOR_STORAGE_SERVICE\.venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r VECTOR_STORAGE_SERVICE\requirements.txt
call deactivate
echo    [OK] VECTOR_STORAGE_SERVICE ready!
echo.

REM ==========================================
REM Setup Backend (Port 8002)
REM ==========================================
echo [3/3] Setting up Backend (Orchestrator)...
if exist "Backend\.venv" (
    echo    Virtual environment already exists
) else (
    echo    Creating virtual environment...
    python -m venv Backend\.venv
)
echo    Installing dependencies...
call Backend\.venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r Backend\requirements.txt
call deactivate
echo    [OK] Backend ready!
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo All virtual environments created and dependencies installed.
echo.
echo To start the services, run: start_services.bat
echo.

pause

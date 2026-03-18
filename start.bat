@echo off
title Conversation Copilot
echo ==========================================
echo   CONVERSATION COPILOT - Starting...
echo ==========================================
echo.

cd /d "%~dp0"

:: Check for .env
if not exist ".env" (
    echo [!] No .env file found. Copying from .env.example...
    copy .env.example .env
    echo [!] Please edit .env with your API keys before running.
    echo     notepad .env
    pause
    exit /b 1
)

:: Check for venv
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment...
    python -m venv .venv
    echo [*] Installing dependencies...
    .venv\Scripts\pip install -r backend\requirements.txt
)

:: Activate and run
echo [*] Starting server on http://127.0.0.1:8765
echo [*] Press Ctrl+C to stop
echo.
.venv\Scripts\python backend\main.py

@echo off
REM Video Edit Studio - Windows baslatici
cd /d "%~dp0"
if not exist .venv python -m venv .venv
echo Paketler kontrol ediliyor...
.venv\Scripts\python -m pip install -q --disable-pip-version-check -r requirements.txt
if not exist .env copy .env.example .env
echo Arayuz: http://localhost:8000
.venv\Scripts\python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause

#!/usr/bin/env bash
# Video Edit Studio — macOS / Linux başlatıcı
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
echo "Paketler kontrol ediliyor..."
.venv/bin/python -m pip install -q --disable-pip-version-check -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Arayüz: http://localhost:8000"
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

# KANA NLP Service

Ini adalah service backend Python untuk NLP Matching Engine KANA.

## Setup Lingkungan (Virtual Environment)
1. Buat virtual environment: `python3 -m venv venv`
2. Aktivasi: 
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

## Instalasi Dependensi
Jalankan: `pip install -r requirements.txt`

## Konfigurasi
Copy `.env.example` menjadi `.env`: `cp .env.example .env`

## Menjalankan Service
Jalankan server Uvicorn (port default 8001 sesuai spec KANA):
`uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload`

## Testing & Dokumentasi
- Cek Health: `curl http://localhost:8001/health`
- Swagger UI API Docs: Buka `http://localhost:8001/docs` di browser.

## Status Implementasi
- [x] Phase 1: Setup FastAPI & Health Check
- [ ] Phase 2: Model Loading & Preprocessing
- [ ] Phase 3: Embedding & Matching Endpoints
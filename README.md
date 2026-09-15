# KANA NLP Service

Backend Python untuk **KANA NLP Matching Engine**, layanan *microservice* pencocokan material dan limbah berbasis *lexical ranking* (BM25) serta *semantic vector similarity*.

## Setup Lingkungan (Virtual Environment)
1. Buat virtual environment:
   `python -m venv venv`
2. Aktivasi:
   - **Linux/macOS:** `source venv/bin/activate`
   - **Windows:** `venv\Scripts\activate`

## Instalasi Dependensi
Jalankan:
`pip install -r requirements.txt`

## Konfigurasi
Salin `.env.example` menjadi `.env`:
`cp .env.example .env`

## Menjalankan Service
Jalankan server Uvicorn (port 8001 sesuai spesifikasi KANA):
`uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload`

## API & Dokumentasi
- **Health Check:** `curl http://localhost:8001/health`
- **Swagger UI API Docs:** Akses `http://localhost:8001/docs` di peramban.

## Pengujian (Testing)
Jalankan seluruh *unit test* otomatis (Phase 3 hingga Phase 6):
`pytest tests/ -v`

## Status Implementasi
- [x] **Phase 1:** Setup FastAPI & Health Check Endpoint
- [x] **Phase 2:** Setup Environment & Arsitektur Project
- [x] **Phase 3:** Model Embedding (`paraphrase-multilingual-MiniLM-L12-v2`, 384 Dimensi) & Endpoint `/v1/embed`
- [x] **Phase 4:** Preprocessing Pipeline Deterministik (Tokenisasi, Normalisasi, Separator Angka/Satuan, Sastrawi Opsional)
- [x] **Phase 5:** BM25 Candidate Generation (`BM25Okapi` Lexical Filtering & Top-K Ranking)
- [x] **Phase 6:** Cosine Similarity Engine (Vektor Matematika & Penanganan Edge Cases)
- [ ] **Phase 7:** Final Match Engine & Score Thresholding (`/v1/match` Endpoint)

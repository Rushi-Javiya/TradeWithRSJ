# deeepr-mvp

An MVP scaffold for a research assistant platform (RAG-based) inspired by deeepr.ai — ingestion, vector search, and a simple query API.

Tech stack (MVP)
- Backend: FastAPI (Python)
- RAG orchestration: LangChain (pipeline stubs)
- Embeddings/LLM: OpenAI (configurable)
- Vector DB: Pinecone (managed) or Qdrant (self-host)
- Storage: S3 / MinIO
- Frontend: Next.js (separate client)

This repo contains a minimal scaffold to get a working end-to-end developer workflow (ingest PDFs → embeddings → upsert → query via RAG).

Quickstart (local, using OpenAI + Pinecone)
1. Create and/or set environment variables in backend/.env:

OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=us-west1-gcp
PINECONE_INDEX=deeepr-index

2. Install backend dependencies (recommended in a virtualenv):

cd backend
pip install -r requirements.txt

3. Run the ingestion (example):

python ingest.py --pdfs ./sample_papers

4. Start the API server:

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

5. Use the /query endpoint to ask questions (see docs for API examples).

What's next
- Implement UI (Next.js) that calls /query and displays source snippets with page numbers.
- Add GROBID integration for citation parsing and metadata extraction.
- Harden ingestion with OCR fallback (Tesseract) for scanned PDFs.

License: MIT

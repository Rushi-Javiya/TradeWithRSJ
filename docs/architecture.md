# Architecture

This document sketches the initial architecture for the MVP.

Components:
- Ingestion service: parses PDFs, extracts metadata, chunks text, creates embeddings, upserts to Vector DB.
- Vector DB: Pinecone (managed) or Qdrant (self-host) stores embeddings and metadata.
- RAG service: backend API that receives user queries, retrieves top-k passages, formats a prompt, and calls an LLM to generate a grounded answer with citations.
- Frontend: Next.js app that provides a chat/search UI and a document viewer showing cited snippets.
- Storage: S3 for raw PDFs and processed artifacts.

Data flow:
1. User uploads PDFs → ingestion
2. Ingestion extracts text, runs embedding model, upserts vectors
3. User query → retriever returns top-k chunks
4. LLM generates an answer conditioned on chunks; backend returns answer + sources


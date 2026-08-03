# Sprint 1 backlog (MVP)

Goals: get an end-to-end RAG demo working with 3 sample PDFs.

Tasks:
- [ ] Setup repo scaffold and CI (this commit)
- [ ] Implement ingest.py to parse PDFs and upsert embeddings (local test with sentence-transformers)
- [ ] Implement basic FastAPI endpoints: /health, /query
- [ ] Wire a simple retrieval + LLM generation flow (LangChain)
- [ ] Create simple frontend page to call /query and display results
- [ ] Add README with setup instructions

Optional stretch:
- Integrate Pinecone for vector storage
- Add GROBID to extract citations

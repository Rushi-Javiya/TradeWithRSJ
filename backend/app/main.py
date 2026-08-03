from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(title="deeepr-mvp API")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query")
async def query(req: QueryRequest):
    """Placeholder RAG query endpoint.
    Implements: retrieve top-k passages from vector DB and call LLM to generate answer.
    """
    q = req.query
    # TODO: wire up LangChain/LlamaIndex pipeline here
    # Example return format: {answer: str, sources: [{doc_id, page, snippet}]}
    return {
        "answer": "This is a stub response. Connect your RAG pipeline in backend/services/rag.py",
        "sources": []
    }

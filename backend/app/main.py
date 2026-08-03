from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from services.rag import rag_service

app = FastAPI(title="deeepr-mvp API")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query")
async def query(req: QueryRequest):
    """RAG query endpoint. Returns answer + sources."""
    q = req.query
    try:
        if not rag_service.is_ready():
            raise HTTPException(status_code=503, detail="Vector store not configured. Run ingest.py or set Pinecone keys.")
        result = rag_service.query(q)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

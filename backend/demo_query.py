"""
A small CLI demo that runs a sample query against the running API or directly via the rag service.

Usage:
  python demo_query.py --mode api --query "What is RAG?"
  python demo_query.py --mode local --query "What is RAG?"

--mode api: calls http://localhost:8000/query
--mode local: imports services.rag.rag_service and runs query directly (must be in Codespace/venv)
"""

import os
import argparse
import requests


def call_api(q):
    url = os.getenv("API_URL", "http://localhost:8000/query")
    r = requests.post(url, json={"query": q})
    print("Status:", r.status_code)
    print(r.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "local"], default="api")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    if args.mode == "api":
        call_api(args.query)
    else:
        # Run local service directly
        from services.rag import rag_service

        if not rag_service.is_ready():
            print("RAG service not ready. Ensure Pinecone key is set or local FAISS index exists (run ingest.py without Pinecone key).")
        print(rag_service.query(args.query))

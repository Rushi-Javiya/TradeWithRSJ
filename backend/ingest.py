"""
Updated ingestion pipeline using LangChain embeddings and vectorstores.
- If PINECONE_API_KEY is set, upserts into Pinecone.
- Else, builds a FAISS index locally and saves to ./vector_store/faiss_index

Usage: python ingest.py --pdfs ./sample_papers
"""

import os
import argparse
from pathlib import Path
import pdfplumber
from tqdm import tqdm

# Import embeddings with fallbacks
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
except Exception:
    try:
        from langchain.embeddings import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None

try:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

# Vectorstores
try:
    from langchain.vectorstores import Pinecone as LangPinecone, FAISS
except Exception:
    from langchain.vectorstores import FAISS
    LangPinecone = None

import pinecone

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_text(pdf_path: Path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                texts.append({"doc": pdf.name, "page": i, "text": text})
    return texts


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def main(pdf_dir: str):
    pdf_dir = Path(pdf_dir)

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    INDEX_NAME = os.getenv("PINECONE_INDEX", "deeepr-index")

    # Choose embeddings
    embeddings = None
    if OPENAI_KEY and OpenAIEmbeddings is not None:
        try:
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            print("Warning: OpenAIEmbeddings init failed:", e)
            embeddings = None

    if embeddings is None:
        if HuggingFaceEmbeddings is not None:
            try:
                embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)
            except Exception as e:
                print("Warning: HuggingFaceEmbeddings init failed:", e)
                embeddings = None
        else:
            raise RuntimeError("No embeddings available. Install compatible langchain + sentence-transformers.")

    all_texts = []
    metadatas = []

    for pdf in pdf_dir.glob("*.pdf"):
        pages = extract_text(pdf)
        for p in pages:
            chunks = chunk_text(p["text"])
            for chunk in chunks:
                all_texts.append(chunk)
                metadatas.append({"doc": p["doc"], "page": p["page"]})

    if not all_texts:
        print("No text extracted from PDFs in", pdf_dir)
        return

    # Upsert to Pinecone or build FAISS
    if PINECONE_API_KEY and LangPinecone is not None:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        print(f"Upserting {len(all_texts)} chunks to Pinecone index '{INDEX_NAME}' (this will create the index if needed)")
        # LangChain helper will create or connect
        try:
            LangPinecone.from_texts(texts=all_texts, embedding=embeddings, index_name=INDEX_NAME)
            print("Upsert to Pinecone completed.")
        except Exception as e:
            print("Failed to upsert to Pinecone:", e)
    else:
        # Build FAISS index locally
        print(f"Building local FAISS index with {len(all_texts)} chunks")
        try:
            faiss_index = FAISS.from_texts(texts=all_texts, embedding=embeddings)
            os.makedirs("./vector_store/faiss_index", exist_ok=True)
            faiss_index.save_local("./vector_store/faiss_index")
            print("Saved FAISS index to ./vector_store/faiss_index")
        except Exception as e:
            print("Failed to build FAISS index:", e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", required=True, help="Directory with PDF files")
    args = parser.parse_args()
    main(args.pdfs)

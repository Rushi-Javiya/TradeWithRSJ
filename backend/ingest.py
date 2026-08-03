"""
Simple ingestion script (stub).
- Parses PDFs with pdfplumber
- Chunks text and creates embeddings
- Upserts to Pinecone (or other vector DB)

Run: python ingest.py --pdfs ./sample_papers
"""

import os
import argparse
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer
import pinecone
from tqdm import tqdm

EMBED_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model for local testing


def extract_text(pdf_path: Path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            texts.append((i, text))
    return texts


def chunk_pages(pages, chunk_size=500, overlap=50):
    """Chunk by approximate token/word counts — simple implementation."""
    chunks = []
    for page_num, text in pages:
        words = text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i+chunk_size]
            chunks.append({
                "page": page_num,
                "text": " ".join(chunk_words)
            })
            i += chunk_size - overlap
    return chunks


def main(pdf_dir: str):
    pdf_dir = Path(pdf_dir)
    model = SentenceTransformer(EMBED_MODEL)

    # Init Pinecone if API key provided
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")
    INDEX_NAME = os.getenv("PINECONE_INDEX", "deeepr-index")

    if PINECONE_API_KEY:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(INDEX_NAME, dimension=384)
        index = pinecone.Index(INDEX_NAME)
    else:
        index = None

    for pdf in pdf_dir.glob("*.pdf"):
        pages = extract_text(pdf)
        chunks = chunk_pages(pages)
        texts = [c["text"] for c in chunks]
        embeds = model.encode(texts, show_progress_bar=True)

        if index:
            # upsert into pinecone
            to_upsert = []
            for i, emb in enumerate(embeds):
                metadata = {"doc": pdf.name, "page": chunks[i]["page"]}
                to_upsert.append((f"{pdf.stem}-{i}", emb.tolist(), metadata))
            # Pinecone accepts batches
            for i in range(0, len(to_upsert), 100):
                batch = to_upsert[i:i+100]
                index.upsert(batch)
        print(f"Indexed {pdf.name} ({len(chunks)} chunks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", required=True, help="Directory with PDF files")
    args = parser.parse_args()
    main(args.pdfs)

import os
from typing import List, Dict, Any

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangPinecone, FAISS
from langchain.schema import Document
from langchain.llms import OpenAI

import pinecone


class RAGService:
    def __init__(self):
        # Read environment
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENV")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "deeepr-index")

        # Embeddings: prefer OpenAI if available, else use a HuggingFace sentence-transformer model
        if self.openai_key:
            self.embeddings = OpenAIEmbeddings()
        else:
            # fallback: local sentence-transformers model
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Vector store: Pinecone if configured, else FAISS local (expects an index created by ingest)
        self.vs = None
        if self.pinecone_key:
            try:
                pinecone.init(api_key=self.pinecone_key, environment=self.pinecone_env)
                # Connect to existing index (created during ingestion)
                self.vs = LangPinecone.from_existing_index(index_name=self.pinecone_index, embedding=self.embeddings)
            except Exception as e:
                print("Warning: Pinecone init failed:", e)
                self.vs = None
        else:
            # Try to load local FAISS index from ./vector_store/faiss_index
            try:
                if os.path.exists("./vector_store/faiss_index"):
                    self.vs = FAISS.load_local("./vector_store/faiss_index", self.embeddings)
                else:
                    print("No Pinecone key and no local FAISS index found at ./vector_store/faiss_index")
                    self.vs = None
            except Exception as e:
                print("Warning: failed to load local FAISS index:", e)
                self.vs = None

        # LLM: prefer OpenAI if API key present
        self.llm = None
        if self.openai_key:
            self.llm = OpenAI(temperature=0)

    def is_ready(self) -> bool:
        return self.vs is not None

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        if not self.vs:
            raise RuntimeError("No vector store configured. Set PINECONE_API_KEY or create a local FAISS index via ingest.py")

        # Retrieve relevant docs
        docs: List[Document] = self.vs.similarity_search(question, k=k)

        # Build context and sources
        sources = []
        context_parts = []
        for i, d in enumerate(docs, start=1):
            md = d.metadata if hasattr(d, "metadata") else {}
            page = md.get("page") if md else None
            docname = md.get("doc") if md else None
            snippet = d.page_content if hasattr(d, "page_content") else str(d)
            sources.append({"doc": docname, "page": page, "snippet": snippet})
            context_parts.append(f"Source {i} — doc: {docname} page: {page}\n{snippet}\n")

        context = "\n---\n".join(context_parts)

        # Prompt
        prompt = (
            "You are an assistant that answers questions using the provided sources. "
            "Always include explicit citations in the form (doc: <name>, page: <num>) after sentences that rely on the source. "
            "If the answer is not contained in the sources, say you don't know rather than hallucinate.\n\n"
            f"SOURCES:\n{context}\nQUESTION: {question}\n\nAnswer concisely."
        )

        if self.llm:
            # Use LangChain OpenAI LLM wrapper
            try:
                response = self.llm(prompt)
                answer = response
            except Exception as e:
                answer = f"LLM call failed: {e}"
        else:
            # Fallback: return concatenated snippets as the "answer" with a note
            answer = (
                "[LLM not configured — set OPENAI_API_KEY for generative answers]\n\n" + context
            )

        return {"answer": answer, "sources": sources}


# Create a module-level singleton
rag_service = RAGService()

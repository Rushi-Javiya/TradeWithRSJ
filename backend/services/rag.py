import os
from typing import List, Dict, Any

# Import embeddings with fallbacks for different langchain versions
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

# Vectorstores and LLM
try:
    from langchain.vectorstores import Pinecone as LangPinecone, FAISS
except Exception:
    # older/newer langchain installations may structure vectorstores differently
    from langchain.vectorstores import FAISS
    LangPinecone = None

try:
    from langchain.schema import Document
except Exception:
    Document = None

try:
    from langchain.llms import OpenAI
except Exception:
    OpenAI = None

import pinecone


class RAGService:
    def __init__(self):
        # Read environment
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENV")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "deeepr-index")

        # Embeddings: prefer OpenAI if available, else use a HuggingFace sentence-transformer model
        self.embeddings = None
        if self.openai_key and OpenAIEmbeddings is not None:
            try:
                self.embeddings = OpenAIEmbeddings()
            except Exception as e:
                print("Warning: OpenAIEmbeddings init failed:", e)
                self.embeddings = None

        if self.embeddings is None:
            if HuggingFaceEmbeddings is not None:
                try:
                    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                except Exception as e:
                    print("Warning: HuggingFaceEmbeddings init failed:", e)
                    self.embeddings = None
            else:
                print("No embeddings implementation available. Please install a compatible langchain and sentence-transformers.")

        # Vector store: Pinecone if configured, else FAISS local (expects an index created by ingest)
        self.vs = None
        if self.pinecone_key and LangPinecone is not None:
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
        if self.openai_key and OpenAI is not None:
            try:
                self.llm = OpenAI(temperature=0)
            except Exception as e:
                print("Warning: OpenAI LLM init failed:", e)
                self.llm = None

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
            md = getattr(d, "metadata", None) or {}
            page = md.get("page") if md else None
            docname = md.get("doc") if md else None
            snippet = getattr(d, "page_content", None) or str(d)
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

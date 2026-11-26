"""Core RAG logic: read documents -> chunk -> embed -> retrieve -> ask the LLM."""
import io
import os

import numpy as np
from pypdf import PdfReader

CHUNK_SIZE = 900      # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
TOP_K = 4             # how many chunks we send to the LLM
MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")

_embedder = None


def get_embedder():
    """Load the embedding model once (small ONNX model, runs on CPU)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder


def embed(texts):
    vectors = np.array(list(get_embedder().embed(texts)), dtype="float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # normalise for cosine


def extract_text(filename, data):
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                      # try not to cut a sentence in half
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


class Store:
    """Tiny in-memory vector store (numpy matrix + list of chunk info)."""

    def __init__(self):
        self.items = []                          # {"doc": name, "text": chunk}
        self.vectors = np.empty((0, 384), dtype="float32")

    def add_document(self, name, text):
        chunks = chunk_text(text)
        if not chunks:
            return 0
        self.items += [{"doc": name, "text": c} for c in chunks]
        self.vectors = np.vstack([self.vectors, embed(chunks)])
        return len(chunks)

    def documents(self):
        return sorted({i["doc"] for i in self.items})

    def clear(self):
        self.__init__()

    def search(self, question, k=TOP_K):
        if not self.items:
            return []
        scores = self.vectors @ embed([question])[0]
        best = np.argsort(scores)[::-1][:k]
        return [{**self.items[i], "score": float(scores[i])} for i in best]


SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using ONLY the numbered "
    "sources provided. Cite sources like [1], [2]. If the sources do not contain "
    "the answer, say you could not find it in the uploaded documents."
)


def ask_llm(question, sources):
    import anthropic
    context = "\n\n".join(f"[{n}] ({s['doc']})\n{s['text']}" for n, s in enumerate(sources, 1))
    client = anthropic.Anthropic()               # reads ANTHROPIC_API_KEY
    reply = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}],
    )
    return reply.content[0].text

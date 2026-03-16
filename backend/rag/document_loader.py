# ============================================================
# RAG DOCUMENT LOADER — Per-Agent Chroma Indexing
# ============================================================

import os
import re
import tempfile
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    BSHTMLLoader,
    Docx2txtLoader,
)

from rank_bm25 import BM25Okapi
import pandas as pd

# Shared embeddings (same model as UniversalRAG — must be consistent)
_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Chroma persist directory
CHROMA_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_agents")
os.makedirs(CHROMA_BASE_DIR, exist_ok=True)

# In-memory BM25 indexes keyed by (agent_id, filename)
_bm25_indexes: dict = {}
_bm25_docs: dict = {}


def _collection_name(agent_id: str) -> str:
    """Sanitize agent_id into a valid Chroma collection name."""
    return f"agent_{re.sub(r'[^a-z0-9]', '', agent_id.lower())[:36]}"


def _get_chroma(agent_id: str) -> Chroma:
    """Return (or create) the Chroma collection for this agent."""
    name = _collection_name(agent_id)
    persist_dir = os.path.join(CHROMA_BASE_DIR, name)
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        collection_name=name,
        embedding_function=_embeddings,
        persist_directory=persist_dir,
    )


def _extract_req_id(text: str):
    m = re.search(r"req[\s\-_]?\d+", text, re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9]", "", m.group(0)).lower() if m else None


def _load_csv(path: str, agent_id: str, user_id: str) -> List[Document]:
    """Load a CSV into BM25 index and also return as Documents for Chroma."""
    fname = os.path.basename(path)
    df = pd.read_csv(path).fillna("N/A")
    docs, corpus = [], []

    for idx, row in df.iterrows():
        raw = " | ".join(f"{k}: {v}" for k, v in row.items())
        norm_id = _extract_req_id(raw)
        doc = Document(
            page_content=f"[Source: {fname}]\n{raw}",
            metadata={
                "source": fname,
                "row": idx + 1,
                "type": "csv",
                "norm_id": norm_id,
                "agent_id": agent_id,
                "user_id": user_id,
            },
        )
        docs.append(doc)
        corpus.append(re.findall(r"\w+", raw.lower()))

    key = (agent_id, fname)
    _bm25_indexes[key] = BM25Okapi(corpus)
    _bm25_docs[key] = docs
    return docs


def _process_file(
    path: str, agent_id: str, user_id: str, chunk_size: int = 1000
) -> List[Document]:
    """Load + chunk a document, injecting agent/user metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=150
    )
    fname = os.path.basename(path)
    name_lower = path.lower()

    if name_lower.endswith(".csv"):
        return _load_csv(path, agent_id, user_id)

    if name_lower.endswith(".pdf"):
        raw_docs = PyPDFLoader(path).load_and_split(splitter)
    elif name_lower.endswith(".docx"):
        raw_docs = Docx2txtLoader(path).load_and_split(splitter)
    elif name_lower.endswith(".txt"):
        raw_docs = TextLoader(path).load_and_split(splitter)
    elif name_lower.endswith(".html"):
        raw_docs = BSHTMLLoader(path).load_and_split(splitter)
    else:
        return []

    # Inject agent/user metadata into every chunk
    for doc in raw_docs:
        doc.metadata["agent_id"] = agent_id
        doc.metadata["user_id"] = user_id
        doc.metadata.setdefault("source", fname)

    return raw_docs


def index_file(
    file_path: str,
    agent_id: str,
    user_id: str,
    chunk_size: int = 1000,
) -> int:
    """
    Index a single file into the agent's dedicated Chroma collection.
    Returns the number of chunks stored.
    """
    docs = _process_file(file_path, agent_id, user_id, chunk_size)
    if not docs:
        return 0

    vector_db = _get_chroma(agent_id)
    vector_db.add_documents(docs)
    return len(docs)


def query_agent(agent_id: str, query: str, top_k: int = 4):
    """
    Search the agent's Chroma collection + BM25 CSV indexes.
    Returns (csv_docs, vector_docs).
    """
    tokens = re.findall(r"\w+", query.lower())
    q_id = _extract_req_id(query)

    # BM25 CSV search for this agent
    csv_results = []
    for (aid, fname), bm25 in _bm25_indexes.items():
        if aid != agent_id:
            continue
        scores = bm25.get_scores(tokens)
        for idx, score in enumerate(scores):
            doc = _bm25_docs[(aid, fname)][idx]
            if q_id and doc.metadata.get("norm_id") == q_id:
                score += 1000
            if score > 0:
                csv_results.append((doc, score))

    csv_results = [
        d for d, _ in sorted(csv_results, key=lambda x: x[1], reverse=True)[:5]
    ]

    # Vector search
    vector_docs = []
    try:
        vector_db = _get_chroma(agent_id)
        vector_docs = vector_db.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"⚠️ Chroma query failed for agent {agent_id}: {e}")

    return csv_results, vector_docs


def collection_name_for(agent_id: str) -> str:
    return _collection_name(agent_id)

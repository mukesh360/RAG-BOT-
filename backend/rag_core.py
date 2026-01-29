# ============================================================
# UNIVERSAL RAG CORE (EXTRACTED FROM STREAMLIT VERSION)
# NO LOGIC CHANGES - EXACT SAME RETRIEVAL AND LLM BEHAVIOR
# ============================================================

import os
import re
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    BSHTMLLoader,
    Docx2txtLoader
)

# ============================================================
# UTILITIES (UNCHANGED)
# ============================================================

def normalize_id(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower() if text else ""

def extract_req_id(text: str):
    m = re.search(r"req[\s\-_]?\d+", text, re.IGNORECASE)
    return normalize_id(m.group(0)) if m else None

def is_csv_intent(query: str) -> bool:
    keywords = ["row", "status", "id", "requirement", "category", "table", "csv", "value"]
    q = query.lower()
    return extract_req_id(q) is not None or any(k in q for k in keywords)

# ============================================================
# UNIVERSAL RAG CORE (UNCHANGED LOGIC)
# ============================================================

class UniversalRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        self.vector_db = None
        self.bm25_indexes = {}
        self.bm25_docs = {}

    # ---------------- CSV (UNCHANGED) ----------------
    def load_csv(self, path):
        fname = os.path.basename(path)
        df = pd.read_csv(path).fillna("N/A")

        docs, corpus = [], []

        for idx, row in df.iterrows():
            raw = " | ".join(f"{k}: {v}" for k, v in row.items())
            norm_id = extract_req_id(raw)

            doc = Document(
                page_content=f"[Source: {fname}]\n{raw}",
                metadata={
                    "source": fname,
                    "row": idx + 1,
                    "type": "csv",
                    "norm_id": norm_id
                }
            )

            docs.append(doc)
            corpus.append(re.findall(r"\w+", raw.lower()))

        self.bm25_indexes[fname] = BM25Okapi(corpus)
        self.bm25_docs[fname] = docs

    # ---------------- FILE ROUTER ----------------
    def process_file(self, path):
        name = path.lower()

        if name.endswith(".csv"):
            self.load_csv(path)
            return []

        if name.endswith(".pdf"):
            return PyPDFLoader(path).load_and_split(self.splitter)
        if name.endswith(".docx"):
            return Docx2txtLoader(path).load_and_split(self.splitter)
        if name.endswith(".txt"):
            return TextLoader(path).load_and_split(self.splitter)
        if name.endswith(".html"):
            return BSHTMLLoader(path).load_and_split(self.splitter)

        return []

    # ---------------- BASE DOCUMENTS PRESERVED ----------------
    def load_documents(self, folder):
        docs = []
        for f in os.listdir(folder):
            docs.extend(self.process_file(os.path.join(folder, f)))

        if docs:
            if self.vector_db:
                self.vector_db.add_documents(docs)
            else:
                self.vector_db = Chroma.from_documents(docs, self.embeddings)

    # ---------------- SEARCH ----------------
    def search(self, query):
        csv_results, doc_results = [], []
        q_id = extract_req_id(query)
        tokens = re.findall(r"\w+", query.lower())

        # Always search CSV
        for fname, bm25 in self.bm25_indexes.items():
            scores = bm25.get_scores(tokens)
            for idx, score in enumerate(scores):
                doc = self.bm25_docs[fname][idx]

                # Strong boost for explicit req ID
                if q_id and doc.metadata["norm_id"] == q_id:
                    score += 1000

                # Medium boost if requirement text matches
                if "requirement_text" in doc.page_content.lower():
                    score += 5

                if score > 0:
                    csv_results.append((doc, score))

        csv_results = [
            d for d, _ in sorted(csv_results, key=lambda x: x[1], reverse=True)[:5]
        ]

        # Vector DB (PDF / DOCX)
        if self.vector_db:
            doc_results = self.vector_db.similarity_search(query, k=4)

        return csv_results, doc_results

    # ---------------- ANSWER (ZERO HALLUCINATION) ----------------
    def answer(self, query):
        csv_docs, doc_docs = self.search(query)
        all_docs = csv_docs + doc_docs

        if not all_docs:
            return "I don't have any information about that in the loaded documents. Please upload relevant documents or ask about topics covered in the existing files.", []

        context = "\n\n---\n\n".join(d.page_content for d in all_docs)

        # STRICT ANTI-HALLUCINATION PROMPT
        prompt = ChatPromptTemplate.from_template("""You are a document assistant that ONLY answers based on the provided context.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY use information that is EXPLICITLY stated in the context below
2. If the answer is NOT in the context, say: "I cannot find this information in the loaded documents."
3. NEVER make up, infer, or assume information that isn't directly in the context
4. NEVER use your own knowledge - ONLY the context provided
5. If you're unsure, say you cannot find the information rather than guessing
6. Quote or reference specific parts of the context when answering
7. Be concise and factual

CONTEXT (This is the ONLY information you can use):
{context}

USER QUESTION:
{question}

ANSWER (Remember: ONLY use information from the context above, nothing else):""")

        answer = (
            prompt | self.llm | StrOutputParser()
        ).invoke({"context": context, "question": query})

        sources = list(dict.fromkeys(
            d.metadata.get("source", "Unknown") for d in all_docs
        ))[:3]

        return answer, sources

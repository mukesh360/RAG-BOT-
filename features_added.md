# Features Added - DocIntel AI v2.0

Complete documentation of all features in the RAG-Based AI Chatbot system.

---

## 🧠 RAG Core Features

### Hybrid Retrieval System
The system uses a sophisticated dual-retrieval approach:

| Component | Technology | Use Case |
|-----------|------------|----------|
| **BM25 Search** | rank-bm25 (Okapi) | CSV/tabular data - keyword matching |
| **Vector Search** | ChromaDB + HuggingFace | PDF, DOCX, TXT, HTML - semantic similarity |

### Embeddings Engine
```python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Normalization**: Enabled for consistent similarity scores
- **Device**: CPU-optimized for compatibility

### LLM Configuration
```python
ChatOllama(
    model="qwen2.5:7b",
    temperature=0  # Deterministic responses
)
```
- **Model**: Qwen 2.5 7B via Ollama
- **Temperature**: 0 (zero randomness for consistent answers)

### Document Processing

| Format | Loader | Processing |
|--------|--------|------------|
| **CSV** | pandas + BM25 | Row-by-row indexing with ID extraction |
| **PDF** | PyPDFLoader | Page-based chunking |
| **DOCX** | Docx2txtLoader | Full text extraction + splitting |
| **TXT** | TextLoader | Direct chunking |
| **HTML** | BSHTMLLoader | HTML parsing + text extraction |

### Text Chunking Strategy
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```
- **Chunk Size**: 1000 characters for optimal context
- **Overlap**: 150 characters to preserve context across chunks

### CSV-Specific Features

| Feature | Description |
|---------|-------------|
| **Requirement ID Extraction** | Regex pattern `req[\s\-_]?\d+` auto-detects IDs |
| **ID Normalization** | Consistent ID matching regardless of format |
| **Score Boosting** | +1000 boost for exact ID matches |
| **Row Metadata** | Each row tracked with source file and row number |

### Smart Query Routing
```python
def is_csv_intent(query):
    keywords = ["row", "status", "id", "requirement", 
                "category", "table", "csv", "value"]
```
- Automatically detects CSV-related queries
- Routes to BM25 for tabular data questions

### Hybrid Search Algorithm
1. **CSV Search (BM25)**
   - Tokenizes query into words
   - Scores all CSV rows against tokens
   - Applies ID boost (+1000) for exact matches
   - Returns top 5 results

2. **Vector Search (Chroma)**
   - Semantic similarity search
   - Returns top 4 most similar chunks

3. **Result Merging**
   - Combines CSV + document results
   - De-duplicates sources

---

## 🛡️ Zero Hallucination System (NEW)

### Strict Anti-Hallucination Prompt
The LLM is bound by 7 critical rules:

```
1. ONLY use information EXPLICITLY stated in context
2. If answer NOT in context → "I cannot find this information"
3. NEVER make up, infer, or assume information
4. NEVER use own knowledge - ONLY context provided
5. If unsure → say cannot find rather than guess
6. Quote or reference specific parts of context
7. Be concise and factual
```

### Improved Error Handling
- **No Results**: Clear message explaining no matching documents
- **Graceful Fallback**: Suggests uploading relevant documents
- **Source Limit**: Shows up to 3 sources for transparency

---

## 🖥️ FastAPI Backend Features

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main chat UI |
| `/query` | POST | Process question, return answer + sources |
| `/upload` | POST | Upload and index new documents |
| `/history` | GET | Get chat history |
| `/history/clear` | POST | Clear chat history (NEW) |
| `/health` | GET | Health check endpoint |

### Session Management
- **In-Memory History**: Stores last 50 Q&A pairs
- **Cache Lookup**: Returns cached answer for repeated questions
- **Clear Function**: Reset history via API

### File Upload Pipeline
1. Validate file extension
2. Save to temp directory
3. Process with appropriate loader
4. Index into vector DB or BM25
5. Return success count

---

## 🎨 UI/UX Features

### Visual Effects

| Effect | Description |
|--------|-------------|
| **Animated Background** | Moving gradient mesh |
| **4 Floating Orbs** | Multi-colored gradient orbs |
| **Animated Grid** | Moving grid pattern |
| **Particle System** | 30 floating particles |
| **Rainbow Borders** | Animated gradient borders |
| **Cursor Glow** | Purple glow follows cursor |
| **Glassmorphism** | Frosted glass on sidebar |

### Interactive Features

| Feature | Description |
|---------|-------------|
| **Ripple Effects** | Click ripples on buttons |
| **3D Tilt** | Cards tilt on hover |
| **Typewriter** | Text types on prompt click |
| **Confetti** | Celebration on upload |
| **Toast Notifications** | Animated feedback |
| **Typing Indicator** | 3 bouncing dots |

### New Components
- **New Chat Button** - Reset conversation
- **4 Suggested Prompts** - Quick start options
- **Premium Scrollbar** - Gradient styled

---

## 📁 Project Structure

```
RAG-Based-AI-Chatbot/
├── backend/
│   ├── main.py          # FastAPI app + endpoints
│   ├── rag_core.py      # UniversalRAG class
│   ├── schemas.py       # Pydantic models
│   └── storage.py       # In-memory history
├── frontend/
│   ├── templates/
│   │   └── index.html   # Chat UI
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── base_documents/      # Pre-loaded docs
├── requirements.txt
└── README.md
```

---

## 🔧 Tech Stack Summary

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI + Uvicorn |
| **LLM** | Ollama (Qwen 2.5:7b) |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **Keyword Search** | BM25Okapi |
| **Document Loaders** | LangChain Community |
| **Frontend** | HTML + CSS + Vanilla JS |
| **Styling** | Bootstrap 5 + Custom CSS |

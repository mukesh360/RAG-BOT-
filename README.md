# DocIntel AI - Universal RAG System

A document intelligence system using RAG (Retrieval-Augmented Generation) with FastAPI frontend.

## Features

- **Multi-format Support**: CSV, PDF, DOCX, TXT, HTML
- **Hybrid Retrieval**: BM25 for CSV + Chroma Vector DB for documents
- **No Hallucination**: Answers grounded in retrieved context only
- **Chat Interface**: Modern Bootstrap 5 UI with message history
- **File Upload**: Upload and index new documents on the fly

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **RAG**: LangChain + Chroma + BM25
- **LLM**: ChatOllama (qwen2.5:7b)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Frontend**: HTML + CSS + Vanilla JavaScript + Bootstrap 5

## Project Structure

```
RAG-Based-AI-Chatbot/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── rag_core.py      # UniversalRAG class
│   ├── schemas.py       # Pydantic models
│   └── storage.py       # In-memory history
├── frontend/
│   ├── templates/
│   │   └── index.html   # Chat UI
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
├── base_documents/      # Pre-loaded documents
├── requirements.txt
└── README.md
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running** with qwen2.5:7b model:
   ```bash
   ollama pull qwen2.5:7b
   ollama serve
   ```

## Running the Application

Start the FastAPI server:

```bash
cd c:\RAG\RAG-Based-AI-Chatbot
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at: **http://localhost:8000**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve chat UI |
| `/query` | POST | Ask a question |
| `/upload` | POST | Upload files |
| `/history` | GET | Get chat history |
| `/health` | GET | Health check |

## Usage

1. **Ask Questions**: Type your question in the chat input and press Enter
2. **Upload Documents**: Use the sidebar to upload CSV, PDF, DOCX, TXT, or HTML files
3. **View Sources**: Each answer shows the source documents used

## Base Documents

Place your documents in the `base_documents/` folder to have them automatically indexed on startup.


how to run ?
ollama pull qwen2.5:7b
ollama serve

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
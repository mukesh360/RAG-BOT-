# ============================================================
# FASTAPI APPLICATION - UNIVERSAL RAG UI
# ============================================================

import os
import tempfile
from datetime import datetime
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from .rag_core import UniversalRAG
from .schemas import QueryRequest, QueryResponse, HistoryItem, UploadResponse, HealthResponse, AgentConfig, ConfigResponse
from .storage import history_storage

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="DocIntel AI",
    description="Universal RAG System with FastAPI",
    version="1.0.0"
)

# Get the project root directory
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")

# Initialize RAG system
rag = UniversalRAG()

# Temp directory for uploads
TEMP_DIR = tempfile.mkdtemp()

# ============================================================
# STARTUP EVENT - LOAD BASE DOCUMENTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Load base documents on startup"""
    base_docs_path = PROJECT_ROOT / "base_documents"
    if base_docs_path.exists():
        print(f"📂 Loading base documents from {base_docs_path}")
        rag.load_documents(str(base_docs_path))
        print("✅ Base documents loaded successfully")
    else:
        print("⚠️ No base_documents folder found")

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current agent configuration"""
    return ConfigResponse(**rag.get_config())


@app.post("/config", response_model=ConfigResponse)
async def update_config(config: AgentConfig):
    """Update agent configuration at runtime"""
    update_data = {k: v for k, v in config.dict().items() if v is not None}
    rag.update_config(**update_data)
    return ConfigResponse(**rag.get_config())


@app.get("/history", response_model=List[HistoryItem])
async def get_history():
    """Get chat history"""
    return history_storage.get_all()


@app.post("/history/clear")
async def clear_history():
    """Clear chat history"""
    history_storage.clear()
    return {"status": "success", "message": "History cleared"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a question and return answer with sources"""
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Check cache first
    cached_answer, cached_sources = history_storage.find_cached(question)
    
    if cached_answer:
        # Return cached response
        entry = history_storage.add(question, cached_answer, cached_sources)
        return QueryResponse(
            answer=cached_answer,
            sources=cached_sources,
            timestamp=entry["timestamp"]
        )
    
    # Get fresh answer from RAG
    try:
        answer, sources = rag.answer(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    # Store in history
    entry = history_storage.add(question, answer, sources)
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        timestamp=entry["timestamp"]
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and index files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    allowed_extensions = {".csv", ".pdf", ".docx", ".txt", ".html"}
    processed_count = 0
    
    for file in files:
        # Validate extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            continue
        
        # Save to temp directory
        file_path = os.path.join(TEMP_DIR, file.filename)
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            processed_count += 1
        except Exception as e:
            print(f"Error saving file {file.filename}: {e}")
            continue
    
    # Index all files in temp directory
    if processed_count > 0:
        try:
            rag.load_documents(TEMP_DIR)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error indexing files: {str(e)}")
    
    return UploadResponse(
        status="success",
        files_processed=processed_count,
        message=f"Successfully processed {processed_count} file(s)"
    )


# ============================================================
# RUN WITH UVICORN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ============================================================
# FASTAPI APPLICATION - UNIVERSAL RAG UI + SUPABASE AUTH + AGENTS
# ============================================================

import os
import tempfile
from datetime import datetime
from typing import List
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .rag_core import UniversalRAG
from .schemas import QueryRequest, QueryResponse, HistoryItem, UploadResponse, HealthResponse, AgentConfig, ConfigResponse
from .storage import supabase_storage, document_storage
from .auth.routes import router as auth_router
from .auth.dependencies import get_current_user
from .agents.routes import router as agents_router

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="DocIntel AI",
    description="Universal RAG System with Agent Framework + Supabase Auth",
    version="3.0.0"
)

# Get the project root directory
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")

# Initialize RAG system (base documents)
rag = UniversalRAG()

# Temp directory for uploads
TEMP_DIR = tempfile.mkdtemp()

# ============================================================
# INCLUDE ROUTERS
# ============================================================

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(agents_router, prefix="/agents", tags=["Agents"])

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
# PAGE ROUTES (PUBLIC)
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page (auth check happens on frontend)"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Serve the signup page"""
    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/agents-page", response_class=HTMLResponse)
async def agents_page(request: Request):
    """Serve the agents management page"""
    return templates.TemplateResponse("agents.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint (public)"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

# ============================================================
# PROTECTED API ROUTES (require valid JWT)
# ============================================================

@app.get("/config", response_model=ConfigResponse)
async def get_config(user: dict = Depends(get_current_user)):
    """Get current agent configuration"""
    return ConfigResponse(**rag.get_config())


@app.post("/config", response_model=ConfigResponse)
async def update_config(config: AgentConfig, user: dict = Depends(get_current_user)):
    """Update agent configuration at runtime"""
    update_data = {k: v for k, v in config.dict().items() if v is not None}
    rag.update_config(**update_data)
    return ConfigResponse(**rag.get_config())


@app.get("/history", response_model=List[HistoryItem])
async def get_history(user: dict = Depends(get_current_user)):
    """Get chat history for the authenticated user"""
    return supabase_storage.get_all(user["id"])


@app.post("/history/clear")
async def clear_history(user: dict = Depends(get_current_user)):
    """Clear chat history for the authenticated user"""
    supabase_storage.clear(user["id"])
    return {"status": "success", "message": "History cleared"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, user: dict = Depends(get_current_user)):
    """
    Process a question using the BASE RAG system.
    If agent_id is provided, redirect logic to agent-specific RAG.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    user_id = user["id"]

    # If an agent_id is supplied, delegate to agent RAG
    if request.agent_id:
        from .agents.routes import query_agent_endpoint, AgentQueryRequest
        from .schemas import AgentQueryRequest as AQR
        return await query_agent_endpoint(
            agent_id=request.agent_id,
            request=AQR(question=question),
            user=user
        )

    # Check cache first (user-scoped)
    cached_answer, cached_sources = supabase_storage.find_cached(user_id, question)

    if cached_answer:
        entry = supabase_storage.add(user_id, question, cached_answer, cached_sources)
        return QueryResponse(
            answer=cached_answer,
            sources=cached_sources,
            timestamp=entry["timestamp"]
        )

    # Get fresh answer from base RAG
    try:
        answer, sources = rag.answer(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    # Store in history (user-scoped, no agent_id for base queries)
    entry = supabase_storage.add(user_id, question, answer, sources)

    return QueryResponse(
        answer=answer,
        sources=sources,
        timestamp=entry["timestamp"]
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    agent_id: str = Form(None),
    files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user)
):
    """
    Upload and index files.
    If agent_id is provided, index into that agent's collection.
    Otherwise, index into base RAG.
    """
    if agent_id and agent_id.strip():
        from .agents.routes import upload_to_agent
        result = await upload_to_agent(agent_id=agent_id, files=files, user=user)
        return UploadResponse(
            status=result["status"],
            files_processed=result["files_processed"],
            message=f"Successfully indexed to agent: {result.get('agent_name', 'Agent')}"
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    allowed_extensions = {".csv", ".pdf", ".docx", ".txt", ".html"}
    processed_count = 0
    user_id = user["id"]

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            continue

        file_path = os.path.join(TEMP_DIR, file.filename)
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Store document metadata in Supabase
            document_storage.add(
                user_id=user_id,
                filename=file.filename,
                file_type=ext.lstrip("."),
                file_size=len(content)
            )

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


@app.get("/documents")
async def get_documents(user: dict = Depends(get_current_user)):
    """Get all documents uploaded by the authenticated user"""
    return document_storage.get_all(user["id"])


# ============================================================
# RUN WITH UVICORN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

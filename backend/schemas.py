# ============================================================
# PYDANTIC MODELS FOR FASTAPI ENDPOINTS
# ============================================================

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    question: str


class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    answer: str
    sources: List[str]
    timestamp: str


class HistoryItem(BaseModel):
    """Model for a single chat history item"""
    timestamp: str
    question: str
    answer: str
    sources: List[str]


class UploadResponse(BaseModel):
    """Response model for /upload endpoint"""
    status: str
    files_processed: int
    message: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    timestamp: str


class AgentConfig(BaseModel):
    """Configurable agent settings"""
    model_name: Optional[str] = "qwen2.5:7b"
    temperature: Optional[float] = 0.0
    top_k: Optional[int] = 4
    chunk_size: Optional[int] = 1000
    system_prompt: Optional[str] = None


class ConfigResponse(BaseModel):
    """Response model for /config endpoint"""
    model_name: str
    temperature: float
    top_k: int
    chunk_size: int
    system_prompt: str


# ============================================================
# AUTH MODELS
# ============================================================

class AuthRequest(BaseModel):
    """Request model for /auth/signup and /auth/login"""
    email: str
    password: str


class AuthResponse(BaseModel):
    """Response model for auth endpoints"""
    access_token: str
    user_email: str
    message: str

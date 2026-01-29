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

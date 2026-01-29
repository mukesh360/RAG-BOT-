# ============================================================
# IN-MEMORY SESSION HISTORY STORAGE
# ============================================================

from collections import deque
from typing import List, Optional, Tuple
from datetime import datetime


class HistoryStorage:
    """In-memory storage for chat history (max 50 messages)"""
    
    def __init__(self, max_size: int = 50):
        self.history = deque(maxlen=max_size)
    
    def add(self, question: str, answer: str, sources: List[str]) -> dict:
        """Add a new message to history"""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "sources": sources
        }
        self.history.append(entry)
        return entry
    
    def get_all(self) -> List[dict]:
        """Get all history items"""
        return list(self.history)
    
    def find_cached(self, query: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Find a cached answer for the same question"""
        q = query.strip().lower()
        for h in reversed(self.history):
            if h["question"].strip().lower() == q:
                return h["answer"], h["sources"]
        return None, None
    
    def clear(self) -> None:
        """Clear all history"""
        self.history.clear()


# Global singleton instance
history_storage = HistoryStorage()

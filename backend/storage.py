# ============================================================
# CHAT HISTORY STORAGE — SUPABASE (per-user) + IN-MEMORY FALLBACK
# ============================================================

from collections import deque
from typing import List, Optional, Tuple
from datetime import datetime


class HistoryStorage:
    """In-memory storage for chat history (max 50 messages) — fallback only"""

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


# ============================================================
# SUPABASE CHAT STORAGE (per-user, persistent)
# ============================================================

class SupabaseChatStorage:
    """
    Stores chat history in Supabase per user_id.
    Uses the chat_history table with RLS.
    """

    def __init__(self):
        from .auth.supabase_client import supabase_admin
        self.supabase = supabase_admin
        # In-memory cache as fallback
        self._fallback = HistoryStorage()

    def add(self, user_id: str, question: str, answer: str, sources: List[str], agent_id: Optional[str] = None) -> dict:
        """Insert a chat entry into Supabase for the given user (optionally linked to an agent)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "sources": sources
        }

        try:
            row = {
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "sources": sources
            }
            if agent_id:
                row["agent_id"] = agent_id
            self.supabase.table("chat_history").insert(row).execute()
        except Exception as e:
            print(f"⚠️ Supabase chat_history insert failed: {e}")
            # Fallback to in-memory
            self._fallback.add(question, answer, sources)

        return entry

    def get_all(self, user_id: str) -> List[dict]:
        """Get all chat history for a specific user from Supabase"""
        try:
            response = (
                self.supabase.table("chat_history")
                .select("question, answer, sources, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .limit(50)
                .execute()
            )

            return [
                {
                    "timestamp": row["created_at"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "sources": row["sources"] or []
                }
                for row in (response.data or [])
            ]
        except Exception as e:
            print(f"⚠️ Supabase chat_history fetch failed: {e}")
            return self._fallback.get_all()

    def find_cached(self, user_id: str, query: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Find a cached answer for the same question (user-scoped)"""
        try:
            response = (
                self.supabase.table("chat_history")
                .select("answer, sources")
                .eq("user_id", user_id)
                .eq("question", query.strip())
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if response.data:
                row = response.data[0]
                return row["answer"], row["sources"] or []
        except Exception as e:
            print(f"⚠️ Supabase cache lookup failed: {e}")

        return None, None

    def clear(self, user_id: str) -> None:
        """Delete all chat history for a specific user"""
        try:
            self.supabase.table("chat_history").delete().eq("user_id", user_id).execute()
        except Exception as e:
            print(f"⚠️ Supabase chat_history clear failed: {e}")


# ============================================================
# SUPABASE DOCUMENT STORAGE (per-user)
# ============================================================

class SupabaseDocumentStorage:
    """Tracks uploaded documents per user in Supabase."""

    def __init__(self):
        from .auth.supabase_client import supabase_admin
        self.supabase = supabase_admin

    def add(self, user_id: str, filename: str, file_type: str, file_size: int) -> dict:
        """Record an uploaded document for the user"""
        try:
            self.supabase.table("user_documents").insert({
                "user_id": user_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": file_size
            }).execute()
        except Exception as e:
            print(f"⚠️ Supabase document insert failed: {e}")

        return {"filename": filename, "file_type": file_type, "file_size": file_size}

    def get_all(self, user_id: str) -> List[dict]:
        """Get all documents uploaded by a user"""
        try:
            response = (
                self.supabase.table("user_documents")
                .select("id, filename, file_type, file_size, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"⚠️ Supabase documents fetch failed: {e}")
            return []

    def delete(self, user_id: str, doc_id: int) -> None:
        """Delete a document record"""
        try:
            self.supabase.table("user_documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
        except Exception as e:
            print(f"⚠️ Supabase document delete failed: {e}")


# ============================================================
# GLOBAL INSTANCES
# ============================================================

# In-memory fallback (for backwards compatibility)
history_storage = HistoryStorage()

# Supabase persistent storage (per-user)
supabase_storage = SupabaseChatStorage()

# Supabase document storage (per-user)
document_storage = SupabaseDocumentStorage()

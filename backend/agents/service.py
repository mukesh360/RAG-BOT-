# ============================================================
# AGENT SERVICE — Supabase CRUD for agents
# ============================================================

from typing import List, Optional
from uuid import UUID


class AgentService:
    """
    Handles agent CRUD operations against the Supabase `agents` table.
    Uses supabase_admin (service role) because standard anon key + RLS
    would require the user JWT to be forwarded to the Supabase client.
    FastAPI already validates the JWT via get_current_user before calling
    these methods, so we control access manually via user_id checks.
    """

    def __init__(self):
        from ..auth.supabase_client import supabase_admin
        self.db = supabase_admin

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------
    def create(self, user_id: str, name: str, description: str) -> dict:
        """Create a new agent for the given user."""
        try:
            response = self.db.table("agents").insert({
                "user_id": user_id,
                "name": name.strip(),
                "description": description.strip()
            }).execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            raise RuntimeError(f"Failed to create agent: {e}")

    # ------------------------------------------------------------------
    # LIST
    # ------------------------------------------------------------------
    def list_for_user(self, user_id: str) -> List[dict]:
        """Return all agents owned by the user."""
        try:
            response = (
                self.db.table("agents")
                .select("id, name, description, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"⚠️ AgentService.list_for_user failed: {e}")
            return []

    # ------------------------------------------------------------------
    # GET (with ownership check)
    # ------------------------------------------------------------------
    def get(self, agent_id: str, user_id: str) -> Optional[dict]:
        """Get a single agent, verifying it belongs to user_id."""
        try:
            response = (
                self.db.table("agents")
                .select("id, name, description, created_at, user_id")
                .eq("id", agent_id)
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            print(f"⚠️ AgentService.get failed: {e}")
            return None

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------
    def delete(self, agent_id: str, user_id: str) -> bool:
        """Delete an agent and its associated RAG document metadata."""
        try:
            # Remove associated document records first
            self.db.table("rag_documents").delete().eq("agent_id", agent_id).eq("user_id", user_id).execute()
            # Delete the agent record
            response = (
                self.db.table("agents")
                .delete()
                .eq("id", agent_id)
                .eq("user_id", user_id)
                .execute()
            )
            return True
        except Exception as e:
            print(f"⚠️ AgentService.delete failed: {e}")
            return False


    # ------------------------------------------------------------------
    def add_rag_document(
        self,
        agent_id: str,
        user_id: str,
        file_name: str,
        embedding_collection: str
    ) -> dict:
        """Record a document that was indexed into Chroma for a given agent."""
        try:
            response = self.db.table("rag_documents").insert({
                "agent_id": agent_id,
                "user_id": user_id,
                "file_name": file_name,
                "embedding_collection": embedding_collection
            }).execute()
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"⚠️ AgentService.add_rag_document failed: {e}")
            return {}

    def list_rag_documents(self, agent_id: str, user_id: str) -> List[dict]:
        """List all indexed documents for a specific agent (with ownership check)."""
        try:
            response = (
                self.db.table("rag_documents")
                .select("id, file_name, embedding_collection, upload_time")
                .eq("agent_id", agent_id)
                .eq("user_id", user_id)
                .order("upload_time", desc=True)
                .execute()
            )
            return response.data or []
        except Exception as e:
            print(f"⚠️ AgentService.list_rag_documents failed: {e}")
            return []


# Singleton instance
agent_service = AgentService()

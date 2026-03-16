# ============================================================
# AGENT ROUTES — /agents endpoints
# ============================================================

import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from ..auth.dependencies import get_current_user
from ..schemas import AgentCreate, AgentOut, AgentQueryRequest, QueryResponse
from .service import agent_service
from ..rag.document_loader import index_file, query_agent, collection_name_for

router = APIRouter()

ALLOWED_EXTENSIONS = {".csv", ".pdf", ".docx", ".txt", ".html"}
TEMP_DIR = tempfile.mkdtemp(prefix="agent_uploads_")


# ============================================================
# CREATE AGENT
# ============================================================

@router.post("", response_model=AgentOut, status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreate, user: dict = Depends(get_current_user)):
    """Create a new agent (isolated RAG knowledge space) for the authenticated user."""
    if not payload.name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent name cannot be empty."
        )
    agent = agent_service.create(
        user_id=user["id"],
        name=payload.name,
        description=payload.description or ""
    )
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent."
        )
    return AgentOut(**agent)


# ============================================================
# LIST AGENTS
# ============================================================

@router.get("", response_model=List[AgentOut])
async def list_agents(user: dict = Depends(get_current_user)):
    """List all agents owned by the authenticated user."""
    return agent_service.list_for_user(user["id"])


# ============================================================
# UPLOAD DOCUMENTS TO AGENT
# ============================================================

@router.post("/{agent_id}/upload")
async def upload_to_agent(
    agent_id: str,
    files: List[UploadFile] = File(...),
    user: dict = Depends(get_current_user)
):
    """
    Upload and index files into a specific agent's Chroma collection.
    Files are chunked, embedded, and stored with agent_id/user_id metadata.
    """
    # Verify agent ownership
    agent = agent_service.get(agent_id=agent_id, user_id=user["id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have permission to access it."
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided."
        )

    processed = []
    failed = []
    collection = collection_name_for(agent_id)

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            failed.append({"file": file.filename, "reason": f"Unsupported format: {ext}"})
            continue

        # Save to temp
        safe_name = os.path.basename(file.filename)
        tmp_path = os.path.join(TEMP_DIR, f"{agent_id}_{safe_name}")
        try:
            content = await file.read()
            with open(tmp_path, "wb") as f:
                f.write(content)

            # Index into agent's Chroma collection
            chunk_count = index_file(
                file_path=tmp_path,
                agent_id=agent_id,
                user_id=user["id"]
            )

            # Record metadata in Supabase rag_documents table
            agent_service.add_rag_document(
                agent_id=agent_id,
                user_id=user["id"],
                file_name=file.filename,
                embedding_collection=collection
            )

            processed.append({
                "file": file.filename,
                "chunks": chunk_count,
                "collection": collection
            })

        except Exception as e:
            failed.append({"file": file.filename, "reason": str(e)})
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {
        "status": "success",
        "agent_id": agent_id,
        "agent_name": agent["name"],
        "files_processed": len(processed),
        "files_failed": len(failed),
        "processed": processed,
        "failed": failed,
    }


# ============================================================
# QUERY AN AGENT
# ============================================================

@router.post("/{agent_id}/query", response_model=QueryResponse)
async def query_agent_endpoint(
    agent_id: str,
    request: AgentQueryRequest,
    user: dict = Depends(get_current_user)
):
    """
    Query a specific agent's RAG knowledge space.
    Only returns answers from the agent's indexed documents (zero hallucination).
    """
    from datetime import datetime
    from ..auth.supabase_client import supabase_admin

    # Verify agent ownership
    agent = agent_service.get(agent_id=agent_id, user_id=user["id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have permission to access it."
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    # Retrieve from agent's vector store
    csv_docs, vector_docs = query_agent(agent_id, question, top_k=request.top_k or 4)
    all_docs = csv_docs + vector_docs

    if not all_docs:
        answer = "No relevant information found in this agent's knowledge base. Please upload documents first."
        sources = []
    else:
        # Build context
        context = "\n\n---\n\n".join(d.page_content for d in all_docs)
        sources = list(dict.fromkeys(
            d.metadata.get("source", "Unknown") for d in all_docs
        ))[:3]

        # Use LLM with strict context-only prompt
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        STRICT_PROMPT = """You are a helpful document assistant. Answer ONLY using the context below.

RULES:
1. Answer ONLY from the provided context — never use external knowledge
2. If the answer is not in the context, say: "No relevant information found in this agent's documents."
3. Quote or reference specific parts of the context when relevant
4. Be concise and factual

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

        llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)
        prompt = ChatPromptTemplate.from_template(STRICT_PROMPT)
        try:
            answer = (prompt | llm | StrOutputParser()).invoke({
                "context": context,
                "question": question
            })
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM error: {str(e)}"
            )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to chat_history
    try:
        supabase_admin.table("chat_history").insert({
            "user_id": user["id"],
            "agent_id": agent_id,
            "question": question,
            "answer": answer,
            "sources": sources
        }).execute()
    except Exception as e:
        print(f"⚠️ Failed to save chat history: {e}")

    return QueryResponse(answer=answer, sources=sources, timestamp=timestamp)


# ============================================================
# DELETE AGENT
# ============================================================

@router.delete("/{agent_id}", status_code=status.HTTP_200_OK)
async def delete_agent(
    agent_id: str,
    user: dict = Depends(get_current_user)
):
    """Delete an agent and its associated document metadata."""
    agent = agent_service.get(agent_id=agent_id, user_id=user["id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found or you don't have permission to delete it."
        )
    success = agent_service.delete(agent_id=agent_id, user_id=user["id"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete agent."
        )
    return {"status": "deleted", "agent_id": agent_id}


# ============================================================
# LIST AGENT DOCUMENTS
# ============================================================

@router.get("/{agent_id}/documents")
async def list_agent_documents(
    agent_id: str,
    user: dict = Depends(get_current_user)
):
    """List all indexed documents for a specific agent."""
    agent = agent_service.get(agent_id=agent_id, user_id=user["id"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found."
        )
    return agent_service.list_rag_documents(agent_id=agent_id, user_id=user["id"])

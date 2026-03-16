# DocIntel AI - Intelligent Multi-Agent RAG System

DocIntel AI is a next-generation document intelligence platform that leverages Retrieval-Augmented Generation (RAG) and an Agent-based architecture to provide accurate, context-aware answers from your documents. 

Built with a focus on precision and performance, DocIntel AI allows you to create specialized agents with isolated knowledge bases, ensuring that your queries are answered using the right context every time.

---

## 🚀 Key Features

- **Multi-Agent Architecture**: Create specialized agents (e.g., "Legal Assistant", "Study Guide", "Project X") for specific domains.
- **Isolated Knowledge Bases**: Each agent manages its own isolated vector repository, preventing cross-domain halluncinations.
- **Unified Document Upload**: Upload PDF, DOCX, CSV, TXT, and HTML files and index them directly to specific agents or a global knowledge base.
- **Zero Hallucination Retrieval**: Strict LLM prompting ensures answers are only derived from the provided document context.
- **Modern Glassmorphism UI**: A premium, interactive chat interface with real-time feedback and smooth animations.
- **Voice-to-Text Integration**: Hands-free interaction with integrated speech recognition.
- **Full History Tracking**: Persistent chat history stored via Supabase.

---

## 🏗️ System Architecture

DocIntel AI follows a modular architecture:

1.  **Backend**: FastAPI acts as the orchestrator, handling API requests, authentication, and communication with the RAG engine.
2.  **RAG Engine**: Powered by LangChain, it handles document loading, chunking, and embedding.
3.  **Vector Store**: Chroma DB is used for efficient semantic search within agent-specific collections.
4.  **Database**: Supabase manages user profiles, agent metadata, and chat history.
5.  **Frontend**: A modern SPA (Single Page Application) built with Vanilla JS, Bootstrap 5, and custom glassmorphism effects.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **RAG Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [Chroma DB](https://www.trychroma.com/)
- **LLM**: [Ollama](https://ollama.com/) (qwen2.5:7b)
- **Database / Auth**: [Supabase](https://supabase.com/)
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Bootstrap 5

---

## 📁 Project Structure

```bash
DocIntel-AI/
├── backend/
│   ├── agents/           # Agent management services and routes
│   ├── auth/             # Supabase authentication logic
│   ├── rag/              # Document processing and RAG pipeline
│   ├── schemas.py        # Pydantic data models
│   ├── storage.py        # Database interaction layer
│   └── main.py           # Application entry point
├── frontend/
│   ├── static/           # CSS, JS, and Assets
│   └── templates/        # HTML Jinja2 templates
├── base_documents/       # Initial/Global documents
├── chroma_agents/        # Persistent vector storage for agents
├── .env                  # Environment configuration
└── requirements.txt      # Python dependencies
```

---

## 🤖 How the Agent System Works

Each **Agent** in DocIntel AI represents a unique knowledge container. When you create an agent:
1.  A new metadata record is created in Supabase.
2.  A unique **Chroma Collection** is initialized for that agent.
3.  Documents indexed to that agent are isolated from others, allowing you to "switch brains" by selecting different agents in the chat sidebar.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running
- A [Supabase](https://supabase.com/) project

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/DocIntel-AI.git
cd DocIntel-AI
```

### 3. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Pull the LLM Model
```bash
ollama pull qwen2.5:7b
```

---

## 🔑 Environment Variables Setup

Create a `.env` file in the root directory and add your credentials:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
# Ollama and local settings
OLLAMA_BASE_URL=http://localhost:11434
PORT=8000
```

---

## 🚀 Running the Project Locally

Start the FastAPI server using Uvicorn:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the application at `http://localhost:8000`.

---

## 📄 Document Upload and Indexing

1.  **Global Index**: Documents placed in `base_documents/` are indexed on startup.
2.  **Agent Index**: Use the "Index to Agent" module in the chat sidebar.
    - Select an Agent.
    - Upload files (PDF, DOCX, etc.).
    - The backend chunks the files and stores embeddings in the agent's collection.

---

## 🧠 How the Chatbot Generates Answers

DocIntel AI uses a strict RAG pipeline:
1.  **Retrieval**: Semantic search finds the most relevant chunks in the active agent's collection.
2.  **Augmentation**: These chunks are injected into a strict system prompt.
3.  **Generation**: The LLM generates an answer based *only* on the provided context. If no context exists, it informs the user rather than hallucinating.

---

## 🔮 Future Improvements

- [ ] Support for more LLM providers (OpenAI, Anthropic).
- [ ] Advanced citation and source highlighting.
- [ ] Collaborative agents for shared knowledge bases.
- [ ] Integration with cloud storage (Google Drive, Dropbox).

---

© 2026 DocIntel AI. Intelligent Retrieval for the Modern Workspace.
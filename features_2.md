# 🤖 Agent System & RAG Workflow Guide

Welcome to the **DocIntel AI** learning guide! This document explains how the "brain" of our chatbot works. Whether you're a student, an offline learner, or just curious, this guide will break down complex AI concepts into simple, easy-to-understand steps.

---

### 1. What is an "Agent" in DocIntel AI?

Think of an **Agent** as a specialized "brain" or a digital assistant that only knows about the topics you give it. 

Instead of one giant AI that knows a little bit about everything, DocIntel AI lets you create multiple agents. Each agent has its own **isolated knowledge base** (a "Vector Database"). This means if you create a "Biology Agent" and a "History Agent," they will never mix up their facts!

---

### 2. How Agents Are Created

Creating an agent is as simple as filling out a form. Here is how it works:

1.  **User Action**: You go to the **Agent Management Page**.
2.  **Creation**: You type in a name (like "My Exam Prep") and a description.
3.  **Backend Processing**: When you click "Create," the application sends a request to our server:
    - `POST /agents`
4.  **Storage**: The server saves this agent into our database (Supabase) and prepares a special folder in the vector database just for this agent.
5.  **Result**: Your new agent immediately appears in the dropdown menu on the main chat page!

---

### 3. How Documents Are Uploaded to an Agent

Once you've created an agent, you need to give it "books" to read. This is called **Indexing**.

- **Select Agent**: You pick your agent from the dropdown.
- **Upload**: You drag and drop a PDF, Word doc, or Text file.
- **The Pipeline**:
    1. **Text Extraction**: The system reads the document and extracts all the words.
    2. **Chunking**: The system cuts the text into smaller, manageable pieces (like paragraphs).
    3. **Embedding**: The system turns those pieces of text into "numbers" (vectors) that represent the meaning.
    4. **Storage**: These numbers are saved in the agent's special collection in **Chroma DB**.

---

### 4. RAG Pipeline Explained (Step-by-Step)

**RAG** stands for **Retrieval-Augmented Generation**. Here is what happens when you ask a question:

1.  **User Question**: You ask: *"What is the mitochondria?"*
2.  **Embedding Creation**: The system turns your question into "numbers" (vectors).
3.  **Vector Search**: The system looks through the agent's collection in Chroma to find the chunks of text that have "numbers" most similar to your question.
4.  **Retrieval**: The system pulls out the most relevant 3-4 paragraphs from your uploaded documents.
5.  **LLM Context**: The system sends your question **AND** those retrieved paragraphs to the AI (LLM).
6.  **Final Answer**: The AI reads the paragraphs and answers your question *only* using that information.

---

### 5. How the Chatbot Answers Questions

The flow is designed to be accurate and safe:
- **Input**: User types a query.
- **Check**: The system identifies which Agent is currently active.
- **Search**: It searches *only* that agent's documents.
- **Drafting**: It writes a response based on the search results.
- **Output**: You get an answer with **Sources** listed so you can verify the information yourself.

---

### 6. Why Agent-based Knowledge is Useful

Using agents makes your life easier:
- **Multiple Domains**: Keep your "Work Documents" separate from your "Personal Notes."
- **Company Documents**: Create agents for "HR Policies," "Technical Manuals," or "Sales Data."
- **Exam Prep**: Create separate agents for each subject (Physics, Math, Chemistry).
- **Research**: Build a dedicated assistant for a specific research paper or project.

---

### 7. For Offline Learners & Students

DocIntel AI is a perfect companion for your studies!

- **Upload Your Syllabus**: Create an agent for your course and upload the syllabus and textbooks.
- **Instant Summaries**: Ask the agent to summarize long chapters or explain complex diagrams found in the text.
- **Practice Quizzes**: Ask: *"Based on these notes, give me 3 practice questions."*
- **Study Anywhere**: Since the indexing is local, your data stays private and specialized to your materials.

**Pro-Tip**: Use the **Voice-to-Text** button to "talk" to your documents and learn while you're taking notes!

---

*Happy Learning with DocIntel AI!*

-- ============================================================
-- SUPABASE SETUP — Run this in the Supabase SQL Editor
-- Dashboard → SQL Editor → New Query → Paste → Run
-- ============================================================

-- ============================================================
-- 1. PROFILES TABLE
-- Auto-linked to auth.users via trigger
-- ============================================================

CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Auto-create profile when a new user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email)
    VALUES (NEW.id, NEW.email);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop trigger if it exists, then recreate
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- 2. AGENTS TABLE
-- Each agent is an isolated RAG knowledge space per user
-- ============================================================

CREATE TABLE IF NOT EXISTS public.agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agents_user_id ON public.agents(user_id);

ALTER TABLE public.agents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own agents" ON public.agents;
CREATE POLICY "Users can view own agents"
    ON public.agents FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own agents" ON public.agents;
CREATE POLICY "Users can insert own agents"
    ON public.agents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own agents" ON public.agents;
CREATE POLICY "Users can delete own agents"
    ON public.agents FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================
-- 3. CHAT HISTORY TABLE
-- Stores per-user Q&A with sources (now includes optional agent_id)
-- ============================================================

CREATE TABLE IF NOT EXISTS public.chat_history (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES public.agents(id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Add agent_id column if upgrading from v2 (safe to run multiple times)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='chat_history' AND column_name='agent_id'
    ) THEN
        ALTER TABLE public.chat_history ADD COLUMN agent_id UUID REFERENCES public.agents(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Index for fast user-specific queries
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON public.chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON public.chat_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_agent_id ON public.chat_history(agent_id);

-- ============================================================
-- 4. ROW LEVEL SECURITY FOR CHAT HISTORY
-- ============================================================

ALTER TABLE public.chat_history ENABLE ROW LEVEL SECURITY;

-- PROFILES: Users can read and update only their own profile
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

-- CHAT HISTORY: Users can CRUD only their own messages
DROP POLICY IF EXISTS "Users can view own chat history" ON public.chat_history;
CREATE POLICY "Users can view own chat history"
    ON public.chat_history FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own chat history" ON public.chat_history;
CREATE POLICY "Users can insert own chat history"
    ON public.chat_history FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own chat history" ON public.chat_history;
CREATE POLICY "Users can delete own chat history"
    ON public.chat_history FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================
-- 5. RAG DOCUMENTS TABLE
-- Tracks documents indexed into Chroma per agent
-- ============================================================

CREATE TABLE IF NOT EXISTS public.rag_documents (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES public.agents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    embedding_collection TEXT NOT NULL,
    upload_time TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_agent_id ON public.rag_documents(agent_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_user_id ON public.rag_documents(user_id);

ALTER TABLE public.rag_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own rag_documents" ON public.rag_documents;
CREATE POLICY "Users can view own rag_documents"
    ON public.rag_documents FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own rag_documents" ON public.rag_documents;
CREATE POLICY "Users can insert own rag_documents"
    ON public.rag_documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own rag_documents" ON public.rag_documents;
CREATE POLICY "Users can delete own rag_documents"
    ON public.rag_documents FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================
-- 6. USER DOCUMENTS TABLE (base RAG uploads)
-- Tracks uploaded files per user (not agent-specific)
-- ============================================================

CREATE TABLE IF NOT EXISTS public.user_documents (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_type TEXT,
    file_size BIGINT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_documents_user_id ON public.user_documents(user_id);

ALTER TABLE public.user_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own documents" ON public.user_documents;
CREATE POLICY "Users can view own documents"
    ON public.user_documents FOR SELECT
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own documents" ON public.user_documents;
CREATE POLICY "Users can insert own documents"
    ON public.user_documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete own documents" ON public.user_documents;
CREATE POLICY "Users can delete own documents"
    ON public.user_documents FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================
-- 7. KNOWLEDGE IMPROVEMENTS TABLE
-- FAQ memory built by offline learning job
-- ============================================================

CREATE TABLE IF NOT EXISTS public.knowledge_improvements (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID REFERENCES public.agents(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_improvements_agent_id ON public.knowledge_improvements(agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_improvements_frequency ON public.knowledge_improvements(frequency DESC);

ALTER TABLE public.knowledge_improvements ENABLE ROW LEVEL SECURITY;

-- Service role (backend) manages this table — no user-facing read required
-- but allow authenticated users to view for transparency
DROP POLICY IF EXISTS "Authenticated users can view knowledge improvements" ON public.knowledge_improvements;
CREATE POLICY "Authenticated users can view knowledge improvements"
    ON public.knowledge_improvements FOR SELECT
    TO authenticated
    USING (true);

-- ============================================================
-- DONE! All tables + RLS are ready.
-- ============================================================

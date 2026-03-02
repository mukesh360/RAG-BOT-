# ============================================================
# SUPABASE CLIENT INITIALIZATION
# ============================================================

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load .env from project root
load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment. "
        "Create a .env file in the project root with these values."
    )

# For AUTH operations (login/signup) — uses anon key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# For DATABASE operations (chat_history, user_documents) — uses service role key
# Service role key bypasses RLS. This is safe because FastAPI's get_current_user
# dependency already validates the JWT, and we manually pass user_id in all queries.
if SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
else:
    # Fallback to anon key (RLS may block operations)
    print("⚠️ SUPABASE_SERVICE_ROLE_KEY not set. Database operations may fail due to RLS.")
    supabase_admin = supabase
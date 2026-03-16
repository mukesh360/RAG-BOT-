# 🔐 Updates — DocIntel AI v2.1 (Supabase Auth)

New authentication layer added using **Supabase Auth** (email + password). RAG logic and retrieval system remain **completely unchanged**.

---

## ✨ New Features

### 1. User Authentication (Supabase)

| Feature | Description |
|---------|-------------|
| **Signup** | `POST /auth/signup` — Register with email + password |
| **Login** | `POST /auth/login` — Authenticate and receive JWT access token |
| **Logout** | `POST /auth/logout` — Server-side logout endpoint |
| **Get Current User** | `GET /auth/me` — Returns authenticated user info from JWT |

### 2. Protected API Endpoints

All core endpoints now require a valid JWT token in the `Authorization: Bearer <token>` header:

| Endpoint | Protection |
|----------|------------|
| `POST /query` | 🔒 Requires login |
| `POST /upload` | 🔒 Requires login |
| `GET /history` | 🔒 Requires login |
| `POST /history/clear` | 🔒 Requires login |
| `GET /config` | 🔒 Requires login |
| `POST /config` | 🔒 Requires login |
| `GET /health` | 🌐 Public |
| `GET /`, `/login`, `/signup` | 🌐 Public (pages) |

### 3. Login & Signup Pages

- **`/login`** — Dark-themed login page with glassmorphism card, animated gradient orbs, email + password form
- **`/signup`** — Matching signup page with client-side password confirmation validation
- Both pages use **Bootstrap 5** + **Bootstrap Icons** + Inter font
- Auto-redirect to chat after successful login

### 4. Frontend Auth System (`auth.js`)

| Function | Purpose |
|----------|---------|
| `getToken()` / `setToken()` / `clearToken()` | JWT stored in `localStorage` |
| `isLoggedIn()` | Check if user has a valid token |
| `handleLogin(email, password)` | Login → store token → redirect to `/` |
| `handleSignup(email, password)` | Register new account |
| `handleLogout()` | Clear token → redirect to `/login` |
| `authFetch(url, options)` | Drop-in `fetch()` replacement that auto-injects JWT header |

### 5. Auth Guard & Logout Button

- **Auth guard** on main chat page — redirects unauthenticated users to `/login`
- **Logout button** added to header (red-tinted, right side) — clears token and redirects to login
- **Global 401 handler** — if any API call returns 401, user is auto-redirected to `/login`

---

## 📁 New Files

```
backend/auth/
├── __init__.py           # Package init
├── supabase_client.py    # Supabase client singleton (from .env)
├── dependencies.py       # get_current_user() JWT validation dependency
└── routes.py             # /auth/signup, /login, /logout, /me endpoints

frontend/
├── templates/
│   ├── login.html        # Login page (dark theme)
│   └── signup.html       # Signup page (dark theme)
└── static/js/
    └── auth.js           # Token management + authFetch wrapper

.env.example              # Template for Supabase credentials
```

## 📝 Modified Files

| File | Changes |
|------|---------|
| `backend/main.py` | Added auth router, `/login` + `/signup` routes, protected all API endpoints with `get_current_user` |
| `backend/schemas.py` | Added `AuthRequest` and `AuthResponse` Pydantic models |
| `frontend/static/js/app.js` | All `fetch()` → `authFetch()`, auth guard on page load |
| `frontend/templates/index.html` | Added `auth.js` script tag, logout button in header |
| `requirements.txt` | Added `supabase`, `python-dotenv` |

## 🚫 NOT Modified

- `backend/rag_core.py` — RAG logic untouched
- `backend/storage.py` — No changes
- `frontend/static/css/style.css` — No changes

---

## ⚙️ Setup

```bash
# 1. Add Supabase credentials
cp .env.example .env
# Edit .env with your SUPABASE_URL and SUPABASE_ANON_KEY

# 2. Install new dependencies
pip install supabase python-dotenv

# 3. Run
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔧 Tech Added

| Package | Purpose |
|---------|---------|
| `supabase` | Official Supabase Python client for auth |
| `python-dotenv` | Load `.env` variables at startup |

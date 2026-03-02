// ============================================================
// AUTH HELPER — Token Management & Auth Fetch Wrapper
// ============================================================

const TOKEN_KEY = 'access_token';
const USER_EMAIL_KEY = 'user_email';

// ============================================================
// TOKEN MANAGEMENT
// ============================================================

function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_EMAIL_KEY);
}

function isLoggedIn() {
    return !!getToken();
}

function getUserEmail() {
    return localStorage.getItem(USER_EMAIL_KEY) || '';
}

function setUserEmail(email) {
    localStorage.setItem(USER_EMAIL_KEY, email);
}

// ============================================================
// AUTH ACTIONS
// ============================================================

/**
 * Login with email and password.
 * On success: stores token and redirects to main page.
 * On failure: throws an error with message.
 */
async function handleLogin(email, password) {
    const response = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
    }

    setToken(data.access_token);
    setUserEmail(data.user_email);
    window.location.href = '/';
}

/**
 * Signup with email and password.
 * On success: returns (token may be empty if email confirmation required).
 * On failure: throws an error with message.
 */
async function handleSignup(email, password) {
    const response = await fetch('/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Signup failed');
    }

    // If token is returned (no email confirmation), auto-login
    if (data.access_token) {
        setToken(data.access_token);
        setUserEmail(data.user_email);
    }

    return data;
}

/**
 * Logout: clear tokens and redirect to login.
 */
function handleLogout() {
    // Notify server (best-effort)
    fetch('/auth/logout', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + getToken() }
    }).catch(() => { });

    clearToken();
    window.location.href = '/login';
}

// ============================================================
// AUTH FETCH WRAPPER
// ============================================================

/**
 * Wrapper around fetch() that auto-injects the Authorization header.
 * On 401 response, redirects to /login.
 */
async function authFetch(url, options = {}) {
    const token = getToken();

    // Merge auth header
    const headers = options.headers || {};

    // Only set Content-Type if not FormData (browser sets it automatically for FormData)
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = headers['Content-Type'] || 'application/json';
    }

    if (token) {
        headers['Authorization'] = 'Bearer ' + token;
    }

    options.headers = headers;

    const response = await fetch(url, options);

    // Handle 401 — token expired or invalid
    if (response.status === 401) {
        clearToken();
        window.location.href = '/login';
        throw new Error('Session expired. Please log in again.');
    }

    return response;
}

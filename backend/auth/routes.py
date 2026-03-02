# ============================================================
# AUTH ROUTES — SIGNUP / LOGIN / LOGOUT / ME
# ============================================================

from fastapi import APIRouter, Depends, HTTPException, status
from ..schemas import AuthRequest, AuthResponse
from .supabase_client import supabase
from .dependencies import get_current_user

router = APIRouter()


@router.post("/signup", response_model=AuthResponse)
async def signup(request: AuthRequest):
    """Register a new user with email and password"""
    try:
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password
        })

        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup failed. Please check your email and password."
            )

        # Some Supabase configs require email confirmation
        # In that case, session may be None
        access_token = ""
        if response.session:
            access_token = response.session.access_token

        return AuthResponse(
            access_token=access_token,
            user_email=response.user.email or request.email,
            message="Signup successful! Please check your email to confirm your account."
            if not access_token
            else "Signup successful!"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signup failed: {str(e)}"
        )


@router.post("/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Authenticate user and return access token"""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })

        if not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login failed. Check your credentials."
            )

        return AuthResponse(
            access_token=response.session.access_token,
            user_email=response.user.email or request.email,
            message="Login successful!"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Login failed: {str(e)}"
        )


@router.post("/logout")
async def logout():
    """
    Logout endpoint.
    Primary logout is client-side (clear localStorage token).
    This endpoint exists for API completeness.
    """
    return {"status": "success", "message": "Logged out. Clear token on client."}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Get current authenticated user info"""
    return {
        "status": "authenticated",
        "user": user
    }

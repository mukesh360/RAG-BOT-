# ============================================================
# AUTH DEPENDENCY — JWT VALIDATION
# ============================================================

from fastapi import Header, HTTPException, status
from .supabase_client import supabase


async def get_current_user(authorization: str = Header(...)):
    """
    Validate Supabase JWT from the Authorization header.
    Expected format: "Bearer <access_token>"
    Returns the authenticated user dict.
    """
    # Extract token from "Bearer <token>"
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use: Bearer <token>"
        )

    token = authorization.removeprefix("Bearer ").strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing"
        )

    try:
        # Validate token with Supabase — this verifies expiration and signature
        user_response = supabase.auth.get_user(token)
        user = user_response.user

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        return {
            "id": user.id,
            "email": user.email,
            "created_at": str(user.created_at) if user.created_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}"
        )

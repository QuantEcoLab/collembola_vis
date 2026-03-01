"""Authentication: JWT login with role-based access control."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_DEFAULT_SECRET_KEY = "collembola-vis-secret-key-change-in-production-2024"
_SECRET_KEY = os.environ.get("AUTH_SECRET_KEY", _DEFAULT_SECRET_KEY)
_ALGORITHM = "HS256"
_TOKEN_EXPIRE_DAYS = 7

if _SECRET_KEY == _DEFAULT_SECRET_KEY:
    logger.warning("WARNING: using default AUTH_SECRET_KEY — set AUTH_SECRET_KEY env var in production")

# ── Hardcoded users (password stored as bcrypt hash) ─────────────────────────
# user1 / user12345  → role: user
# user2 / user23456  → role: user
# user3 / user34567  → role: user
# admin / admin12345 → role: admin
_DEFAULT_USERS: dict[str, dict] = {
    "user1": {
        "hash": b"$2b$12$YAIbQEfmn5d0IXDdGg7QzOUzrMYknVwyYuXIwrauZMrMn9NApc6Si",
        "role": "user",
    },
    "user2": {
        "hash": b"$2b$12$A9FMSIIuhm9E8KjTDd62H.J0kViN5liOw1Qte62lk5OhZYdEWGM8G",
        "role": "user",
    },
    "user3": {
        "hash": b"$2b$12$0z11MEVFPwv2CwOvgBwUCutwo4DmRlb0TDMDAoGiFBSMSTRMhV42u",
        "role": "user",
    },
    "admin": {
        "hash": b"$2b$12$AWueC8rp4LcYDNbNNJt5E.Hs/s5YKO1UWfc9zAsbiolCySJcXZsfG",
        "role": "admin",
    },
}

_auth_users_env = os.environ.get("AUTH_USERS")
if _auth_users_env:
    _loaded = json.loads(_auth_users_env)
    # Ensure hash values are bytes (JSON stores them as strings)
    _USERS: dict[str, dict] = {
        u: {**d, "hash": d["hash"].encode() if isinstance(d["hash"], str) else d["hash"]}
        for u, d in _loaded.items()
    }
else:
    _USERS = _DEFAULT_USERS


def _verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)


# ── OAuth2 scheme (reads Bearer token from Authorization header) ──────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Helpers ───────────────────────────────────────────────────────────────────
def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": username, "role": role, "exp": expire},
        _SECRET_KEY,
        algorithm=_ALGORITHM,
    )


def verify_token(token: str) -> dict:
    """Decode JWT and return {username, role}, or raise 401."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        username: str | None = payload.get("sub")
        if not username or username not in _USERS:
            raise credentials_exc
        role: str = payload.get("role", _USERS[username]["role"])
        return {"username": username, "role": role}
    except JWTError:
        raise credentials_exc


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Return the current username (any authenticated user)."""
    return verify_token(token)["username"]


async def require_admin(token: str = Depends(oauth2_scheme)) -> str:
    """Return username if the current user has the admin role, else raise 403."""
    info = verify_token(token)
    if info["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return info["username"]


# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/auth", tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Exchange username + password for a JWT access token."""
    user = _USERS.get(form.username)
    if not user or not _verify_password(form.password, user["hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form.username, user["role"])
    return Token(access_token=token, token_type="bearer")


@router.get("/me")
async def me(token: str = Depends(oauth2_scheme)):
    info = verify_token(token)
    return {"username": info["username"], "role": info["role"]}

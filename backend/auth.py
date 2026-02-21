"""Authentication: JWT login for a single hardcoded user."""

from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
_SECRET_KEY = "collembola-vis-secret-key-change-in-production-2024"
_ALGORITHM = "HS256"
_TOKEN_EXPIRE_DAYS = 7

# ── Hardcoded users (password stored as bcrypt hash) ─────────────────────────
# Username: user1, Password: user12345
_USERS: dict[str, bytes] = {
    "user1": b"$2b$12$YAIbQEfmn5d0IXDdGg7QzOUzrMYknVwyYuXIwrauZMrMn9NApc6Si",
}


def _verify_password(plain: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed)


# ── OAuth2 scheme (reads Bearer token from Authorization header) ──────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Helpers ───────────────────────────────────────────────────────────────────
def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": username, "exp": expire}, _SECRET_KEY, algorithm=_ALGORITHM)


def verify_token(token: str) -> str:
    """Decode JWT and return username, or raise 401."""
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
        return username
    except JWTError:
        raise credentials_exc


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    return verify_token(token)


# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api/auth", tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Exchange username + password for a JWT access token."""
    hashed = _USERS.get(form.username)
    if not hashed or not _verify_password(form.password, hashed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(form.username), token_type="bearer")


@router.get("/me")
async def me(username: str = Depends(get_current_user)):
    return {"username": username}

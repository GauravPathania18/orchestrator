from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.core.auth import verify_password, create_access_token, decode_access_token, get_password_hash
from pydantic import BaseModel
import os

router = APIRouter()

# Simple mock user for demo (In production, use a database)
# Default: admin / admin123
MOCK_USER = {
    "username": os.getenv("ADMIN_USERNAME", "admin"),
    "hashed_password": get_password_hash(os.getenv("ADMIN_PASSWORD", "admin123"))
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

class Token(BaseModel):
    access_token: str
    token_type: str

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None or username != MOCK_USER["username"]:
        raise credentials_exception
    return username

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != MOCK_USER["username"] or not verify_password(form_data.password, MOCK_USER["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

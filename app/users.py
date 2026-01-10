import uuid
import os
import logging
from typing import Optional
from fastapi import Depends , Request
from fastapi_users import BaseUserManager , FastAPIUsers , UUIDIDMixin , models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy
)
from dotenv import load_dotenv
from fastapi_users.db import SQLAlchemyUserDatabase
from app.db import User , get_user_db
import pathlib

logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# FIX: Move hardcoded secret to environment variable for security
# In production, this must be set via environment variable
SECRET = os.getenv("SECRET_KEY")

if not SECRET:
    raise RuntimeError("SECRET_KEY is missing. Set it as environment variable.")

class UserManager(UUIDIDMixin , BaseUserManager[User , uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user:User , request:Optional[Request] = None):
        # FIX: Use logging instead of print for production-ready logging
        logger.info(f"User {user.id} ({user.email}) has registered.")

    async def on_after_forget_password(self , user:User , token:str , request:Optional[Request] = None):
        # FIX: Log password reset requests (token should be sent via email in production)
        logger.info(f"Password reset requested for user {user.id} ({user.email})")

    async def on_after_request_verify(self , user:User , token:str , request: Optional[Request] = None):
        # FIX: Log verification requests (token should be sent via email in production)
        logger.info(f"Email verification requested for user {user.id} ({user.email})")

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

# FIX: tokenUrl should match the actual route path for OpenAPI documentation
# The route is registered with prefix="/api/auth/jwt" in main.py
# So the full path is "/api/auth/jwt/login"
bearer_transport = BearerTransport(tokenUrl="/api/auth/jwt/login")

def get_jwt_strategy():
    return JWTStrategy(secret=SECRET , lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name = "jwt",
    transport = bearer_transport,
    get_strategy = get_jwt_strategy,
)

fastapi_users = FastAPIUsers(
    get_user_manager,
    [auth_backend],
)
current_active_user = fastapi_users.current_user(active=True)


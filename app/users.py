import re
import uuid
import os
import logging
from typing import Optional
from fastapi import Depends, Request, HTTPException, status
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, models, exceptions
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy
)
from dotenv import load_dotenv
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.exc import IntegrityError
from app.db import User, get_user_db
from app.schemas import UserCreate
import pathlib

logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


SECRET = os.getenv("SECRET_KEY")

if not SECRET:
    raise RuntimeError("SECRET_KEY is missing. Set it as environment variable.")


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def validate_password(self, password: str, user) -> None:
        if len(password) < 6:
            raise exceptions.InvalidPasswordException(
                reason="Password must be at least 6 characters."
            )
        if len(password) > 128:
            raise exceptions.InvalidPasswordException(
                reason="Password must be at most 128 characters."
            )

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} (@{user.username}, {user.email}) has registered.")

    async def create(self, user_create: UserCreate, safe: bool = False, request: Optional[Request] = None) -> User:
        # Validate username format
        if not re.match(r'^[a-zA-Z0-9_]+$', user_create.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username can only contain letters, numbers, and underscores."
            )
        try:
            return await super().create(user_create, safe=safe, request=request)
        except IntegrityError as e:
            error_str = str(e.orig).lower()
            if "username" in error_str:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken."
                )
            elif "email" in error_str:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A user with this email already exists."
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Registration failed due to a conflict. Please try a different username or email."
            )

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None):
        logger.info(f"Password reset requested for user {user.id} ({user.email})")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None):
        logger.info(f"Email verification requested for user {user.id} ({user.email})")


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

bearer_transport = BearerTransport(tokenUrl="/api/auth/jwt/login")

def get_jwt_strategy():
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers(
    get_user_manager,
    [auth_backend],
)
current_active_user = fastapi_users.current_user(active=True)

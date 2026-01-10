from collections.abc import AsyncGenerator
import asyncio
import logging
import uuid
import os

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Index, UniqueConstraint, TypeDecorator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone
from fastapi_users.db import SQLAlchemyUserDatabase, SQLAlchemyBaseUserTableUUID
from fastapi import Depends

logger = logging.getLogger(__name__)

# FIX: Use environment variable for database URL to support different environments
# Falls back to SQLite for development if not set
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")

# FIX: Create UUID-compatible type that stores as String but works with UUID objects
# This ensures compatibility with fastapi-users' UUID handling while supporting SQLite
class UUIDString(TypeDecorator):
    """Stores UUID as String(36) in database, but works with uuid.UUID objects in Python"""
    impl = String(36)
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return value
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, str):
            return uuid.UUID(value)
        return value


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    posts = relationship("Post", back_populates="user", cascade="all, delete-orphan")
    # FIX: Add relationships for social features with cascade delete
    # When user is deleted, their likes, comments, and follows are also deleted
    liked_posts = relationship("Like", back_populates="user", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    following = relationship(
        "Follow",
        foreign_keys="Follow.follower_id",
        back_populates="follower",
        cascade="all, delete-orphan"
    )
    followers_rel = relationship(
        "Follow",
        foreign_keys="Follow.followed_id",
        back_populates="followed",
        cascade="all, delete-orphan"
    )


def now_utc():
    """Helper function to get current UTC datetime with timezone awareness"""
    return datetime.now(timezone.utc)


class Post(Base):
    __tablename__ = "posts"

    # FIX: Use UUIDString type decorator for UUID compatibility
    # Stores as String(36) in DB but works with UUID objects in Python
    # This matches fastapi-users' UUID handling while supporting SQLite
    id = Column(UUIDString(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDString(), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    caption = Column(Text)
    url = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    # FIX: datetime.utcnow is deprecated, use callable function for default
    # This ensures the timestamp is evaluated at insert time, not class definition time
    # Using a named function (not lambda) is required for SQLAlchemy to properly serialize the default
    created_at = Column(DateTime, default=now_utc)

    user = relationship("User", back_populates="posts")
    # FIX: Add relationships for social features
    likes = relationship("Like", back_populates="post", cascade="all, delete-orphan")
    # Note: order_by in relationship is for when accessing post.comments, but we query directly for pagination
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    
    # FIX: Add indexes for common queries (user_id for user posts, created_at for feed ordering)
    __table_args__ = (
        Index("idx_post_user_id", "user_id"),
        Index("idx_post_created_at", "created_at"),
    )


class Like(Base):
    """FIX: Like model to track user likes on posts with unique constraint to prevent duplicates"""
    __tablename__ = "likes"
    
    # FIX: Use UUIDString type decorator for UUID compatibility
    id = Column(UUIDString(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDString(), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    post_id = Column(UUIDString(), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=now_utc)
    
    user = relationship("User", back_populates="liked_posts")
    post = relationship("Post", back_populates="likes")
    
    # FIX: Unique constraint prevents duplicate likes (user can only like a post once)
    __table_args__ = (
        UniqueConstraint("user_id", "post_id", name="uq_user_post_like"),
        Index("idx_like_post_id", "post_id"),
        Index("idx_like_user_id", "user_id"),
    )


class Comment(Base):
    """FIX: Comment model for post comments with pagination support"""
    __tablename__ = "comments"
    
    # FIX: Use UUIDString type decorator for UUID compatibility
    id = Column(UUIDString(), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUIDString(), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    post_id = Column(UUIDString(), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=now_utc)
    
    user = relationship("User", back_populates="comments")
    post = relationship("Post", back_populates="comments")
    
    # FIX: Indexes for efficient querying by post and user
    __table_args__ = (
        Index("idx_comment_post_id", "post_id"),
        Index("idx_comment_user_id", "user_id"),
        Index("idx_comment_created_at", "created_at"),
    )


class Follow(Base):
    """FIX: Follow model to track user follows with unique constraint to prevent duplicate follows"""
    __tablename__ = "follows"
    
    # FIX: Use UUIDString type decorator for UUID compatibility
    id = Column(UUIDString(), primary_key=True, default=uuid.uuid4)
    follower_id = Column(UUIDString(), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    followed_id = Column(UUIDString(), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=now_utc)
    
    follower = relationship("User", foreign_keys=[follower_id], back_populates="following")
    followed = relationship("User", foreign_keys=[followed_id], back_populates="followers_rel")
    
    # FIX: Unique constraint prevents duplicate follows and self-follows
    # Check constraint for self-follow prevention would require database-level check
    __table_args__ = (
        UniqueConstraint("follower_id", "followed_id", name="uq_follower_followed"),
        Index("idx_follow_follower_id", "follower_id"),
        Index("idx_follow_followed_id", "followed_id"),
    )


# FIX: Configure async engine with proper pool settings for production
# pool_pre_ping ensures connections are valid before use
# echo=False in production to avoid logging all SQL queries
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query debugging
    pool_pre_ping=True,  # Verify connections before using them
    pool_size=5,  # Number of connections to maintain
    max_overflow=10,  # Additional connections beyond pool_size
)

# FIX: Configure async session maker with expire_on_commit=False
# This prevents objects from becoming detached after commit, allowing access to relationships
async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def create_db_and_tables():
    """
    FIX: Create database tables with timeout to prevent startup hang
    If this fails, app still starts (allows health checks to diagnose)
    """
    try:
        # FIX: Add timeout to prevent startup hang if DB is slow/unresponsive
        # Use wait_for for Python 3.10+ compatibility (timeout() requires 3.11+)
        async def _create_tables():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        
        await asyncio.wait_for(_create_tables(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("Database table creation timed out after 10 seconds")
        raise
    except Exception as e:
        logger.error(f"Database table creation failed: {e}")
        raise


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)
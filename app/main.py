"""
Production-ready FastAPI application for social media backend.

This module implements a RESTful API for managing posts, file uploads,
and user authentication with comprehensive error handling, logging,
validation, and security best practices.
"""

import asyncio
import logging
import logging.handlers
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    HTTPException,
    File,
    UploadFile,
    Form,
    Depends,
    Query,
    status,
    Request,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, desc, delete, and_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.schemas import (
    PostResponse,
    PostListResponse,
    UploadResponse,
    DeleteResponse,
    ErrorResponse,
    HealthResponse,
    PostCreate,
    UserRead,
    UserCreate,
    UserUpdate,
    CommentCreate,
    CommentResponse,
    CommentListResponse,
    LikeResponse,
    FollowResponse,
    UserProfileResponse,
)
from app.db import Post, create_db_and_tables, get_async_session, User, Like, Comment, Follow
from app.users import auth_backend, current_active_user, fastapi_users
from app.images import imagekit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

# FIX: Configure logging with rotation to prevent unbounded file growth
# RotatingFileHandler automatically rotates logs when they reach maxBytes
# This prevents disk space exhaustion in production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files (50MB total max)
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}
ALLOWED_FILE_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    FIX: Non-blocking startup - DB table creation runs in background
    App starts immediately even if DB setup fails (allows health checks to work)
    """
    logger.info("Starting application...")
    
    # FIX: Run DB table creation in background to avoid blocking startup
    # If it fails, app still starts (allows /health endpoint to diagnose issues)
    async def init_db():
        try:
            await create_db_and_tables()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}. App will continue but DB operations may fail.")
    
    # Start DB initialization but don't wait for it - app starts immediately
    asyncio.create_task(init_db())
    
    yield
    
    logger.info("Shutting down application...")


# Initialize FastAPI application
app = FastAPI(
    title="Social Media Backend API",
    description="Production-ready REST API for social media platform with authentication, file uploads, and feed management",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions"""
    logger.warning(f"ValueError on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "status_code": status.HTTP_400_BAD_REQUEST,
        },
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    """Handle database errors"""
    logger.error(f"Database error on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Database Error",
            "detail": "An internal database error occurred. Please try again later.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error on {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


# Include authentication routers
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/api/auth/jwt",
    tags=["Authentication"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/api/auth",
    tags=["Authentication"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/api/auth",
    tags=["Authentication"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/api/auth",
    tags=["Authentication"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/api/users",
    tags=["Users"],
)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health check",
    description="Check the health status of the API and database",
)
async def health_check(session: AsyncSession = Depends(get_async_session)):
    """
    Health check endpoint to verify API and database status.
    
    FIX: Non-blocking health check with timeout - responds immediately even if DB is slow
    """
    try:
        # FIX: Add timeout to prevent health check from hanging if DB is slow/unresponsive
        # This ensures /health endpoint always responds quickly
        await asyncio.wait_for(session.execute(select(1)), timeout=2.0)
        db_status = "connected"
    except asyncio.TimeoutError:
        logger.warning("Database health check timed out")
        db_status = "timeout"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    # FIX: Use timezone-aware datetime (datetime.utcnow is deprecated)
    # This ensures consistent timestamp handling across different environments
    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        version=APP_VERSION,
        database=db_status,
        timestamp=datetime.now(timezone.utc),
    )


# Helper functions
def validate_file_size(file: UploadFile) -> None:
    """Validate file size before processing"""
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )


def validate_file_type(content_type: str | None) -> None:
    """Validate file MIME type"""
    # FIX: Handle None content_type to prevent runtime errors
    # Some clients may not send Content-Type header
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content type is required. Please specify Content-Type header.",
        )
    if content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{content_type}' not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}",
        )


def get_file_type(content_type: str | None) -> str:
    """Determine file type category (image or video)"""
    # FIX: Handle None content_type to prevent AttributeError
    if not content_type:
        return "unknown"
    if content_type.startswith("image/"):
        return "image"
    elif content_type.startswith("video/"):
        return "video"
    return "unknown"


# API Endpoints
@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Posts"],
    summary="Upload a file",
    description="Upload an image or video file with an optional caption. Maximum file size: 50MB",
)
async def upload_file(
    file: Annotated[UploadFile, File(..., description="Image or video file to upload")],
    caption: str = Form(default="", max_length=2000, description="Optional caption for the post"),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Upload an image or video file to create a new post.
    
    Args:
        file: The image or video file to upload
        caption: Optional caption text (max 2000 characters)
        user: Authenticated user (from dependency)
        session: Database session (from dependency)
    
    Returns:
        UploadResponse: Created post information
    
    Raises:
        HTTPException: If file validation fails or upload errors occur
    """
    temp_file_path = None
    temp_file_handle = None
    
    try:
        # Validate file type before processing
        validate_file_type(file.content_type)
        
        # FIX: Stream file to disk instead of loading entire file into memory
        # This prevents memory exhaustion with large files and is production-ready
        # FIX: Use NamedTemporaryFile instead of mktemp() to avoid race conditions
        # mktemp() creates a security vulnerability due to time-of-check-time-of-use (TOCTOU) race condition
        # NamedTemporaryFile creates and opens the file atomically, eliminating the race condition
        file_extension = Path(file.filename).suffix if file.filename else ""
        temp_file_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,  # Don't auto-delete, we'll clean up manually in finally block
            suffix=file_extension,
            prefix="upload_",
        )
        temp_file_path = temp_file_handle.name
        
        file_size = 0
        chunk_size = 8192  # 8KB chunks for efficient streaming
        
        # Reset file pointer (FastAPI may have read it already for validation)
        # FIX: Handle case where seek might not be supported (shouldn't happen with FastAPI UploadFile)
        try:
            await file.seek(0)
        except (AttributeError, OSError) as e:
            logger.warning(f"Could not seek file: {e}. File may have been consumed.")
            # If seek fails, we can't re-read, but validation should have already read it
            # This is a rare edge case, but we handle it gracefully
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File could not be processed. Please try uploading again.",
            )
        
        # FIX: Stream file in chunks - file.read() is async, but file.write() blocks
        # Run file write operations in thread pool to avoid blocking event loop
        def write_chunk(chunk_data):
            """Helper function to write chunk synchronously in thread pool"""
            temp_file_handle.write(chunk_data)
        
        # Stream file in chunks to temp file while tracking size
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            
            # Check size during streaming to fail fast
            if file_size > MAX_FILE_SIZE:
                temp_file_handle.close()
                temp_file_handle = None  # Mark as closed to avoid double-close in finally
                os.unlink(temp_file_path)
                temp_file_path = None  # Mark as cleaned up to avoid double-unlink in finally
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f}MB",
                )
            
            # FIX: Run blocking file write in thread pool to avoid blocking event loop
            await run_in_threadpool(write_chunk, chunk)
        
        temp_file_handle.close()
        temp_file_handle = None
        
        if file_size == 0:
            # Clean up empty temp file before raising exception
            os.unlink(temp_file_path)
            temp_file_path = None
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty",
            )
        
        # Upload to ImageKit
        logger.info(f"Uploading file {file.filename} ({file_size / (1024*1024):.2f}MB) for user {user.id}")
        
        # FIX: ImageKit SDK is synchronous and blocks event loop - run in thread pool
        # This prevents blocking all other requests while file uploads to ImageKit
        def upload_to_imagekit(file_path: str, filename: str, content_type: str):
            """Synchronous ImageKit upload function to run in thread pool"""
            with open(file_path, "rb") as temp_file:
                return imagekit.upload_file(
                    file=temp_file,
                    file_name=filename,
                    options=UploadFileRequestOptions(
                        use_unique_file_name=True,
                        tags=["backend-upload", get_file_type(content_type)],
                    ),
                )
        
        try:
            # FIX: Run blocking ImageKit upload in thread pool to avoid blocking event loop
            upload_result = await run_in_threadpool(
                upload_to_imagekit,
                temp_file_path,
                file.filename or f"upload_{uuid.uuid4()}{file_extension}",
                file.content_type or "",
            )
        except Exception as e:
            logger.error(f"ImageKit upload failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="File upload to storage service failed. Please try again later.",
            )
        
        # Validate upload result - check for url attribute which indicates success
        if not hasattr(upload_result, "url") or not upload_result.url:
            logger.error(f"ImageKit upload failed: {upload_result}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="File upload to storage service failed",
            )
        
        # Create post record
        file_name = (
            getattr(upload_result, "name", None)
            or getattr(upload_result, "file_name", None)
            or file.filename
            or f"upload_{uuid.uuid4()}{file_extension}"
        )
        
        post = Post(
            user_id=user.id,
            caption=caption.strip() if caption else None,
            url=upload_result.url,
            file_type=get_file_type(file.content_type),
            file_name=file_name,
        )
        
        # FIX: Add post to session and commit within try-except to handle DB errors
        # If commit fails after ImageKit upload, the file remains in ImageKit but isn't referenced in DB
        # Deleting from ImageKit would require storing file_id, adding complexity and potential for
        # additional failures. Orphaned files in ImageKit are acceptable - they can be cleaned up
        # periodically via ImageKit's admin interface or a separate cleanup job if needed.
        session.add(post)
        try:
            await session.commit()
            await session.refresh(post)
        except SQLAlchemyError as db_error:
            logger.error(
                f"Database error after successful ImageKit upload for user {user.id}: {db_error}. "
                f"ImageKit URL: {upload_result.url}",
                exc_info=True
            )
            await session.rollback()
            # Log the ImageKit URL for potential manual cleanup if needed
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File uploaded but failed to save post. Please try again.",
            )
        
        logger.info(f"Post created successfully: {post.id} for user {user.id}")
        
        # FIX: Return response with initial counts (0 for new post)
        return UploadResponse(
            id=post.id,
            user_id=post.user_id,
            caption=post.caption,
            url=post.url,
            file_type=post.file_type,
            file_name=post.file_name,
            created_at=post.created_at,
            message="File uploaded successfully",
        )
    
    except HTTPException:
        # FIX: Re-raise HTTPException without rollback (no DB operations before this point)
        # HTTPExceptions are raised for validation errors before any DB operations
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the file upload",
        )
    finally:
        # FIX: Ensure proper cleanup of file handles and temp files
        # Close file handle if still open (error occurred during streaming)
        if temp_file_handle and not temp_file_handle.closed:
            try:
                temp_file_handle.close()
            except Exception as e:
                logger.warning(f"Failed to close temp file handle: {e}")
        
        # Cleanup temporary file in all cases (success or failure)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")


@app.get(
    "/api/feed",
    response_model=PostListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Posts"],
    summary="Get personalized feed",
    description="Retrieve paginated personalized feed (followed users first, then others) with likes/comments counts",
)
async def get_feed(
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    page_size: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description="Number of posts per page")
    ] = DEFAULT_PAGE_SIZE,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Get paginated personalized feed - posts from followed users appear first.
    
    FIX: Personalized feed shows followed users' posts first, then others.
    Includes counts for likes and comments, and whether current user has liked each post.
    
    Args:
        page: Page number (starting from 1)
        page_size: Number of posts per page (max 100)
        session: Database session (from dependency)
        user: Authenticated user (from dependency)
    
    Returns:
        PostListResponse: Paginated list of posts with metadata and counts
    """
    try:
        offset = (page - 1) * page_size
        
        # FIX: Get list of followed user IDs for personalized feed ordering
        # UUIDString type decorator automatically converts UUID objects to strings for SQL
        followed_result = await session.execute(
            select(Follow.followed_id).where(Follow.follower_id == user.id)
        )
        followed_user_ids_list = [row[0] for row in followed_result.all()]
        
        # FIX: Order by: followed users first (priority=0), then others (priority=1), then by created_at
        # UUIDString's process_bind_param converts UUID objects to strings for SQL IN clause automatically
        if followed_user_ids_list:
            priority_case = case(
                (Post.user_id.in_(followed_user_ids_list), 0),
                else_=1
            )
            # Get posts with personalized ordering (followed users first)
            posts_query = (
                select(Post)
                .order_by(priority_case, desc(Post.created_at))
                .offset(offset)
                .limit(page_size)
            )
        else:
            # No follows yet, just order by date (most recent first)
            posts_query = (
                select(Post)
                .order_by(desc(Post.created_at))
                .offset(offset)
                .limit(page_size)
            )
        
        result = await session.execute(posts_query)
        posts = result.scalars().all()
        post_ids = [post.id for post in posts]
        
        # FIX: Get counts and likes in efficient queries
        # UUIDString type decorator converts DB strings to UUID objects automatically
        # Get likes counts for these posts
        likes_counts_result = await session.execute(
            select(Like.post_id, func.count(Like.id).label("count"))
            .where(Like.post_id.in_(post_ids))
            .group_by(Like.post_id)
        )
        # Convert UUID objects to strings for dictionary lookup
        likes_counts = {str(row[0]): row[1] for row in likes_counts_result.all()}
        
        # Get comments counts for these posts
        comments_counts_result = await session.execute(
            select(Comment.post_id, func.count(Comment.id).label("count"))
            .where(Comment.post_id.in_(post_ids))
            .group_by(Comment.post_id)
        )
        # Convert UUID objects to strings for dictionary lookup
        comments_counts = {str(row[0]): row[1] for row in comments_counts_result.all()}
        
        # Get user's likes for these posts (to determine is_liked)
        # UUIDString type decorator handles UUID conversion automatically
        user_likes_result = await session.execute(
            select(Like.post_id)
            .where(and_(Like.post_id.in_(post_ids), Like.user_id == user.id))
        )
        user_liked_post_ids = {row[0] for row in user_likes_result.all()}
        
        # Get user information for posts
        user_ids = {post.user_id for post in posts}
        if user_ids:
            users_result = await session.execute(
                select(User).where(User.id.in_(user_ids))
            )
            users = {u.id: u.email for u in users_result.scalars().all()}
        else:
            users = {}
        
        # Get total count for pagination (all posts, not just this page)
        total_result = await session.execute(select(func.count(Post.id)))
        total = total_result.scalar() or 0
        
        # Build response with counts
        posts_data = []
        for post in posts:
            posts_data.append(
                PostResponse(
                    id=post.id,
                    user_id=post.user_id,
                    caption=post.caption,
                    url=post.url,
                    file_type=post.file_type,
                    file_name=post.file_name,
                    created_at=post.created_at,
                    # UUIDString returns UUID objects, fastapi-users User.id is also UUID
                    # Both should be UUID objects, so direct comparison works
                    is_owner=post.user_id == user.id,
                    author_email=users.get(post.user_id),
                    likes_count=likes_counts.get(str(post.id), 0),
                    comments_count=comments_counts.get(str(post.id), 0),
                    is_liked=post.id in user_liked_post_ids,
                )
            )
        
        return PostListResponse(
            posts=posts_data,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total,
            has_prev=page > 1,
        )
    
    except Exception as e:
        logger.error(f"Error fetching feed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching the feed",
        )


@app.get(
    "/api/posts/{post_id}",
    response_model=PostResponse,
    status_code=status.HTTP_200_OK,
    tags=["Posts"],
    summary="Get a post",
    description="Retrieve a specific post by ID",
)
async def get_post(
    post_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Get a specific post by ID.
    
    Args:
        post_id: UUID of the post to retrieve
        session: Database session (from dependency)
        user: Authenticated user (from dependency)
    
    Returns:
        PostResponse: Post information
    
    Raises:
        HTTPException: If post not found
    """
    try:
        result = await session.execute(select(Post).where(Post.id == post_id))
        post = result.scalars().first()
        
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        # Get author information
        user_result = await session.execute(select(User).where(User.id == post.user_id))
        author = user_result.scalars().first()
        
        # FIX: Get counts and like status for single post
        likes_count_result = await session.execute(
            select(func.count(Like.id)).where(Like.post_id == post_id)
        )
        likes_count = likes_count_result.scalar() or 0
        
        comments_count_result = await session.execute(
            select(func.count(Comment.id)).where(Comment.post_id == post_id)
        )
        comments_count = comments_count_result.scalar() or 0
        
        is_liked_result = await session.execute(
            select(Like.id).where(and_(Like.post_id == post_id, Like.user_id == user.id))
        )
        is_liked = is_liked_result.first() is not None
        
        return PostResponse(
            id=post.id,
            user_id=post.user_id,
            caption=post.caption,
            url=post.url,
            file_type=post.file_type,
            file_name=post.file_name,
            created_at=post.created_at,
            is_owner=post.user_id == user.id,
            author_email=author.email if author else None,
            likes_count=likes_count,
            comments_count=comments_count,
            is_liked=is_liked,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching post {post_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching the post",
        )


@app.delete(
    "/api/posts/{post_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    tags=["Posts"],
    summary="Delete a post",
    description="Delete a post. Only the post owner can delete their posts",
)
async def delete_post(
    post_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Delete a post by ID.
    
    Only the owner of the post can delete it.
    
    Args:
        post_id: UUID of the post to delete
        session: Database session (from dependency)
        user: Authenticated user (from dependency)
    
    Returns:
        DeleteResponse: Deletion confirmation
    
    Raises:
        HTTPException: If post not found or user doesn't have permission
    """
    try:
        result = await session.execute(select(Post).where(Post.id == post_id))
        post = result.scalars().first()
        
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        if post.user_id != user.id:
            logger.warning(
                f"User {user.id} attempted to delete post {post_id} owned by {post.user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this post",
            )
        
        # FIX: Use delete() statement for async SQLAlchemy operations
        # While session.delete() still works, delete() statement is the recommended async pattern
        # and provides better control and compatibility with SQLAlchemy 2.0+ async APIs
        await session.execute(delete(Post).where(Post.id == post_id))
        await session.commit()
        
        logger.info(f"Post {post_id} deleted by user {user.id}")
        
        return DeleteResponse(
            success=True,
            message="Post deleted successfully",
            post_id=post_id,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting post {post_id}: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the post",
        )


# FIX: Add endpoints for social features (likes, comments, follows)
@app.post(
    "/api/posts/{post_id}/like",
    response_model=LikeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Posts"],
    summary="Like or unlike a post",
    description="Toggle like on a post. If already liked, removes the like (unlike). Prevents duplicate likes.",
)
async def toggle_like(
    post_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Like or unlike a post. FIX: Unique constraint prevents duplicate likes.
    
    Args:
        post_id: UUID of the post to like/unlike
        session: Database session
        user: Authenticated user
    
    Returns:
        LikeResponse: Success status and current like state
    """
    try:
        # Verify post exists
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalars().first()
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        # Check if user already liked this post
        existing_like_result = await session.execute(
            select(Like).where(and_(Like.post_id == post_id, Like.user_id == user.id))
        )
        existing_like = existing_like_result.scalars().first()
        
        if existing_like:
            # Unlike: delete the like
            await session.execute(
                delete(Like).where(and_(Like.post_id == post_id, Like.user_id == user.id))
            )
            await session.commit()
            logger.info(f"User {user.id} unliked post {post_id}")
            return LikeResponse(
                success=True,
                message="Post unliked successfully",
                post_id=post_id,
                is_liked=False,
            )
        else:
            # Like: create new like (unique constraint prevents duplicates)
            new_like = Like(user_id=user.id, post_id=post_id)
            session.add(new_like)
            try:
                await session.commit()
                logger.info(f"User {user.id} liked post {post_id}")
                return LikeResponse(
                    success=True,
                    message="Post liked successfully",
                    post_id=post_id,
                    is_liked=True,
                )
            except SQLAlchemyError as e:
                await session.rollback()
                # If unique constraint violation, post was already liked (race condition)
                logger.warning(f"Duplicate like attempt by user {user.id} on post {post_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Post is already liked",
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling like on post {post_id}: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while toggling like",
        )


@app.post(
    "/api/posts/{post_id}/comments",
    response_model=CommentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Posts"],
    summary="Create a comment on a post",
    description="Add a comment to a post. Requires authentication.",
)
async def create_comment(
    post_id: uuid.UUID,
    comment_data: CommentCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Create a comment on a post.
    
    Args:
        post_id: UUID of the post to comment on
        comment_data: Comment content
        session: Database session
        user: Authenticated user
    
    Returns:
        CommentResponse: Created comment information
    """
    try:
        # Verify post exists
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalars().first()
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        # Create comment
        comment = Comment(
            user_id=user.id,
            post_id=post_id,
            content=comment_data.content.strip(),
        )
        session.add(comment)
        await session.commit()
        await session.refresh(comment)
        
        # Get author email
        author_result = await session.execute(select(User).where(User.id == user.id))
        author = author_result.scalars().first()
        
        logger.info(f"User {user.id} commented on post {post_id}")
        
        return CommentResponse(
            id=comment.id,
            user_id=comment.user_id,
            post_id=comment.post_id,
            content=comment.content,
            created_at=comment.created_at,
            author_email=author.email if author else None,
            is_owner=True,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comment on post {post_id}: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating comment",
        )


@app.get(
    "/api/posts/{post_id}/comments",
    response_model=CommentListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Posts"],
    summary="Get comments for a post",
    description="Retrieve paginated comments for a post, ordered by creation time (newest first)",
)
async def get_comments(
    post_id: uuid.UUID,
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    page_size: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description="Number of comments per page")
    ] = DEFAULT_PAGE_SIZE,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Get paginated comments for a post. FIX: Ordered by creation time, newest first.
    
    Args:
        post_id: UUID of the post
        page: Page number (starting from 1)
        page_size: Number of comments per page (max 100)
        session: Database session
        user: Authenticated user
    
    Returns:
        CommentListResponse: Paginated list of comments
    """
    try:
        # Verify post exists
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalars().first()
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        offset = (page - 1) * page_size
        
        # Get total count
        total_result = await session.execute(
            select(func.count(Comment.id)).where(Comment.post_id == post_id)
        )
        total = total_result.scalar() or 0
        
        # Get comments with pagination (newest first)
        comments_result = await session.execute(
            select(Comment)
            .where(Comment.post_id == post_id)
            .order_by(desc(Comment.created_at))
            .offset(offset)
            .limit(page_size)
        )
        comments = comments_result.scalars().all()
        
        # Get author information
        user_ids = {comment.user_id for comment in comments}
        authors = {}
        if user_ids:
            authors_result = await session.execute(
                select(User).where(User.id.in_(user_ids))
            )
            authors = {u.id: u.email for u in authors_result.scalars().all()}
        
        # Build response
        comments_data = []
        for comment in comments:
            comments_data.append(
                CommentResponse(
                    id=comment.id,
                    user_id=comment.user_id,
                    post_id=comment.post_id,
                    content=comment.content,
                    created_at=comment.created_at,
                    author_email=authors.get(comment.user_id),
                    is_owner=comment.user_id == user.id,
                )
            )
        
        return CommentListResponse(
            comments=comments_data,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total,
            has_prev=page > 1,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching comments for post {post_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching comments",
        )


@app.post(
    "/api/users/{user_id}/follow",
    response_model=FollowResponse,
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Follow or unfollow a user",
    description="Toggle follow status. If already following, unfollows. Prevents self-follow and duplicates.",
)
async def toggle_follow(
    user_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Follow or unfollow a user. FIX: Prevents self-follow and duplicate follows via unique constraint.
    
    Args:
        user_id: UUID of the user to follow/unfollow
        session: Database session
        user: Authenticated user (follower)
    
    Returns:
        FollowResponse: Success status and current follow state
    """
    try:
        # Prevent self-follow
        if user_id == user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot follow yourself",
            )
        
        # Verify target user exists
        target_user_result = await session.execute(select(User).where(User.id == user_id))
        target_user = target_user_result.scalars().first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )
        
        # Check if already following
        existing_follow_result = await session.execute(
            select(Follow).where(
                and_(Follow.follower_id == user.id, Follow.followed_id == user_id)
            )
        )
        existing_follow = existing_follow_result.scalars().first()
        
        if existing_follow:
            # Unfollow: delete the follow relationship
            await session.execute(
                delete(Follow).where(
                    and_(Follow.follower_id == user.id, Follow.followed_id == user_id)
                )
            )
            await session.commit()
            logger.info(f"User {user.id} unfollowed user {user_id}")
            return FollowResponse(
                success=True,
                message="User unfollowed successfully",
                followed_user_id=user_id,
                is_following=False,
            )
        else:
            # Follow: create new follow relationship (unique constraint prevents duplicates)
            new_follow = Follow(follower_id=user.id, followed_id=user_id)
            session.add(new_follow)
            try:
                await session.commit()
                logger.info(f"User {user.id} followed user {user_id}")
                return FollowResponse(
                    success=True,
                    message="User followed successfully",
                    followed_user_id=user_id,
                    is_following=True,
                )
            except SQLAlchemyError as e:
                await session.rollback()
                # If unique constraint violation, already following (race condition)
                logger.warning(f"Duplicate follow attempt by user {user.id} on user {user_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Already following this user",
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling follow for user {user_id}: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while toggling follow",
        )


@app.get(
    "/api/users/{user_id}/profile",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Get user profile with counts",
    description="Retrieve user profile including followers, following, and posts counts",
)
async def get_user_profile(
    user_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Get user profile with social counts. FIX: Includes followers_count, following_count, posts_count.
    
    Args:
        user_id: UUID of the user whose profile to retrieve
        session: Database session
        user: Authenticated user (to check if following)
    
    Returns:
        UserProfileResponse: User profile with counts
    """
    try:
        # Get target user
        user_result = await session.execute(select(User).where(User.id == user_id))
        target_user = user_result.scalars().first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )
        
        # FIX: Get counts efficiently with separate queries
        followers_count_result = await session.execute(
            select(func.count(Follow.id)).where(Follow.followed_id == user_id)
        )
        followers_count = followers_count_result.scalar() or 0
        
        following_count_result = await session.execute(
            select(func.count(Follow.id)).where(Follow.follower_id == user_id)
        )
        following_count = following_count_result.scalar() or 0
        
        posts_count_result = await session.execute(
            select(func.count(Post.id)).where(Post.user_id == user_id)
        )
        posts_count = posts_count_result.scalar() or 0
        
        # Check if current user follows target user
        is_following_result = await session.execute(
            select(Follow.id).where(
                and_(Follow.follower_id == user.id, Follow.followed_id == user_id)
            )
        )
        is_following = is_following_result.first() is not None
        
        return UserProfileResponse(
            id=target_user.id,
            email=target_user.email,
            is_active=target_user.is_active,
            is_verified=target_user.is_verified if hasattr(target_user, "is_verified") else False,
            followers_count=followers_count,
            following_count=following_count,
            posts_count=posts_count,
            is_following=is_following,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching user profile",
        )


# Root endpoint
@app.get("/", tags=["Root"], summary="API Root", description="Get API information")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "Social Media Backend API",
        "version": APP_VERSION,
        "docs_url": "/api/docs",
        "health_check": "/health",
    }
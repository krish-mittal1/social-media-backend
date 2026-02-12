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
from sqlalchemy import select, func, desc, delete, and_, case, update as sa_update
from sqlalchemy.orm import selectinload
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
    PostUpdate,
    PostMediaResponse,
    UserRead,
    UserCreate,
    UserUpdate,
    CommentCreate,
    CommentResponse,
    CommentListResponse,
    LikeResponse,
    FollowResponse,
    UserProfileResponse,
    UserSearchResult,
    SettingsUpdate,
    ChangePassword,
)
from app.db import Post, PostMedia, create_db_and_tables, get_async_session, User, Like, Comment, Follow
from app.users import auth_backend, current_active_user, fastapi_users
from app.images import imagekit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            "app.log",
            maxBytes=10 * 1024 * 1024,  
            backupCount=5,  
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


MAX_FILE_SIZE = 50 * 1024 * 1024  
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

    async def init_db():
        try:
            await create_db_and_tables()
            logger.info("Database tables created successfully")
            # Migrate existing posts without PostMedia rows
            await migrate_existing_posts()
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}. App will continue but DB operations may fail.")

    asyncio.create_task(init_db())

    yield

    logger.info("Shutting down application...")


async def migrate_existing_posts():
    """Create PostMedia rows for existing posts that don't have any."""
    from app.db import async_session_maker
    try:
        async with async_session_maker() as session:
            # Find posts missing media entries
            result = await session.execute(
                select(Post).outerjoin(PostMedia).where(PostMedia.id == None)
            )
            posts = result.scalars().all()
            if not posts:
                return
            for post in posts:
                media = PostMedia(
                    post_id=post.id,
                    url=post.url,
                    file_type=post.file_type,
                    file_name=post.file_name,
                    position=0,
                )
                session.add(media)
            await session.commit()
            logger.info(f"Migrated {len(posts)} existing posts to PostMedia table")
    except Exception as e:
        logger.warning(f"Post media migration skipped or failed: {e}")



app = FastAPI(
    title="ScrollVerse API",
    description="Production-ready REST API for social media platform with authentication, file uploads, and feed management",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:8000,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



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
    """Handle unexpected exceptions — logs full traceback for debugging"""
    import traceback
    tb = traceback.format_exc()
    logger.error(
        f"Unexpected error on {request.method} {request.url}: {type(exc).__name__}: {exc}\n{tb}"
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if os.getenv("DEBUG", "").lower() == "true" else "An unexpected error occurred. Please try again later.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )



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
        
        await asyncio.wait_for(session.execute(select(1)), timeout=2.0)
        db_status = "connected"
    except asyncio.TimeoutError:
        logger.warning("Database health check timed out")
        db_status = "timeout"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        version=APP_VERSION,
        database=db_status,
        timestamp=datetime.now(timezone.utc),
    )



def validate_file_size(file: UploadFile) -> None:
    """Validate file size before processing"""
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )


def validate_file_type(content_type: str | None) -> None:
    """Validate file MIME type"""
    
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
    
    if not content_type:
        return "unknown"
    if content_type.startswith("image/"):
        return "image"
    elif content_type.startswith("video/"):
        return "video"
    return "unknown"



@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Posts"],
    summary="Upload file(s)",
    description="Upload one or more image/video files with an optional caption. Maximum file size: 50MB each",
)
async def upload_file(
    files: list[UploadFile] = File(..., description="Image or video file(s) to upload"),
    caption: str = Form(default="", max_length=2000, description="Optional caption for the post"),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
):
    # Support legacy single-file clients that send 'file' instead of 'files'
    file = files[0] if files else None
    if not file:
        raise HTTPException(status_code=400, detail="At least one file is required")
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
        
        validate_file_type(file.content_type)
        
      
        file_extension = Path(file.filename).suffix if file.filename else ""
        temp_file_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,  
            suffix=file_extension,
            prefix="upload_",
        )
        temp_file_path = temp_file_handle.name
        
        file_size = 0
        chunk_size = 8192  

        
        try:
            await file.seek(0)
        except (AttributeError, OSError) as e:
            logger.warning(f"Could not seek file: {e}. File may have been consumed.")
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File could not be processed. Please try uploading again.",
            )
        
        
        def write_chunk(chunk_data):
            """Helper function to write chunk synchronously in thread pool"""
            temp_file_handle.write(chunk_data)
        
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            
           
            if file_size > MAX_FILE_SIZE:
                temp_file_handle.close()
                temp_file_handle = None  
                os.unlink(temp_file_path)
                temp_file_path = None  
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f}MB",
                )
            
            #
            await run_in_threadpool(write_chunk, chunk)
        
        temp_file_handle.close()
        temp_file_handle = None
        
        if file_size == 0:
            
            os.unlink(temp_file_path)
            temp_file_path = None
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty",
            )
        
        
        logger.info(f"Uploading file {file.filename} ({file_size / (1024*1024):.2f}MB) for user {user.id}")
        
        
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
        
        
        if not hasattr(upload_result, "url") or not upload_result.url:
            logger.error(f"ImageKit upload failed: {upload_result}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="File upload to storage service failed",
            )
        
        
        file_name = (
            getattr(upload_result, "name", None)
            or getattr(upload_result, "file_name", None)
            or file.filename
            or f"upload_{uuid.uuid4()}{file_extension}"
        )
        
        # Upload additional files (index 1+)
        all_uploaded = [(upload_result.url, get_file_type(file.content_type), file_name)]
        for extra_file in files[1:]:
            validate_file_type(extra_file.content_type)
            ext2 = Path(extra_file.filename).suffix if extra_file.filename else ""
            tf2 = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=ext2, prefix="upload_")
            try:
                await extra_file.seek(0)
                sz2 = 0
                while True:
                    chunk2 = await extra_file.read(8192)
                    if not chunk2:
                        break
                    sz2 += len(chunk2)
                    if sz2 > MAX_FILE_SIZE:
                        tf2.close()
                        os.unlink(tf2.name)
                        raise HTTPException(status_code=413, detail=f"File exceeds max size")
                    tf2.write(chunk2)
                tf2.close()
                r2 = await run_in_threadpool(
                    upload_to_imagekit, tf2.name,
                    extra_file.filename or f"upload_{uuid.uuid4()}{ext2}",
                    extra_file.content_type or "",
                )
                fn2 = getattr(r2, "name", None) or getattr(r2, "file_name", None) or extra_file.filename or f"upload_{uuid.uuid4()}{ext2}"
                all_uploaded.append((r2.url, get_file_type(extra_file.content_type), fn2))
            finally:
                if os.path.exists(tf2.name):
                    os.unlink(tf2.name)

        post = Post(
            user_id=user.id,
            caption=caption.strip() if caption else None,
            url=all_uploaded[0][0],
            file_type=all_uploaded[0][1],
            file_name=all_uploaded[0][2],
        )
        session.add(post)
        try:
            await session.commit()
            await session.refresh(post)
        except SQLAlchemyError as db_error:
            logger.error(f"DB error after upload for user {user.id}: {db_error}", exc_info=True)
            await session.rollback()
            raise HTTPException(status_code=500, detail="File uploaded but failed to save post.")

        # Create PostMedia rows
        media_rows = []
        for idx, (m_url, m_type, m_name) in enumerate(all_uploaded):
            pm = PostMedia(post_id=post.id, url=m_url, file_type=m_type, file_name=m_name, position=idx)
            session.add(pm)
            media_rows.append(pm)
        await session.commit()
        for pm in media_rows:
            await session.refresh(pm)

        logger.info(f"Post {post.id} created with {len(media_rows)} media item(s) for user {user.id}")

        return UploadResponse(
            id=post.id,
            user_id=post.user_id,
            caption=post.caption,
            url=post.url,
            file_type=post.file_type,
            file_name=post.file_name,
            created_at=post.created_at,
            media=[PostMediaResponse(id=m.id, url=m.url, file_type=m.file_type, file_name=m.file_name, position=m.position) for m in media_rows],
            message="File uploaded successfully",
        )
    
    except HTTPException:
        
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the file upload",
        )
    finally:
        
        if temp_file_handle and not temp_file_handle.closed:
            try:
                temp_file_handle.close()
            except Exception as e:
                logger.warning(f"Failed to close temp file handle: {e}")
        
        
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
        
        
        followed_result = await session.execute(
            select(Follow.followed_id).where(Follow.follower_id == user.id)
        )
        followed_user_ids_list = [row[0] for row in followed_result.all()]
        
        
        if followed_user_ids_list:
            priority_case = case(
                (Post.user_id.in_(followed_user_ids_list), 0),
                else_=1
            )
            
            posts_query = (
                select(Post)
                .order_by(priority_case, desc(Post.created_at))
                .offset(offset)
                .limit(page_size)
            )
        else:
            
            posts_query = (
                select(Post)
                .order_by(desc(Post.created_at))
                .offset(offset)
                .limit(page_size)
            )
        
        result = await session.execute(posts_query)
        posts = result.scalars().all()
        post_ids = [post.id for post in posts]

        # Batch load media for all posts
        media_map = {}
        if post_ids:
            media_result = await session.execute(
                select(PostMedia).where(PostMedia.post_id.in_(post_ids)).order_by(PostMedia.position)
            )
            for m in media_result.scalars().all():
                media_map.setdefault(str(m.post_id), []).append(
                    PostMediaResponse(id=m.id, url=m.url, file_type=m.file_type, file_name=m.file_name, position=m.position)
                )
        post_ids = [post.id for post in posts]
        
        
        likes_counts_result = await session.execute(
            select(Like.post_id, func.count(Like.id).label("count"))
            .where(Like.post_id.in_(post_ids))
            .group_by(Like.post_id)
        )
        
        likes_counts = {str(row[0]): row[1] for row in likes_counts_result.all()}
        
        
        comments_counts_result = await session.execute(
            select(Comment.post_id, func.count(Comment.id).label("count"))
            .where(Comment.post_id.in_(post_ids))
            .group_by(Comment.post_id)
        )
       
        comments_counts = {str(row[0]): row[1] for row in comments_counts_result.all()}
        
        
        user_likes_result = await session.execute(
            select(Like.post_id)
            .where(and_(Like.post_id.in_(post_ids), Like.user_id == user.id))
        )
        user_liked_post_ids = {row[0] for row in user_likes_result.all()}
        
        
        user_ids = {post.user_id for post in posts}
        if user_ids:
            users_result = await session.execute(
                select(User).where(User.id.in_(user_ids))
            )
            users = {u.id: {"email": u.email, "username": u.username} for u in users_result.scalars().all()}
        else:
            users = {}
        
        
        total_result = await session.execute(select(func.count(Post.id)))
        total = total_result.scalar() or 0
        
        
        posts_data = []
        for post in posts:
            u_info = users.get(post.user_id, {})
            posts_data.append(
                PostResponse(
                    id=post.id,
                    user_id=post.user_id,
                    caption=post.caption,
                    url=post.url,
                    file_type=post.file_type,
                    file_name=post.file_name,
                    created_at=post.created_at,
                    is_owner=post.user_id == user.id,
                    author_email=u_info.get("email") if isinstance(u_info, dict) else u_info,
                    author_username=u_info.get("username") if isinstance(u_info, dict) else None,
                    likes_count=likes_counts.get(str(post.id), 0),
                    comments_count=comments_counts.get(str(post.id), 0),
                    is_liked=post.id in user_liked_post_ids,
                    media=media_map.get(str(post.id), []),
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
        
        
        user_result = await session.execute(select(User).where(User.id == post.user_id))
        author = user_result.scalars().first()
        
        
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

        # Load media
        media_result = await session.execute(
            select(PostMedia).where(PostMedia.post_id == post_id).order_by(PostMedia.position)
        )
        media_items = [
            PostMediaResponse(id=m.id, url=m.url, file_type=m.file_type, file_name=m.file_name, position=m.position)
            for m in media_result.scalars().all()
        ]

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
            author_username=author.username if author else None,
            likes_count=likes_count,
            comments_count=comments_count,
            is_liked=is_liked,
            media=media_items,
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
        
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalars().first()
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        
        existing_like_result = await session.execute(
            select(Like).where(and_(Like.post_id == post_id, Like.user_id == user.id))
        )
        existing_like = existing_like_result.scalars().first()
        
        if existing_like:
            
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
        
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalars().first()
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        
        comment = Comment(
            user_id=user.id,
            post_id=post_id,
            content=comment_data.content.strip(),
        )
        session.add(comment)
        await session.commit()
        await session.refresh(comment)
        
        
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
            author_username=author.username if author else None,
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
        
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalars().first()
        if not post:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Post with ID {post_id} not found",
            )
        
        offset = (page - 1) * page_size
        
        
        total_result = await session.execute(
            select(func.count(Comment.id)).where(Comment.post_id == post_id)
        )
        total = total_result.scalar() or 0
        
        
        comments_result = await session.execute(
            select(Comment)
            .where(Comment.post_id == post_id)
            .order_by(desc(Comment.created_at))
            .offset(offset)
            .limit(page_size)
        )
        comments = comments_result.scalars().all()
        
        
        user_ids = {comment.user_id for comment in comments}
        authors = {}
        if user_ids:
            authors_result = await session.execute(
                select(User).where(User.id.in_(user_ids))
            )
            authors = {u.id: {"email": u.email, "username": u.username} for u in authors_result.scalars().all()}
        
        
        comments_data = []
        for comment in comments:
            a_info = authors.get(comment.user_id, {})
            comments_data.append(
                CommentResponse(
                    id=comment.id,
                    user_id=comment.user_id,
                    post_id=comment.post_id,
                    content=comment.content,
                    created_at=comment.created_at,
                    author_email=a_info.get("email") if isinstance(a_info, dict) else a_info,
                    author_username=a_info.get("username") if isinstance(a_info, dict) else None,
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
        
        if user_id == user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot follow yourself",
            )
        
        
        target_user_result = await session.execute(select(User).where(User.id == user_id))
        target_user = target_user_result.scalars().first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )
        
        
        existing_follow_result = await session.execute(
            select(Follow).where(
                and_(Follow.follower_id == user.id, Follow.followed_id == user_id)
            )
        )
        existing_follow = existing_follow_result.scalars().first()
        
        if existing_follow:
            
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
        
        user_result = await session.execute(select(User).where(User.id == user_id))
        target_user = user_result.scalars().first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )
        
        
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
        
        
        is_following_result = await session.execute(
            select(Follow.id).where(
                and_(Follow.follower_id == user.id, Follow.followed_id == user_id)
            )
        )
        is_following = is_following_result.first() is not None
        
        return UserProfileResponse(
            id=target_user.id,
            email=target_user.email,
            username=target_user.username,
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



@app.get(
    "/api/users/{user_id}/posts",
    response_model=PostListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Get posts by a user",
    description="Retrieve paginated list of posts created by a specific user",
)
async def get_user_posts(
    user_id: uuid.UUID,
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    page_size: Annotated[
        int, Query(ge=1, le=MAX_PAGE_SIZE, description="Number of posts per page")
    ] = DEFAULT_PAGE_SIZE,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """Get paginated posts by a specific user."""
    try:
        target_result = await session.execute(select(User).where(User.id == user_id))
        target_user = target_result.scalars().first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        offset = (page - 1) * page_size

        total_result = await session.execute(
            select(func.count(Post.id)).where(Post.user_id == user_id)
        )
        total = total_result.scalar() or 0

        posts_result = await session.execute(
            select(Post)
            .where(Post.user_id == user_id)
            .order_by(desc(Post.created_at))
            .offset(offset)
            .limit(page_size)
        )
        posts = posts_result.scalars().all()
        post_ids = [p.id for p in posts]

        # Likes counts
        likes_counts = {}
        if post_ids:
            lc = await session.execute(
                select(Like.post_id, func.count(Like.id).label("count"))
                .where(Like.post_id.in_(post_ids))
                .group_by(Like.post_id)
            )
            likes_counts = {str(r[0]): r[1] for r in lc.all()}

        # Comments counts
        comments_counts = {}
        if post_ids:
            cc = await session.execute(
                select(Comment.post_id, func.count(Comment.id).label("count"))
                .where(Comment.post_id.in_(post_ids))
                .group_by(Comment.post_id)
            )
            comments_counts = {str(r[0]): r[1] for r in cc.all()}

        # User liked
        user_liked = set()
        if post_ids:
            ul = await session.execute(
                select(Like.post_id)
                .where(and_(Like.post_id.in_(post_ids), Like.user_id == user.id))
            )
            user_liked = {r[0] for r in ul.all()}

        # Batch load media
        media_map = {}
        if post_ids:
            mr = await session.execute(
                select(PostMedia).where(PostMedia.post_id.in_(post_ids)).order_by(PostMedia.position)
            )
            for m in mr.scalars().all():
                media_map.setdefault(str(m.post_id), []).append(
                    PostMediaResponse(id=m.id, url=m.url, file_type=m.file_type, file_name=m.file_name, position=m.position)
                )

        posts_data = [
            PostResponse(
                id=p.id,
                user_id=p.user_id,
                caption=p.caption,
                url=p.url,
                file_type=p.file_type,
                file_name=p.file_name,
                created_at=p.created_at,
                is_owner=p.user_id == user.id,
                author_email=target_user.email,
                author_username=target_user.username,
                likes_count=likes_counts.get(str(p.id), 0),
                comments_count=comments_counts.get(str(p.id), 0),
                is_liked=p.id in user_liked,
                media=media_map.get(str(p.id), []),
            )
            for p in posts
        ]

        return PostListResponse(
            posts=posts_data,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total,
            has_prev=page > 1,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching posts for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching user posts",
        )


@app.get("/", tags=["Root"], summary="API Root", description="Get API information")
async def root():
    """Root endpoint providing API information"""
    return {
        "name": "ScrollVerse API",
        "version": APP_VERSION,
        "docs_url": "/api/docs",
        "health_check": "/health",
    }


# ─── USER SEARCH ───────────────────────────────────────────────
@app.get(
    "/api/users/search",
    response_model=list[UserSearchResult],
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Search users",
    description="Search users by username (case-insensitive partial match)",
)
async def search_users(
    q: str = Query("", min_length=1, max_length=50, description="Search query"),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """Search users by username. Returns up to 10 matching users."""
    try:
        result = await session.execute(
            select(User)
            .where(User.username.ilike(f"%{q}%"))
            .where(User.id != user.id)
            .limit(10)
        )
        users_found = result.scalars().all()
        return [
            UserSearchResult(id=u.id, username=u.username, email=u.email)
            for u in users_found
        ]
    except Exception as e:
        logger.error(f"Error searching users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while searching users")


# ─── EDIT POST ─────────────────────────────────────────────────
@app.patch(
    "/api/posts/{post_id}",
    response_model=PostResponse,
    status_code=status.HTTP_200_OK,
    tags=["Posts"],
    summary="Edit a post",
    description="Update a post's caption. Only the post owner can edit.",
)
async def edit_post(
    post_id: uuid.UUID,
    post_data: PostUpdate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """Edit a post caption. Owner-only."""
    try:
        result = await session.execute(select(Post).where(Post.id == post_id))
        post = result.scalars().first()

        if not post:
            raise HTTPException(status_code=404, detail=f"Post {post_id} not found")
        if post.user_id != user.id:
            raise HTTPException(status_code=403, detail="Not authorized to edit this post")

        if post_data.caption is not None:
            post.caption = post_data.caption.strip()

        await session.commit()
        await session.refresh(post)

        # Fetch counts + media
        author_result = await session.execute(select(User).where(User.id == post.user_id))
        author = author_result.scalars().first()
        lc = await session.execute(select(func.count(Like.id)).where(Like.post_id == post_id))
        likes_count = lc.scalar() or 0
        cc = await session.execute(select(func.count(Comment.id)).where(Comment.post_id == post_id))
        comments_count = cc.scalar() or 0
        il = await session.execute(select(Like.id).where(and_(Like.post_id == post_id, Like.user_id == user.id)))
        is_liked = il.first() is not None
        mr = await session.execute(select(PostMedia).where(PostMedia.post_id == post_id).order_by(PostMedia.position))
        media_items = [
            PostMediaResponse(id=m.id, url=m.url, file_type=m.file_type, file_name=m.file_name, position=m.position)
            for m in mr.scalars().all()
        ]

        return PostResponse(
            id=post.id, user_id=post.user_id, caption=post.caption,
            url=post.url, file_type=post.file_type, file_name=post.file_name,
            created_at=post.created_at, is_owner=True,
            author_email=author.email if author else None,
            author_username=author.username if author else None,
            likes_count=likes_count, comments_count=comments_count,
            is_liked=is_liked, media=media_items,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing post {post_id}: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="An error occurred while editing the post")


# ─── SETTINGS UPDATE ──────────────────────────────────────────
@app.patch(
    "/api/users/me/settings",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Update user settings",
    description="Update username. Validates uniqueness.",
)
async def update_settings(
    settings: SettingsUpdate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """Update authenticated user's settings (username)."""
    import re as _re
    try:
        if settings.username is not None:
            new_username = settings.username.strip()
            if not _re.match(r'^[a-zA-Z0-9_]+$', new_username):
                raise HTTPException(status_code=400, detail="Username can only contain letters, numbers, and underscores.")
            # Check uniqueness
            existing = await session.execute(
                select(User).where(User.username == new_username, User.id != user.id)
            )
            if existing.scalars().first():
                raise HTTPException(status_code=400, detail="Username already taken.")
            user.username = new_username

        await session.commit()
        await session.refresh(user)

        # Return profile
        fc = await session.execute(select(func.count(Follow.id)).where(Follow.followed_id == user.id))
        followers_count = fc.scalar() or 0
        fgc = await session.execute(select(func.count(Follow.id)).where(Follow.follower_id == user.id))
        following_count = fgc.scalar() or 0
        pc = await session.execute(select(func.count(Post.id)).where(Post.user_id == user.id))
        posts_count = pc.scalar() or 0

        return UserProfileResponse(
            id=user.id, email=user.email, username=user.username,
            is_active=user.is_active,
            is_verified=getattr(user, 'is_verified', False),
            followers_count=followers_count, following_count=following_count,
            posts_count=posts_count, is_following=False,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}", exc_info=True)
        await session.rollback()
        raise HTTPException(status_code=500, detail="An error occurred while updating settings")


# ─── CHANGE PASSWORD ──────────────────────────────────────────
@app.post(
    "/api/users/me/password",
    status_code=status.HTTP_200_OK,
    tags=["Users"],
    summary="Change password",
    description="Change authenticated user's password",
)
async def change_password(
    data: ChangePassword,
    request: Request,
    user: User = Depends(current_active_user),
):
    """Change password. Verifies current password first."""
    from app.users import get_user_manager, get_user_db
    from app.db import get_async_session as _get_session
    try:
        async for session in _get_session():
            async for user_db in get_user_db(session):
                async for user_manager in get_user_manager(user_db):
                    # Verify current password
                    verified, _ = user_manager.password_helper.verify_and_update(data.current_password, user.hashed_password)
                    if not verified:
                        raise HTTPException(status_code=400, detail="Current password is incorrect")
                    # Validate new password
                    await user_manager.validate_password(data.new_password, user)
                    # Update password
                    user.hashed_password = user_manager.password_helper.hash(data.new_password)
                    await session.commit()
                    return {"success": True, "message": "Password changed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while changing password")

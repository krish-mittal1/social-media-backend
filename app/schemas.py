from pydantic import BaseModel, Field, ConfigDict
from fastapi_users import schemas
from datetime import datetime
from typing import Optional
import uuid
import re


class PostCreate(BaseModel):
    caption: Optional[str] = Field(None, max_length=2000, description="Post caption")


class PostMediaResponse(BaseModel):
    """Response model for a single media item within a post"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    url: str
    file_type: str
    file_name: str
    position: int


class PostResponse(BaseModel):
    """Response model for a single post"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    caption: Optional[str]
    url: str
    file_type: str
    file_name: str
    created_at: datetime
    is_owner: bool = False
    author_email: Optional[str] = None
    author_username: Optional[str] = None

    likes_count: int = 0
    comments_count: int = 0
    is_liked: bool = False
    media: list[PostMediaResponse] = []


class PostListResponse(BaseModel):
    """Response model for paginated post list"""
    posts: list[PostResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class PostUpdate(BaseModel):
    """Schema for editing a post caption"""
    caption: Optional[str] = Field(None, max_length=2000, description="Updated caption")


class UploadResponse(BaseModel):
    """Response model for file upload"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    caption: Optional[str]
    url: str
    file_type: str
    file_name: str
    created_at: datetime
    media: list[PostMediaResponse] = []
    message: str = "File uploaded successfully"


class DeleteResponse(BaseModel):
    """Response model for post deletion"""
    success: bool
    message: str
    post_id: uuid.UUID


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: str
    status_code: int


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    database: str
    timestamp: datetime


class UserRead(schemas.BaseUser[uuid.UUID]):
    username: str


class UserCreate(schemas.BaseUserCreate):
    username: str = Field(
        ...,
        min_length=3,
        max_length=30,
        description="Unique username (3-30 chars, alphanumeric and underscores only)"
    )


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=30,
        description="Unique username (3-30 chars, alphanumeric and underscores only)"
    )


class CommentCreate(BaseModel):
    """Schema for creating a comment"""
    content: str = Field(..., min_length=1, max_length=2000, description="Comment content")


class CommentResponse(BaseModel):
    """Response model for a comment"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    post_id: uuid.UUID
    content: str
    created_at: datetime
    author_email: Optional[str] = None
    author_username: Optional[str] = None
    is_owner: bool = False


class CommentListResponse(BaseModel):
    """Response model for paginated comments"""
    comments: list[CommentResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class LikeResponse(BaseModel):
    """Response model for like action"""
    success: bool
    message: str
    post_id: uuid.UUID
    is_liked: bool


class FollowResponse(BaseModel):
    """Response model for follow/unfollow action"""
    success: bool
    message: str
    followed_user_id: uuid.UUID
    is_following: bool


class UserProfileResponse(BaseModel):
    """Extended user profile with counts"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    is_verified: bool
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0
    is_following: bool = False


class UserSearchResult(BaseModel):
    """User search result item"""
    id: uuid.UUID
    username: str
    email: str


class SettingsUpdate(BaseModel):
    """Schema for updating user settings"""
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=30,
        description="New username (3-30 chars, alphanumeric and underscores only)"
    )


class ChangePassword(BaseModel):
    """Schema for changing password"""
    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(..., min_length=6, max_length=128, description="New password")

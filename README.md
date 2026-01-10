# Social Media Backend (FastAPI)

A production-ready backend for a social media platform inspired by Instagram and TikTok.  
Built with FastAPI, async SQLAlchemy, and JWT authentication.

---

## 🚀 Features

- JWT authentication (FastAPI Users)
- Async SQLAlchemy ORM
- Image & video uploads (ImageKit)
- Posts, likes, comments, and follows
- Personalized feed
- User profiles with social stats
- Health check endpoint
- Production-ready architecture

---

## 🛠 Tech Stack

- **Backend:** FastAPI
- **Database:** PostgreSQL (production), SQLite (local)
- **ORM:** SQLAlchemy (async)
- **Auth:** JWT (FastAPI Users)
- **Media Storage:** ImageKit
- **Server:** Uvicorn

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

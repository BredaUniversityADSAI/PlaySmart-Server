"""
Main application entry point for the Play-O-Meter API.

Responsibilities:
- Load environment variables
- Initialize database models
- Configure CORS
- Register API routers
- Expose health and root endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.database import engine, Base
from app.router import (
    auth,
    toolkit,
    feedback,
    riot,
    match_data,
    videos,
    valorant,
    biometrics,
    hive,
    match_analytics,
)
from app.router import coach as coach_router
from app.router.match_analytics import router as analytics_router
import os

# Import models so SQLAlchemy registers them
from app.models.coach_config import CoachConfigDB  # noqa: F401


# ---------- Startup Setup ----------

# Load environment variables from .env
load_dotenv()

# Create database tables on startup (if they don't exist)
Base.metadata.create_all(bind=engine)


# ---------- App Initialization ----------

app = FastAPI(
    title="Play-O-Meter API",
    description="Breda Guardians Esports Analytics Platform",
    version="1.0.0",
)




# CORS - Read from environment
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
origins_list = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Router Registration ----------

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(toolkit.router, prefix="/api/toolkit", tags=["Toolkit"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(riot.router, prefix="/api", tags=["Riot"])
app.include_router(match_data.router, prefix="/api/matches", tags=["Matches"])
app.include_router(videos.router, prefix="/api", tags=["Videos"])
app.include_router(coach_router.router, prefix="/api/coach", tags=["Coach"])
app.include_router(biometrics.router, prefix="/api/biometrics", tags=["Biometrics"])
app.include_router(valorant.router)
app.include_router(hive.router, prefix="/api/hive", tags=["hive"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])


# ---------- Public Endpoints ----------


@app.get("/")
async def root():
    """
    Root endpoint.

    Returns basic service information.
    """
    return {
        "message": "Welcome to Play-O-Meter API",
        "version": "1.0.0",
        "team": "Breda Guardians",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Used for uptime monitoring and container orchestration.
    """
    return {
        "status": "healthy",
        "service": "play-o-meter-api",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

import os
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE importing any local modules
# This resolves paths relative to where uvicorn is running (the project root)
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import routes

app = FastAPI(
    title="Meeting Intelligence API",
    description="Automated pipeline for meeting transcription and intelligence extraction.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach routes
app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Meeting Intelligence API is running."}
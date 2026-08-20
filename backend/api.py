"""FastAPI web API for MythLens."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

from backend.main import analyze_text_claim, analyze_video_input

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="MythLens API",
    description="Medical myth fact-checking with Gemini, RAG and PubMed evidence.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class TextRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)
    top_k: int = Field(default=3, ge=1, le=5)


class URLRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/")
def home() -> Any:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"name": "MythLens", "status": "frontend_not_found"}
    return FileResponse(str(index_path))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "MythLens",
        "llm": "Gemini",
    }


@app.post("/api/analyze/text")
async def analyze_text(request: TextRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(analyze_text_claim, request.text.strip(), request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {exc}") from exc


@app.post("/api/analyze/url")
async def analyze_url(request: URLRequest) -> dict[str, Any]:
    if not request.url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Please provide a valid http/https URL.")
    try:
        return await run_in_threadpool(analyze_video_input, request.url.strip(), request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video URL analysis failed: {exc}") from exc


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    top_k: int = Form(default=3),
) -> dict[str, Any]:
    allowed_extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".wav", ".m4a", ".aac"}
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video/audio file type.")
    top_k = max(1, min(int(top_k), 5))

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="mythlens_upload_") as temp_file:
            temp_path = temp_file.name
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)

        return await run_in_threadpool(analyze_video_input, temp_path, top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

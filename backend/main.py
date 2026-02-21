"""FastAPI application for the Collembola Detection Pipeline web UI."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.routers import calibration, detection, images, jobs, measurement
from backend.services.detection import run_detection
from backend.services.measurement import run_measurement
from backend.websocket.progress import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    settings.ensure_dirs()

    # Register job handlers
    job_manager.register_handler(JobType.DETECTION, run_detection)
    job_manager.register_handler(JobType.MEASUREMENT, run_measurement)

    yield


app = FastAPI(
    title="Collembola Detection Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(images.router)
app.include_router(calibration.router)
app.include_router(detection.router)
app.include_router(measurement.router)
app.include_router(jobs.router)

# WebSocket
app.include_router(ws_router)

# Serve uploaded images and outputs as static files
uploads_dir = Path(settings.uploads_dir)
outputs_dir = Path(settings.outputs_dir)
uploads_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

app.mount("/files/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")
app.mount("/files/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.get("/api/health")
async def health():
    return {"status": "ok"}

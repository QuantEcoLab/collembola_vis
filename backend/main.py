"""FastAPI application for the Collembola Detection Pipeline web UI."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import db, db_projects
from backend.auth import router as auth_router
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.routers import annotations, calibration, community, detection, finetune, images, jobs, measurement, projects
from backend.routers import models as models_router
from backend.services.batch_detection import run_batch_detection
from backend.services.batch_measurement import run_batch_measurement
from backend.services.detection import run_detection
from backend.services.finetune import run_finetune
from backend.services.measurement import run_measurement
from backend.websocket.progress import router as ws_router

_DIST = Path(__file__).parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    settings.ensure_dirs()
    db.init_db()
    db_projects.init_project_db()

    # Register job handlers
    job_manager.register_handler(JobType.DETECTION, run_detection)
    job_manager.register_handler(JobType.MEASUREMENT, run_measurement)
    job_manager.register_handler(JobType.FINETUNE, run_finetune)
    job_manager.register_handler(JobType.BATCH, run_batch_detection)
    job_manager.register_handler(JobType.BATCH_MEASURE, run_batch_measurement)

    yield


app = FastAPI(
    title="Collembola Detection Pipeline",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/collembola",
)

# CORS — allow Vite dev server in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(auth_router)
app.include_router(images.router)
app.include_router(calibration.router)
app.include_router(detection.router)
app.include_router(measurement.router)
app.include_router(jobs.router)
app.include_router(annotations.router)
app.include_router(models_router.router)
app.include_router(finetune.router)
app.include_router(community.router)
app.include_router(projects.router)

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


# Serve the React SPA — must be last so API routes take priority
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="spa_assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve built frontend; fall back to index.html for SPA routing."""
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST / "index.html"))

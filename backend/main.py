"""FastAPI application for the Collembola Detection Pipeline web UI."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import db, db_projects
from backend.auth import router as auth_router
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.routers import annotations, calibration, community, contours, detection, finetune, images, jobs, measurement, projects
from backend.routers import models as models_router
from backend.routers import detector_dataset
from backend.services.batch_detection import run_batch_detection
from backend.services.batch_measurement import run_batch_measurement
from backend.services.batch_process import run_batch_process
from backend.services.detection import run_detection
from backend.services.finetune import run_finetune
from backend.services.measurement import run_measurement
from backend.services.contours import run_contours, validate_checkpoint
from backend.websocket.progress import router as ws_router

_DIST = Path(os.environ.get("COLLEMBOLA_FRONTEND_DIST", str(Path(__file__).parent.parent / "frontend" / "dist"))).resolve()
_LOCAL_PREVIEW = os.environ.get("COLLEMBOLA_LOCAL_PREVIEW") == "1"
_PREVIEW_MANIFEST = json.loads((_DIST / "preview-build.json").read_text()) if _LOCAL_PREVIEW and (_DIST / "preview-build.json").is_file() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    validate_checkpoint()
    if _LOCAL_PREVIEW and not _PREVIEW_MANIFEST:
        raise RuntimeError("Local preview requires a verified, versioned frontend manifest")
    settings.ensure_dirs()
    db.init_db()
    db_projects.init_project_db()

    # Register job handlers
    job_manager.register_handler(JobType.DETECTION, run_detection)
    job_manager.register_handler(JobType.MEASUREMENT, run_measurement)
    job_manager.register_handler(JobType.CONTOURS, run_contours)
    job_manager.register_handler(JobType.FINETUNE, run_finetune)
    job_manager.register_handler(JobType.BATCH, run_batch_detection)
    job_manager.register_handler(JobType.BATCH_MEASURE, run_batch_measurement)
    job_manager.register_handler(JobType.BATCH_PROCESS, run_batch_process)

    yield


_root_path = os.environ.get("ROOT_PATH", "/collembola")
_cors_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app = FastAPI(
    title="Collembola Detection Pipeline",
    version="1.0.0",
    lifespan=lifespan,
    root_path=_root_path,
)

# CORS — allow Vite dev server in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def preview_identity_headers(request: Request, call_next):
    response = await call_next(request)
    if _LOCAL_PREVIEW and _PREVIEW_MANIFEST:
        response.headers["X-Collembola-Preview-Build"] = _PREVIEW_MANIFEST["build_id"]
        response.headers["X-Collembola-Preview-Pid"] = str(os.getpid())
        if request.url.path.startswith(("/assets", "/api")) or "text/html" in response.headers.get("content-type", ""):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        if "text/html" in response.headers.get("content-type", ""):
            # Clear HTTP cache only; preserve auth/workspace/calibration localStorage and IndexedDB.
            response.headers["Clear-Site-Data"] = '"cache"'
    return response


@app.get("/api/preview-build")
async def preview_build():
    if not _LOCAL_PREVIEW or not _PREVIEW_MANIFEST:
        raise HTTPException(404, "Local preview diagnostics are disabled")
    return {"build_id": _PREVIEW_MANIFEST["build_id"], "built_at_utc": _PREVIEW_MANIFEST["built_at_utc"],
            "entrypoint_js": _PREVIEW_MANIFEST["entrypoint_js"], "entrypoint_sha256": _PREVIEW_MANIFEST["entrypoint_sha256"],
            "index_sha256": _PREVIEW_MANIFEST["index_sha256"], "frontend_dist": str(_DIST), "pid": os.getpid()}


@app.get("/sw.js")
@app.get("/service-worker.js")
@app.get("/dev-sw.js")
async def disable_preview_worker():
    if not _LOCAL_PREVIEW:
        raise HTTPException(404, "Not found")
    script = """self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil((async () => {
  await Promise.all((await caches.keys()).map(name => caches.delete(name)));
  await self.registration.unregister();
})()));
self.addEventListener('fetch', event => event.respondWith(fetch(event.request, {cache:'no-store'})));
"""
    return Response(script, media_type="application/javascript", headers={"Cache-Control": "no-store", "Service-Worker-Allowed": "/"})

# API routers
app.include_router(auth_router)
app.include_router(images.router)
app.include_router(calibration.router)
app.include_router(detection.router)
app.include_router(measurement.router)
app.include_router(contours.router)
app.include_router(jobs.router)
app.include_router(annotations.router)
app.include_router(models_router.router)
app.include_router(finetune.router)
app.include_router(community.router)
app.include_router(projects.router)
app.include_router(detector_dataset.router)

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
            return FileResponse(str(candidate), headers={"Cache-Control": "no-store"} if candidate.suffix == ".html" else None)
        return FileResponse(str(_DIST / "index.html"), headers={"Cache-Control": "no-store"})

"""Job listing and status endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user
from backend.jobs.manager import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(get_current_user)])


@router.get("")
async def list_jobs():
    """List all jobs."""
    return [j.to_dict() for j in job_manager.list_jobs()]


@router.get("/{job_id}")
async def get_job(job_id: str):
    """Get a specific job."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@router.get("/queue/stats")
async def get_queue_stats():
    """Get job queue statistics (workers, running, pending, completed, failed)."""
    return job_manager.get_stats()

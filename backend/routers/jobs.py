"""Job listing and status endpoints."""

from fastapi import APIRouter, HTTPException

from backend.jobs.manager import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


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

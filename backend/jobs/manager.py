"""In-memory job queue with single background worker thread."""

import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime
from queue import Queue
from typing import Any

from .models import Job, JobStatus, JobType


class JobManager:
    """Manages an in-memory job queue with a single worker thread.

    GPU operations must serialize anyway, so a single worker is appropriate.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: Queue[str] = Queue()
        self._subscribers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self._handlers: dict[JobType, Callable] = {}

    def register_handler(self, job_type: JobType, handler: Callable):
        """Register a handler function for a job type.

        Handler signature: handler(job: Job, progress_callback: Callable) -> dict
        """
        self._handlers[job_type] = handler

    def submit(self, job_type: JobType, params: dict[str, Any] | None = None) -> Job:
        """Submit a new job and return it."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, type=job_type, params=params or {})
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def register_completed(self, job_type: JobType, params: dict, result: dict) -> Job:
        """Register a pre-completed job without going through the queue.

        Used by batch detection to record per-image sub-jobs that were run
        inline (not via the worker thread).
        """
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now()
        job = Job(
            id=job_id,
            type=job_type,
            params=params,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result=result,
            created_at=now,
            started_at=now,
            completed_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def subscribe(self, job_id: str, callback: Callable):
        """Subscribe to progress updates for a job."""
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(callback)

    def unsubscribe(self, job_id: str, callback: Callable):
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if callback in subs:
                subs.remove(callback)

    def _notify(self, job: Job):
        """Notify all subscribers of a job update."""
        with self._lock:
            subs = list(self._subscribers.get(job.id, []))
        for cb in subs:
            try:
                cb(job)
            except Exception:
                pass

    def _run_worker(self):
        """Worker loop — pulls jobs from queue and executes them."""
        while True:
            job_id = self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue

            handler = self._handlers.get(job.type)
            if handler is None:
                job.status = JobStatus.FAILED
                job.error = f"No handler registered for {job.type}"
                self._notify(job)
                continue

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._notify(job)

            def progress_callback(progress: float, message: str):
                job.progress = progress
                job.message = message
                self._notify(job)

            try:
                result = handler(job, progress_callback)
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.result = result or {}
                job.completed_at = datetime.now()
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = f"{type(e).__name__}: {e}"
                job.completed_at = datetime.now()
                traceback.print_exc()

            self._notify(job)


# Singleton instance
job_manager = JobManager()

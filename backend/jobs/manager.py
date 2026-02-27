"""Multi-worker job queue with persistent state and background processing."""

import json
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any

from .models import Job, JobStatus, JobType


class JobManager:
    """Manages a persistent job queue with multiple concurrent workers.

    Features:
    - Up to MAX_WORKERS concurrent job executions
    - Job state persisted to disk (survives restart)
    - Queue state persisted (pending jobs restored on startup)
    - Session-independent (jobs run even if user disconnects)
    """

    MAX_WORKERS = 5  # Maximum concurrent jobs

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: Queue[str] = Queue()
        self._subscribers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
        self._handlers: dict[JobType, Callable] = {}
        self._workers: list[threading.Thread] = []
        self._running_count = 0
        self._shutdown = False
        
        # Load persisted jobs and queue state
        self._load_state()
        
        # Start worker pool
        for i in range(self.MAX_WORKERS):
            worker = threading.Thread(target=self._run_worker, args=(i,), daemon=True, name=f"JobWorker-{i}")
            worker.start()
            self._workers.append(worker)

    def register_handler(self, job_type: JobType, handler: Callable):
        """Register a handler function for a job type.

        Handler signature: handler(job: Job, progress_callback: Callable) -> dict
        """
        self._handlers[job_type] = handler

    def _get_queue_file(self) -> Path:
        """Path to persisted queue state file."""
        from backend.config import settings
        return settings.outputs_dir / "_queue_state.json"

    def _load_state(self) -> None:
        """Load jobs and queue state from disk on startup."""
        queue_file = self._get_queue_file()
        if not queue_file.exists():
            return
        
        try:
            data = json.loads(queue_file.read_text())
            pending_job_ids = data.get("pending_jobs", [])
            
            # Load each pending job from disk and re-queue it
            for job_id in pending_job_ids:
                job = self._load_from_disk(job_id)
                if job and job.status == JobStatus.PENDING:
                    self._jobs[job_id] = job
                    self._queue.put(job_id)
                elif job and job.status == JobStatus.RUNNING:
                    # Job was running when server stopped - reset to pending
                    job.status = JobStatus.PENDING
                    job.message = "Recovered after restart"
                    self._jobs[job_id] = job
                    self._queue.put(job_id)
                    self._persist(job)
        except Exception as e:
            print(f"Warning: Failed to load queue state: {e}")

    def _save_queue_state(self) -> None:
        """Persist current queue state to disk."""
        try:
            # Get all pending job IDs (in queue or not yet started)
            pending_ids = []
            with self._lock:
                pending_ids = [
                    job_id for job_id, job in self._jobs.items()
                    if job.status in (JobStatus.PENDING, JobStatus.RUNNING)
                ]
            
            data = {"pending_jobs": pending_ids, "saved_at": datetime.now().isoformat()}
            self._get_queue_file().write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Warning: Failed to save queue state: {e}")

    def submit(self, job_type: JobType, params: dict[str, Any] | None = None) -> Job:
        """Submit a new job and return it."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, type=job_type, params=params or {})
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)  # Persist immediately so it can be recovered
        self._queue.put(job_id)
        self._save_queue_state()  # Save queue state
        return job

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        return self._load_from_disk(job_id)

    def _load_from_disk(self, job_id: str) -> Job | None:
        """Try to reconstruct a completed job from its persisted result.json."""
        try:
            from backend.config import settings
            result_file = settings.outputs_dir / job_id / "result.json"
            if not result_file.exists():
                return None
            data = json.loads(result_file.read_text())
            job = Job(
                id=data["id"],
                type=JobType(data["type"]),
                status=JobStatus(data["status"]),
                progress=data.get("progress", 1.0),
                message=data.get("message", ""),
                result=data.get("result") or {},
                error=data.get("error"),
                params={},
                created_at=datetime.fromisoformat(data["created_at"]),
                started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            )
            with self._lock:
                self._jobs[job_id] = job
            return job
        except Exception:
            return None

    def _persist(self, job: Job) -> None:
        """Write job result to disk so it survives service restarts."""
        try:
            from backend.config import settings
            out_dir = settings.outputs_dir / job.id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(json.dumps(job.to_dict()))
        except Exception:
            pass

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
        self._persist(job)
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

    def _run_worker(self, worker_id: int):
        """Worker loop — pulls jobs from queue and executes them.
        
        Args:
            worker_id: Unique identifier for this worker thread (0 to MAX_WORKERS-1)
        """
        print(f"JobWorker-{worker_id} started")
        
        while not self._shutdown:
            try:
                # Block for up to 1 second, then check shutdown flag
                job_id = self._queue.get(timeout=1.0)
            except Empty:
                continue
            
            job = self._jobs.get(job_id)
            if job is None:
                continue

            handler = self._handlers.get(job.type)
            if handler is None:
                job.status = JobStatus.FAILED
                job.error = f"No handler registered for {job.type}"
                job.completed_at = datetime.now()
                self._persist(job)
                self._notify(job)
                self._save_queue_state()
                continue

            # Track running count for monitoring
            with self._lock:
                self._running_count += 1
            
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._persist(job)
            self._notify(job)
            self._save_queue_state()

            def progress_callback(progress: float, message: str):
                if job is not None:
                    job.progress = progress
                    job.message = message
                    self._notify(job)

            try:
                print(f"JobWorker-{worker_id} executing {job.type.value} job {job.id}")
                result = handler(job, progress_callback)
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.result = result or {}
                job.completed_at = datetime.now()
                print(f"JobWorker-{worker_id} completed job {job.id}")
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = f"{type(e).__name__}: {e}"
                job.completed_at = datetime.now()
                print(f"JobWorker-{worker_id} failed job {job.id}: {e}")
                traceback.print_exc()

            self._persist(job)
            self._notify(job)
            self._save_queue_state()
            
            with self._lock:
                self._running_count -= 1
        
        print(f"JobWorker-{worker_id} shutting down")

    def get_stats(self) -> dict[str, Any]:
        """Get current job queue statistics."""
        with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
            running = self._running_count
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
        
        return {
            "workers": self.MAX_WORKERS,
            "running": running,
            "pending": pending,
            "completed": completed,
            "failed": failed,
            "queue_size": self._queue.qsize(),
        }


# Singleton instance
job_manager = JobManager()

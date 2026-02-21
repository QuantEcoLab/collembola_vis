"""WebSocket endpoint for real-time job progress."""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.auth import verify_token
from backend.jobs.manager import job_manager

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str, token: str = ""):
    """WebSocket endpoint that streams job progress updates."""
    await websocket.accept()

    try:
        verify_token(token)
    except Exception:
        await websocket.send_json({"error": "Unauthorized"})
        await websocket.close(code=4001)
        return

    job = job_manager.get(job_id)
    if job is None:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    # Send current state immediately
    await websocket.send_json(job.to_dict())

    # If already terminal, close
    if job.status.value in ("completed", "failed"):
        await websocket.close()
        return

    # Set up an asyncio queue to bridge the sync callback → async websocket
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_update(updated_job):
        loop.call_soon_threadsafe(queue.put_nowait, updated_job.to_dict())

    job_manager.subscribe(job_id, on_update)

    try:
        while True:
            data = await queue.get()
            await websocket.send_json(data)
            if data.get("status") in ("completed", "failed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, on_update)

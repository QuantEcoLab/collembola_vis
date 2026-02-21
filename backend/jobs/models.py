"""Job data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    DETECTION = "detection"
    MEASUREMENT = "measurement"
    CALIBRATION = "calibration"
    BATCH = "batch"


@dataclass
class Job:
    id: str
    type: JobType
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

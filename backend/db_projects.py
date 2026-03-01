"""SQLite-backed projects database."""

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.config import settings

_DB_PATH = settings.data_dir / "community.db"
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_project_db() -> None:
    """Create project tables if they don't exist. Call once at startup."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_images (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                image_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                added_by TEXT NOT NULL,
                added_at TEXT NOT NULL,
                detection_job_id TEXT,
                measurement_job_id TEXT,
                folder TEXT DEFAULT NULL,
                UNIQUE(project_id, image_id)
            )
        """)
        conn.commit()
        # Idempotent migration: add folder column to existing databases
        try:
            conn.execute("ALTER TABLE project_images ADD COLUMN folder TEXT DEFAULT NULL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proj_images_project_id ON project_images(project_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proj_images_image_id ON project_images(image_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_created_by ON projects(created_by)")
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_project(name: str, description: str, username: str) -> str:
    """Create a project. Returns new project id."""
    pid = str(uuid.uuid4())
    now = _now()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, description, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (pid, name, description, username, now, now),
        )
        conn.commit()
    return pid


def list_projects() -> list[dict[str, Any]]:
    """Return all projects with image_count and thumbnail_image_id."""
    with _lock, _connect() as conn:
        rows = conn.execute("""
            SELECT
                p.id, p.name, p.description, p.created_by, p.created_at, p.updated_at,
                COUNT(pi.id) AS image_count,
                MIN(pi.image_id) AS thumbnail_image_id
            FROM projects p
            LEFT JOIN project_images pi ON pi.project_id = p.id
            GROUP BY p.id
            ORDER BY p.updated_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict[str, Any] | None:
    """Return project with full images list, or None if not found."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        project = dict(row)
        images = conn.execute(
            "SELECT * FROM project_images WHERE project_id = ? ORDER BY filename ASC",
            (project_id,),
        ).fetchall()
        project["images"] = [dict(i) for i in images]
    return project


def update_project(project_id: str, name: str, description: str) -> bool:
    """Update project name/description. Returns True if found."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE projects SET name = ?, description = ?, updated_at = ? WHERE id = ?",
            (name, description, _now(), project_id),
        )
        conn.commit()
    return cur.rowcount > 0


def delete_project(project_id: str) -> bool:
    """Delete project and cascade to project_images. Returns True if found."""
    with _lock, _connect() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
    return cur.rowcount > 0


def add_image(project_id: str, image_id: str, filename: str, username: str) -> str:
    """Add image to project. Returns new row id."""
    row_id = str(uuid.uuid4())
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO project_images (id, project_id, image_id, filename, added_by, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row_id, project_id, image_id, filename, username, _now()),
        )
        # Update project's updated_at
        conn.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (_now(), project_id),
        )
        conn.commit()
    return row_id


def remove_image(project_id: str, image_id: str) -> bool:
    """Remove image from project. Returns True if found."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "DELETE FROM project_images WHERE project_id = ? AND image_id = ?",
            (project_id, image_id),
        )
        conn.commit()
    return cur.rowcount > 0


def set_detection_job(project_id: str, image_id: str, job_id: str) -> None:
    """Set detection_job_id for a project image."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE project_images SET detection_job_id = ? WHERE project_id = ? AND image_id = ?",
            (job_id, project_id, image_id),
        )
        conn.commit()


def set_measurement_job(project_id: str, image_id: str, job_id: str) -> None:
    """Set measurement_job_id for a project image."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE project_images SET measurement_job_id = ? WHERE project_id = ? AND image_id = ?",
            (job_id, project_id, image_id),
        )
        conn.commit()


def set_image_folder(project_id: str, image_id: str, folder: str | None) -> bool:
    """Set (or clear) the folder for a project image. Returns True if the row was found."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE project_images SET folder = ? WHERE project_id = ? AND image_id = ?",
            (folder, project_id, image_id),
        )
        conn.commit()
    return cur.rowcount > 0


def list_folders(project_id: str) -> list[str]:
    """Return distinct non-null folder names for a project, sorted."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT folder FROM project_images WHERE project_id = ? AND folder IS NOT NULL ORDER BY folder",
            (project_id,),
        ).fetchall()
    return [row[0] for row in rows]


def delete_folder(project_id: str, folder_name: str) -> int:
    """Unassign all images from a folder (sets folder to NULL). Returns affected row count."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE project_images SET folder = NULL WHERE project_id = ? AND folder = ?",
            (project_id, folder_name),
        )
        conn.commit()
    return cur.rowcount


def rename_folder(project_id: str, old_name: str, new_name: str) -> int:
    """Rename a folder across all images. Returns affected row count."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "UPDATE project_images SET folder = ? WHERE project_id = ? AND folder = ?",
            (new_name, project_id, old_name),
        )
        conn.commit()
    return cur.rowcount

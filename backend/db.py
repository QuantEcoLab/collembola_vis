"""SQLite-backed community detections database."""

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from backend.config import settings

_DB_PATH = settings.data_dir / "community.db"
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at startup."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS community_detections (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                image_id TEXT,
                image_name TEXT NOT NULL,
                image_width INTEGER,
                image_height INTEGER,
                num_detections INTEGER NOT NULL,
                um_per_pixel REAL,
                conf_threshold REAL,
                boxes TEXT NOT NULL
            )
        """)
        # Migration: add image_id column to existing tables that don't have it
        try:
            conn.execute("ALTER TABLE community_detections ADD COLUMN image_id TEXT")
        except Exception:
            pass  # Column already exists
        conn.commit()


def upsert(
    *,
    username: str,
    submitted_at: str,
    image_id: str,
    image_name: str,
    image_width: int | None,
    image_height: int | None,
    num_detections: int,
    um_per_pixel: float | None,
    conf_threshold: float | None,
    boxes: list[dict[str, Any]],
) -> str:
    """Insert or update a submission keyed on (username, image_id). Returns the entry id."""
    boxes_json = json.dumps(boxes)
    with _lock, _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM community_detections WHERE username = ? AND image_id = ?",
            (username, image_id),
        ).fetchone()
        if existing:
            entry_id = existing["id"]
            conn.execute(
                """
                UPDATE community_detections
                SET submitted_at=?, image_name=?, image_width=?, image_height=?,
                    num_detections=?, um_per_pixel=?, conf_threshold=?, boxes=?
                WHERE id=?
                """,
                (submitted_at, image_name, image_width, image_height,
                 num_detections, um_per_pixel, conf_threshold, boxes_json, entry_id),
            )
        else:
            entry_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO community_detections
                  (id, username, submitted_at, image_id, image_name, image_width, image_height,
                   num_detections, um_per_pixel, conf_threshold, boxes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, username, submitted_at, image_id, image_name,
                 image_width, image_height, num_detections, um_per_pixel,
                 conf_threshold, boxes_json),
            )
        conn.commit()
    return entry_id


def list_recent(limit: int = 20, offset: int = 0, search: str = "") -> list[dict[str, Any]]:
    """Return recent submissions without the boxes field. Optionally filter by search term."""
    with _lock, _connect() as conn:
        if search:
            pattern = f"%{search}%"
            rows = conn.execute(
                """
                SELECT id, username, submitted_at, image_name, image_width, image_height,
                       num_detections, um_per_pixel, conf_threshold
                FROM community_detections
                WHERE image_name LIKE ? OR username LIKE ?
                ORDER BY submitted_at DESC
                LIMIT ? OFFSET ?
                """,
                (pattern, pattern, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, username, submitted_at, image_name, image_width, image_height,
                       num_detections, um_per_pixel, conf_threshold
                FROM community_detections
                ORDER BY submitted_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


def get_by_id(entry_id: str) -> dict[str, Any] | None:
    """Return full record including boxes, or None if not found."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM community_detections WHERE id = ?", (entry_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["boxes"] = json.loads(d["boxes"])
    return d


def get_stats() -> dict[str, int]:
    """Return aggregate statistics."""
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_submissions,
                SUM(num_detections) AS total_detections,
                COUNT(DISTINCT username) AS total_users,
                COUNT(DISTINCT image_name) AS total_images
            FROM community_detections
            """
        ).fetchone()
    return {
        "total_submissions": row["total_submissions"] or 0,
        "total_detections": row["total_detections"] or 0,
        "total_users": row["total_users"] or 0,
        "total_images": row["total_images"] or 0,
    }

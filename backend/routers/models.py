"""Models endpoint — list available .pt model files."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends

from backend.auth import get_current_user
from backend.config import settings

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(get_current_user)],
)


@router.get("")
async def list_models():
    """Scan the models directory and return all .pt files sorted newest-first."""
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for pt in sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = pt.stat()
        result.append({
            "name": pt.name,
            "path": str(pt),
            "size_mb": round(stat.st_size / 1_048_576, 2),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return result

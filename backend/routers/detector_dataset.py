"""Authenticated collection, immutable version inspection and explicit YOLO download."""
import json
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend.auth import get_current_user
from backend.config import settings
from backend.services import detector_dataset as service

router = APIRouter(prefix='/api/detector-dataset', tags=['detector-dataset'], dependencies=[Depends(get_current_user)])


class Box(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=100)
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = Field(default=1, ge=0, le=1)
    status: Literal['accepted', 'rejected', 'added']


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    source_job_id: str | None = Field(default=None, pattern=r'^[a-zA-Z0-9_-]{1,80}$')
    source_description: str = Field(default='', max_length=2000)
    boxes: list[Box] = Field(max_length=20000)


def translate(call, *args):
    try:
        return call(*args)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except service.DatasetError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/save')
def save(body: SaveRequest, username: str = Depends(get_current_user)):
    return translate(service.save_snapshot, body, username)


@router.get('')
def listing():
    result = []
    for path in service.versions():
        data = translate(service.read_version, path.parent.name)
        result.append({k: data[k] for k in ('version_id', 'created_at', 'created_by', 'parent_version')} | {
            'image_count': len(data['entries']), 'box_count': sum(len(e['boxes']) for e in data['entries'])})
    registry = settings.detector_split_registry
    coverage = None
    if registry.is_file():
        data = json.loads(registry.read_text())
        coverage = {'created_at': data['created_at'], 'roots': data['roots'], 'sources': len(data['source_memberships']), 'images': len(data['images'])}
    return {'versions': result, 'registry': coverage}


@router.get('/versions/{version}')
def detail(version: str):
    return translate(service.read_version, version)


@router.get('/versions/{version}/images/{entry_id}')
def original(version: str, entry_id: str):
    data = translate(service.read_version, version)
    entry = next((e for e in data['entries'] if e['entry_id'] == entry_id), None)
    if not entry:
        raise HTTPException(404, 'Fotografija nije u ovoj verziji.')
    return FileResponse(translate(service.object_path, entry))


@router.get('/versions/{version}/export')
def export(version: str, background_tasks: BackgroundTasks):
    path = translate(service.export_version, version)
    background_tasks.add_task(path.unlink, missing_ok=True)
    return FileResponse(path, media_type='application/zip', filename=f'detector-{version}-yolo.zip', background=background_tasks)

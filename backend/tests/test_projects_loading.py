import inspect
import sqlite3
import threading
from contextlib import closing

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import db_projects
from backend.auth import get_current_user
from backend.routers import projects


@pytest.fixture
def project_api(tmp_path, monkeypatch):
    database = tmp_path / 'preserved.db'
    monkeypatch.setattr(db_projects, '_DB_PATH', database)
    monkeypatch.setattr(db_projects, '_lock', threading.Lock())
    monkeypatch.setattr(db_projects, 'PROJECT_LIST_TIMEOUT_SECONDS', .05)
    db_projects.init_project_db()
    project_id = db_projects.create_project('Existing project', 'Keep me', 'tester')
    db_projects.add_image(project_id, 'existing-image', 'existing.jpg', 'tester')
    with closing(sqlite3.connect(database)) as con:
        original = con.execute('SELECT * FROM projects').fetchall(), con.execute('SELECT * FROM project_images').fetchall()
    app = FastAPI()
    app.include_router(projects.router)
    app.dependency_overrides[get_current_user] = lambda: 'tester'
    with TestClient(app) as client:
        yield client, database, project_id
    with closing(sqlite3.connect(database)) as con:
        assert (con.execute('SELECT * FROM projects').fetchall(), con.execute('SELECT * FROM project_images').fetchall()) == original


def test_project_list_returns_existing_data_without_caching_or_mutation(project_api):
    client, _, project_id = project_api
    response = client.get('/api/projects')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json()[0]['id'] == project_id
    assert response.json()[0]['image_count'] == 1
    assert not inspect.iscoroutinefunction(projects.list_projects)  # DB waits stay off the ASGI loop.


def test_python_lock_contention_returns_503_instead_of_pending_forever(project_api):
    client, _, _ = project_api
    db_projects._lock.acquire()
    try:
        response = client.get('/api/projects')
    finally:
        db_projects._lock.release()
    assert response.status_code == 503
    assert response.headers['retry-after'] == '2'
    assert 'zauzeta' in response.json()['detail']
    assert client.get('/api/projects').status_code == 200


def test_sqlite_exclusive_lock_returns_503_then_preserved_projects_after_unlock(project_api):
    client, database, project_id = project_api
    with closing(sqlite3.connect(database)) as writer:
        writer.execute('BEGIN EXCLUSIVE')
        try:
            response = client.get('/api/projects')
        finally:
            writer.rollback()
    assert response.status_code == 503
    assert 'zaključana' in response.json()['detail']
    response = client.get('/api/projects')
    assert response.status_code == 200
    assert response.json()[0]['id'] == project_id and response.json()[0]['image_count'] == 1

"""The workspace only shows detection boxes whose job belongs to the displayed photo."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.jobs.models import Job, JobStatus, JobType
from backend.routers import detection as router_module


@pytest.fixture
def client_and_jobs(tmp_path, monkeypatch):
    jobs = {}
    monkeypatch.setattr(router_module.job_manager, "get", lambda key: jobs.get(key))
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[get_current_user] = lambda: "tester"
    csv_path = tmp_path / "photo_detections.csv"
    csv_path.write_text("x1,y1,x2,y2,width,height,confidence,class\n10,20,60,80,50,60,.9,0\n")
    with TestClient(app) as client:
        yield client, jobs, csv_path


def completed(job_id, result):
    return Job(id=job_id, type=JobType.DETECTION, status=JobStatus.COMPLETED, result=result)


def test_boxes_report_the_photo_of_the_job(client_and_jobs):
    client, jobs, csv_path = client_and_jobs
    jobs["new"] = completed("new", {"image_id": "abc123", "csv_path": str(csv_path), "image_path": "data/uploads/abc123/p.jpg"})
    body = client.get("/api/detection/boxes/new").json()
    assert body["image_id"] == "abc123"
    assert body["boxes"] == [{"id": "box-0", "x1": 10., "y1": 20., "x2": 60., "y2": 80., "conf": .9, "status": "accepted"}]


def test_boxes_of_older_jobs_report_the_upload_folder_of_their_photo(client_and_jobs):
    client, jobs, csv_path = client_and_jobs
    jobs["old"] = completed("old", {"csv_path": str(csv_path), "image_path": "data/uploads/43e4922fba46/CAC_PDF_Ctrl_2_002.jpg"})
    assert client.get("/api/detection/boxes/old").json()["image_id"] == "43e4922fba46"

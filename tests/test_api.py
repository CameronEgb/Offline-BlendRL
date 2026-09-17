import pytest
from fastapi.testclient import TestClient

from src.api.app import app, jobs


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_environments(client):
    response = client.get("/api/environments")
    assert response.status_code == 200
    data = response.json()
    assert "environments" in data
    env_names = [e["name"] for e in data["environments"]]
    # Should include cartpole and should NOT include _base
    assert any("cartpole" in name for name in env_names)
    assert not any(name.startswith("_") for name in env_names)


def test_list_methods(client):
    response = client.get("/api/methods")
    assert response.status_code == 200
    data = response.json()
    assert "registered_agents" in data
    assert "method_styles" in data
    assert "ppo" in data["registered_agents"]


def test_list_experiments_filters_base(client):
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    exp_names = [e["name"] for e in data["experiments"]]
    # Base templates must be excluded
    for name in exp_names:
        assert not name.endswith("_base")
        assert "/_" not in name


def test_list_runs(client):
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert isinstance(data["runs"], list)


def test_get_metrics_not_found(client):
    response = client.get("/api/runs/nonexistent_group/nonexistent_exp/nonexistent_agent/metrics")
    assert response.status_code == 404
    assert response.json()["detail"] == "Metrics not found"


def test_cancel_nonexistent_job(client):
    response = client.post("/api/experiments/nonexistent-job-id/cancel")
    assert response.status_code == 404


def test_cancel_inactive_job(client):
    job_id = "test-completed-job"
    jobs[job_id] = {"status": "completed"}
    try:
        response = client.post(f"/api/experiments/{job_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "already" in data["message"]
    finally:
        jobs.pop(job_id, None)


def test_check_status_nonexistent(client):
    response = client.get("/api/experiments/fake-job-id/status")
    assert response.status_code == 404

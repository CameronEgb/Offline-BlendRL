"""TensorBoard server managed by the API for the GUI's TensorBoard tab."""

import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

import src.app.api.app as api
from src.app.api.app import app


@pytest.fixture
def client():
    yield TestClient(app)
    api._stop_tensorboard()


def test_status_when_not_started(client):
    api._stop_tensorboard()
    api._tensorboard.clear()
    body = client.get("/api/tensorboard").json()
    assert body["available"] is True
    assert (body["running"], body["ready"], body["url"]) == (False, False, None)


@pytest.mark.slow
def test_start_serves_tensorboard_then_stop_ends_it(client):
    started = client.post("/api/tensorboard/start").json()
    assert started["running"]
    url = started["url"]
    assert url.startswith("http://127.0.0.1:")

    deadline = time.time() + 60
    while not client.get("/api/tensorboard").json()["ready"]:
        assert time.time() < deadline, "TensorBoard did not become ready"
        time.sleep(0.5)
    with urllib.request.urlopen(url + "data/environment", timeout=5) as response:
        assert response.status == 200

    assert client.post("/api/tensorboard/start").json()["url"] == url  # starting again reuses the server

    stopped = client.post("/api/tensorboard/stop").json()
    assert not stopped["running"]
    assert "exit_code" not in stopped  # a deliberate stop is not reported as a crash
    with pytest.raises(OSError):
        urllib.request.urlopen(url + "data/environment", timeout=2)

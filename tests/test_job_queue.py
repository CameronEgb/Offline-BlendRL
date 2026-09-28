"""Job queue behind the GUI's "Add to queue" button: once started, queued jobs train one at a time, in order."""

import threading
import time
import uuid
from collections import defaultdict

import pytest
from fastapi.testclient import TestClient

import src.app.api.app as api
from frontend.model import BASE_EXPERIMENT, Config


@pytest.fixture
def queue(monkeypatch):
    """An isolated job table and queue, with training replaced by jobs the test finishes on demand."""
    monkeypatch.setattr(api, "jobs", {})
    monkeypatch.setattr(api, "queue_order", [])
    monkeypatch.setattr(api, "queue_state", {"running": False})
    started, release, teardown = [], defaultdict(threading.Event), threading.Event()

    def fake_run(job_id, req):
        table = api.jobs  # this test's table, even if the fixture has already been undone
        with api._jobs_lock:
            if table[job_id]["status"] == "cancelled":
                return
            table[job_id]["status"] = "running"
        started.append(table[job_id]["experiment_id"])
        while not (release[job_id].is_set() or teardown.is_set()):
            release[job_id].wait(0.05)
        with api._jobs_changed:
            table[job_id].update(status="completed", finished=time.time())
            api._jobs_changed.notify_all()

    monkeypatch.setattr(api, "run_experiment_task", fake_run)
    yield TestClient(api.app), started, release
    teardown.set()  # never leave the shared worker thread blocked on this test's jobs
    time.sleep(0.2)


def enqueue(client, name):
    return client.post("/api/experiments/launch", json={
        "experiment": BASE_EXPERIMENT, "overrides": Config(name=name).overrides(), "queue": True})


def wait_for(predicate, timeout=10):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "condition not met"
        time.sleep(0.05)


def test_queue_runs_jobs_one_at_a_time_in_order_with_moves_and_removals(queue):
    client, started, release = queue
    tag = uuid.uuid4().hex[:6]
    a, b, c = (enqueue(client, f"{n}_{tag}").json() for n in "abc")
    assert all(job["status"] == "queued" for job in (a, b, c))
    assert client.post("/api/queue/start").json()["running"]

    wait_for(lambda: started == [f"a_{tag}"])
    snapshot = client.get("/api/queue").json()
    assert [j["experiment_id"] for j in snapshot["active"]] == [f"a_{tag}"]
    assert [(j["experiment_id"], j["position"]) for j in snapshot["queued"]] == [(f"b_{tag}", 0), (f"c_{tag}", 1)]

    # A job using the same experiment ID as a queued one would overwrite its results
    assert enqueue(client, f"c_{tag}").status_code == 409

    assert client.post(f"/api/queue/{c['job_id']}/move", json={"position": 0}).json()["position"] == 0
    assert client.post(f"/api/experiments/{b['job_id']}/cancel").json()["status"] == "cancelled"
    assert [j["experiment_id"] for j in client.get("/api/queue").json()["queued"]] == [f"c_{tag}"]

    time.sleep(0.3)
    assert started == [f"a_{tag}"]  # nothing else starts while a job is running

    release[a["job_id"]].set()
    wait_for(lambda: started == [f"a_{tag}", f"c_{tag}"])
    release[c["job_id"]].set()
    wait_for(lambda: not client.get("/api/queue").json()["active"])

    final = {j["experiment_id"]: j["status"] for j in client.get("/api/queue").json()["finished"]}
    assert final == {f"a_{tag}": "completed", f"c_{tag}": "completed", f"b_{tag}": "cancelled"}
    assert started == [f"a_{tag}", f"c_{tag}"]  # the removed job never ran


def test_queued_job_waits_for_a_directly_launched_job(queue, monkeypatch):
    client, started, release = queue
    tag = uuid.uuid4().hex[:6]
    direct = client.post("/api/experiments/launch", json={
        "experiment": BASE_EXPERIMENT, "overrides": Config(name=f"direct_{tag}").overrides()}).json()
    wait_for(lambda: started == [f"direct_{tag}"])
    queued = enqueue(client, f"queued_{tag}").json()
    client.post("/api/queue/start")
    time.sleep(0.3)
    assert started == [f"direct_{tag}"]
    release[direct["job_id"]].set()
    wait_for(lambda: started == [f"direct_{tag}", f"queued_{tag}"])
    release[queued["job_id"]].set()


def test_only_queued_jobs_can_be_moved(queue):
    client, _, release = queue
    job = enqueue(client, f"solo_{uuid.uuid4().hex[:6]}").json()
    client.post("/api/queue/start")
    wait_for(lambda: client.get("/api/queue").json()["active"])
    assert client.post(f"/api/queue/{job['job_id']}/move", json={"position": 0}).status_code == 409
    release[job["job_id"]].set()


def test_queue_waits_for_start_pauses_on_request_and_pauses_itself_when_drained(queue):
    client, started, release = queue
    tag = uuid.uuid4().hex[:6]
    first, second = (enqueue(client, f"{n}_{tag}").json() for n in ("first", "second"))
    assert first["queue_running"] is False
    time.sleep(0.3)
    assert started == [] and client.get("/api/queue").json()["running"] is False  # adding does not start

    client.post("/api/queue/start")
    wait_for(lambda: started == [f"first_{tag}"])
    assert client.post("/api/queue/pause").json()["running"] is False
    release[first["job_id"]].set()
    wait_for(lambda: not client.get("/api/queue").json()["active"])
    time.sleep(0.3)
    assert started == [f"first_{tag}"]  # paused: the current job finished, the next did not start

    client.post("/api/queue/start")
    wait_for(lambda: started == [f"first_{tag}", f"second_{tag}"])
    release[second["job_id"]].set()
    wait_for(lambda: client.get("/api/queue").json()["running"] is False)  # drained, so it paused itself

    late = enqueue(client, f"late_{tag}").json()
    time.sleep(0.3)
    assert started == [f"first_{tag}", f"second_{tag}"]  # a job added after draining waits for Start
    assert client.post("/api/queue/start").json()["running"]
    wait_for(lambda: started[-1] == f"late_{tag}")
    release[late["job_id"]].set()


def test_starting_an_empty_queue_leaves_it_paused(queue):
    client, _, _ = queue
    assert client.post("/api/queue/start").json() == {"running": False, "queued": 0}

from __future__ import annotations

import json
import time

import pytest
from sanic_testing.reusable import ReusableClient

from idea_refiner.api.app import create_app

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


@pytest.fixture
def app(settings, execute):
    return create_app(settings, execute=execute, name=f"test_{time.time_ns()}")


_PORT = iter(range(48100, 48200))


@pytest.fixture
def client(app):
    """One server kept alive across requests, so background runs survive between calls."""
    c = ReusableClient(app, host="127.0.0.1", port=next(_PORT))
    with c:
        yield c


def _wait_done(client, run_id: str, tries: int = 100) -> dict:
    for _ in range(tries):
        _, r = client.get(f"/api/runs/{run_id}")
        if r.json["status"] in ("done", "error"):
            return r.json
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_health_boards_providers(app):
    _, r = app.test_client.get("/api/health")
    assert r.status == 200 and r.json["ok"] is True
    _, r = app.test_client.get("/api/boards")
    assert {b["name"] for b in r.json} >= {"startup", "architecture"}
    _, r = app.test_client.get("/api/boards/startup")
    assert len(r.json["agents"]) == 5
    _, r = app.test_client.get("/api/boards/missing")
    assert r.status == 404
    _, r = app.test_client.get("/api/providers")
    assert any(p["name"] == "ollama" and p["local"] for p in r.json)


def test_validate_board(app):
    _, r = app.test_client.post("/api/boards/validate", json={"yaml": "agents:\n  - {id: a, role: R, focus: F}\n"})
    assert r.status == 200 and r.json["board"]["agents"][0]["role"] == "R"
    _, r = app.test_client.post("/api/boards/validate", json={"yaml": "agents: []"})
    assert r.status == 422 and r.json["ok"] is False


def test_run_lifecycle_and_sse(client, settings):
    _, r = client.post("/api/runs", json={"idea": IDEA, "board": "startup", "title": "t"})
    assert r.status == 202
    run_id = r.json["id"]
    data = _wait_done(client, run_id)
    assert data["status"] == "done" and data["result"]["pitch"].startswith("Startup Founder says")
    assert (settings.runs_dir / run_id / "report.md").exists()

    _, r = client.get(f"/api/runs/{run_id}/events")
    assert r.status == 200 and r.content_type.startswith("text/event-stream")
    events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_done"
    assert sum(e["type"] == "agent_done" for e in events) == 11

    _, r = client.get(f"/api/runs/{run_id}/report.md")
    assert r.status == 200 and "## Refined pitch" in r.text
    _, r = client.get("/api/runs")
    assert r.json[0]["id"] == run_id
    _, r = client.delete(f"/api/runs/{run_id}")
    assert r.json["deleted"] is True
    _, r = client.get(f"/api/runs/{run_id}")
    assert r.status == 404


def test_run_validation_errors(app):
    _, r = app.test_client.post("/api/runs", json={"idea": "short"})
    assert r.status == 422
    _, r = app.test_client.post("/api/runs", json={"idea": IDEA, "board": "nope"})
    assert r.status == 422 and "unknown board" in r.json["error"]


def test_run_failure_is_reported(settings):
    def boom(agents, tasks):
        raise RuntimeError("provider down")

    app = create_app(settings, execute=boom, name=f"test_fail_{time.time_ns()}")
    with ReusableClient(app, host="127.0.0.1", port=next(_PORT)) as client:
        _, r = client.post("/api/runs", json={"idea": IDEA})
        data = _wait_done(client, r.json["id"])
    assert data["status"] == "error" and "provider down" in data["error"]

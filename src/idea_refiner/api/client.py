"""Thin httpx client for a remote idea-refiner API (used by ``refiner --server``)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from ..models import BoardSpec, Event, RunRequest, RunResult


class ApiError(Exception):
    pass


class Client:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _ok(self, r: httpx.Response) -> dict | list:
        if r.is_error:
            raise ApiError(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def health(self) -> dict:
        return self._ok(self.http.get("/api/health"))

    def providers(self) -> list[dict]:
        return self._ok(self.http.get("/api/providers"))

    def boards(self) -> list[BoardSpec]:
        return [BoardSpec.model_validate(b) for b in self._ok(self.http.get("/api/boards"))]

    def start_run(self, req: RunRequest) -> str:
        return self._ok(self.http.post("/api/runs", json=req.model_dump(mode="json", exclude_none=True)))["id"]

    def events(self, run_id: str) -> Iterator[Event]:
        """Follow the SSE stream until the run finishes."""
        with self.http.stream("GET", f"/api/runs/{run_id}/events", timeout=None) as r:
            if r.is_error:
                raise ApiError(f"events {run_id}: {r.status_code}")
            for line in r.iter_lines():
                if line.startswith("data:"):
                    yield Event.model_validate(json.loads(line[5:].strip()))

    def run(self, run_id: str) -> dict:
        return self._ok(self.http.get(f"/api/runs/{run_id}"))

    def result(self, run_id: str) -> RunResult:
        data = self.run(run_id)
        if data["status"] == "error":
            raise ApiError(data["error"])
        if data["result"] is None:
            raise ApiError(f"run {run_id} is {data['status']}")
        return RunResult.model_validate(data["result"])

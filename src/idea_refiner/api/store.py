"""In-memory run registry with fan-out to SSE subscribers, persisted to ``runs_dir`` on completion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .. import report
from ..models import Event, RunRequest, RunResult

Status = Literal["queued", "running", "done", "error"]
TERMINAL = {"run_done", "error"}


@dataclass
class RunState:
    id: str
    request: RunRequest
    status: Status = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    events: list[Event] = field(default_factory=list)
    result: RunResult | None = None
    error: str | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    task: asyncio.Task | None = None

    def push(self, e: Event) -> None:
        self.events.append(e)
        for q in self.subscribers:
            q.put_nowait(e)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error")

    def summary(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "board": self.request.board,
            "title": self.request.title,
            "idea": self.request.idea[:200],
            "created_at": self.created_at.isoformat(),
            "events": len(self.events),
            "error": self.error,
            "seconds": self.result.seconds if self.result else None,
        }


class RunStore:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs: dict[str, RunState] = {}
        for r in report.list_runs(runs_dir):  # previously saved runs show up as done
            st = RunState(
                id=r.id,
                request=RunRequest(idea=r.idea, board=r.board, title=r.title),
                status="done",
                created_at=r.created_at,
            )
            st.result = r
            self.runs[r.id] = st

    def create(self, req: RunRequest) -> RunState:
        st = RunState(id=uuid4().hex[:12], request=req)
        self.runs[st.id] = st
        return st

    def get(self, run_id: str) -> RunState | None:
        return self.runs.get(run_id)

    def list(self) -> list[RunState]:
        return sorted(self.runs.values(), key=lambda s: s.created_at, reverse=True)

    def delete(self, run_id: str) -> bool:
        st = self.runs.pop(run_id, None)
        if st is None:
            return False
        d = self.runs_dir / run_id
        for f in d.glob("*") if d.is_dir() else []:
            f.unlink()
        if d.is_dir():
            d.rmdir()
        return True

    def finish(self, st: RunState, result: RunResult) -> None:
        st.result, st.status = result, "done"
        report.save(result, self.runs_dir)

    def fail(self, st: RunState, error: str) -> None:
        st.error, st.status = error, "error"

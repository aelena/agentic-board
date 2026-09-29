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
            "board": self.result.board if self.result else self.request.board,
            "project": self.request.project,
            "title": self.request.title,
            "idea": self.request.idea[:200],
            "created_at": self.created_at.isoformat(),
            "events": len(self.events),
            "error": self.error,
            "seconds": self.result.seconds if self.result else None,
        }


class RunStore:
    """Runs of plain requests live in ``runs_dir``; runs of a project in ``<projects_dir>/<name>/runs``."""

    def __init__(self, runs_dir: Path, projects_dir: Path | None = None):
        self.runs_dir, self.projects_dir = runs_dir, projects_dir
        self.runs: dict[str, RunState] = {}
        self.dirs: dict[str, Path] = {}
        roots = [runs_dir]
        if projects_dir and projects_dir.is_dir():
            roots += [p / "runs" for p in sorted(projects_dir.iterdir()) if (p / "runs").is_dir()]
        for root in roots:
            for r in report.list_runs(root):  # previously saved runs show up as done
                st = RunState(
                    id=r.id,
                    request=RunRequest(idea=r.idea, board=r.board, project=r.project, title=r.title),
                    status="done",
                    created_at=r.created_at,
                )
                st.result = r
                self.runs[r.id] = st
                self.dirs[r.id] = root / r.id

    def root_for(self, result: RunResult) -> Path:
        if result.project and self.projects_dir:
            return self.projects_dir / result.project / "runs"
        return self.runs_dir

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
        d = self.dirs.pop(run_id, self.runs_dir / run_id)
        for f in d.glob("*") if d.is_dir() else []:
            f.unlink()
        if d.is_dir():
            d.rmdir()
        return True

    def finish(self, st: RunState, result: RunResult) -> None:
        st.result, st.status = result, "done"
        self.dirs[result.id] = report.save(result, self.root_for(result))

    def fail(self, st: RunState, error: str) -> None:
        st.error, st.status = error, "error"

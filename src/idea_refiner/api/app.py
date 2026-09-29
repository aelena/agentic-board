"""Sanic REST API.

    GET  /api/health
    GET  /api/providers
    GET  /api/boards                 GET /api/boards/<name>        POST /api/boards/validate {yaml}
    POST /api/runs {RunRequest}      -> 202 {id}
    GET  /api/runs                   GET /api/runs/<id>            DELETE /api/runs/<id>
    GET  /api/runs/<id>/events       (Server-Sent Events, replays history then streams live)
    GET  /api/runs/<id>/report.md

If ``web/dist`` exists (built Svelte app) it is served at ``/``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError
from sanic import Request, Sanic
from sanic.exceptions import NotFound
from sanic.response import HTTPResponse, json, text
from sanic_ext import Extend

from .. import __version__, boards, report
from ..boards import BoardError
from ..config import PROVIDERS, Settings
from ..config import settings as default_settings
from ..engine import Execute, crew_execute, run_board, warm_up
from ..models import Event, RunRequest
from .store import TERMINAL, RunState, RunStore

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


def _providers() -> list[dict]:
    return [
        {
            "name": p.name,
            "default_model": p.default_model,
            "key_env": p.key_env,
            "key_present": p.key_present,
            "local": p.local,
            "base_url": p.base_url,
        }
        for p in PROVIDERS.values()
    ]


def _sse(e: Event) -> str:
    return f"event: {e.type}\ndata: {e.model_dump_json()}\n\n"


def create_app(
    settings: Settings = default_settings, execute: Execute = crew_execute, name: str = "idea_refiner"
) -> Sanic:
    app = Sanic(name)
    app.config.CORS_ORIGINS = "*"
    app.config.OAS = False
    Extend(app)
    app.ctx.settings = settings
    app.ctx.store = RunStore(settings.runs_dir)
    app.ctx.execute = execute

    @app.after_server_start
    async def _warm(app_, _loop):
        # crewai import + default LLM client, off the event loop, so the first run does not pay for it
        app_.add_task(asyncio.to_thread(warm_up, settings))

    async def _run(st: RunState) -> None:
        store: RunStore = app.ctx.store
        loop = asyncio.get_running_loop()
        st.status = "running"
        emit = lambda e: loop.call_soon_threadsafe(st.push, e)  # noqa: E731 - runs in the worker thread
        try:
            board = boards.load_board(st.request.board, settings)
            result = await asyncio.to_thread(
                run_board,
                board,
                st.request.idea,
                request_llm=st.request.llm,
                phases=st.request.phases,
                title=st.request.title,
                refine=st.request.refine,
                settings=settings,
                emit=emit,
                execute=app.ctx.execute,
                run_id=st.id,
            )
            store.finish(st, result)
        except Exception as e:  # noqa: BLE001
            store.fail(st, f"{type(e).__name__}: {e}")
            if not st.events or st.events[-1].type != "error":
                st.push(Event(type="error", run_id=st.id, text=st.error))

    def _get(run_id: str) -> RunState:
        st = app.ctx.store.get(run_id)
        if st is None:
            raise NotFound(f"run {run_id} not found")
        return st

    @app.get("/api/health")
    async def health(_: Request):
        return json({"ok": True, "version": __version__, "provider": settings.resolved_provider()})

    @app.get("/api/providers")
    async def providers(_: Request):
        return json(_providers())

    @app.get("/api/boards")
    async def list_boards(_: Request):
        return json([b.model_dump(mode="json") for b in boards.list_boards(settings)])

    @app.get("/api/boards/<name>")
    async def get_board(_: Request, name: str):
        try:
            return json(boards.load_board(name, settings).model_dump(mode="json"))
        except BoardError as e:
            raise NotFound(str(e)) from e

    @app.post("/api/boards/validate")
    async def validate_board(request: Request):
        body = request.json or {}
        try:
            return json(
                {
                    "ok": True,
                    "board": boards.parse_board(body.get("yaml", ""), body.get("name", "inline")).model_dump(
                        mode="json"
                    ),
                }
            )
        except BoardError as e:
            return json({"ok": False, "error": str(e)}, status=422)

    @app.post("/api/runs")
    async def start_run(request: Request):
        try:
            req = RunRequest.model_validate(request.json or {})
            boards.load_board(req.board, settings)  # fail fast on unknown board
        except ValidationError as e:
            return json({"error": e.errors(include_url=False)}, status=422)
        except BoardError as e:
            return json({"error": str(e)}, status=422)
        st = app.ctx.store.create(req)
        # asyncio.create_task rather than app.add_task: the latter only queues until server start
        # when the loop is not the one Sanic owns (e.g. under test clients). Keep a reference.
        st.task = asyncio.create_task(_run(st))
        return json({"id": st.id, "status": st.status}, status=202)

    @app.get("/api/runs")
    async def list_runs(_: Request):
        return json([s.summary() for s in app.ctx.store.list()])

    @app.get("/api/runs/<run_id>")
    async def get_run(_: Request, run_id: str):
        st = _get(run_id)
        return json({**st.summary(), "result": st.result.model_dump(mode="json") if st.result else None})

    @app.delete("/api/runs/<run_id>")
    async def delete_run(_: Request, run_id: str):
        return json({"deleted": app.ctx.store.delete(run_id)})

    @app.get("/api/runs/<run_id>/report.md")
    async def run_report(_: Request, run_id: str):
        st = _get(run_id)
        if st.result is None:
            return text("run not finished", status=409)
        return text(report.to_markdown(st.result), content_type="text/markdown; charset=utf-8")

    @app.get("/api/runs/<run_id>/events")
    async def run_events(request: Request, run_id: str):
        st = _get(run_id)
        resp = await request.respond(
            content_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        q = st.subscribe()  # subscribe first so nothing is lost between replay and live
        try:
            for e in list(st.events):
                await resp.send(_sse(e))
            if st.finished or any(e.type in TERMINAL for e in st.events):
                return await resp.eof()
            while True:
                try:
                    e = await asyncio.wait_for(q.get(), timeout=15)
                except TimeoutError:
                    await resp.send(": keepalive\n\n")
                    continue
                await resp.send(_sse(e))
                if e.type in TERMINAL:
                    break
            await resp.eof()
        finally:
            st.unsubscribe(q)

    if WEB_DIST.is_dir():
        app.static("/", WEB_DIST, index="index.html", name="web")

        @app.exception(NotFound)
        async def spa_fallback(request: Request, _):
            if request.path.startswith("/api/"):
                return json({"error": "not found"}, status=404)
            return await app.ctx.spa_index(request)

        async def spa_index(_: Request) -> HTTPResponse:
            return HTTPResponse((WEB_DIST / "index.html").read_bytes(), content_type="text/html")

        app.ctx.spa_index = spa_index

    return app


def serve(
    host: str | None = None,
    port: int | None = None,
    dev: bool = False,
    workers: int = 1,
    settings: Settings = default_settings,
) -> None:
    app = create_app(settings)
    kwargs = dict(host=host or settings.host, port=port or settings.port, access_log=dev)
    if dev:
        app.run(dev=True, **kwargs)
    elif workers <= 1 or sys.platform == "win32":  # no fork on Windows
        app.run(single_process=True, **kwargs)
    else:
        app.run(workers=workers, **kwargs)

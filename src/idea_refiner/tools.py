"""Agent tools, referenced by name from board YAML (``tools: [web_search]``).

Only names in ``REGISTRY`` can be used: YAML never names a Python class or a script, so a board cannot
make the server run arbitrary code. Each tool declares the env vars it needs; a run checks them before
the first LLM call. Tool output is capped (web pages are long) and ``scrape`` refuses private,
loopback and link-local addresses so a prompt-injected URL cannot reach internal services.

Every call is reported: CrewAI emits tool-usage events on its global bus, and ``route`` sends each one
to the run and seat that owns the calling agent.

MCP servers declared in a board (``mcp_servers:``) are resolved into tools here and wrapped exactly like
the built-in ones, so they share the same budget, breaker, output cap and reporting.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

OUTPUT_LIMIT = 6000  # characters of tool output an agent gets back


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    env: tuple[str, ...]
    factory: Callable[[], Any]  # -> crewai BaseTool, imported lazily


def _serper():
    from crewai_tools import SerperDevTool

    return SerperDevTool(n_results=6)


def _scrape():
    from crewai_tools import ScrapeWebsiteTool

    return ScrapeWebsiteTool()


REGISTRY: dict[str, ToolDef] = {
    t.name: t
    for t in (
        ToolDef(
            "web_search", "Google search results (title, link, snippet) via serper.dev", ("SERPER_API_KEY",), _serper
        ),
        ToolDef("scrape", "Read the text of a public web page", (), _scrape),
    )
}
TOOL_NAMES = frozenset(REGISTRY)


def installed() -> bool:
    try:
        import crewai_tools  # noqa: F401
    except ImportError:
        return False
    return True


def missing(names: list[str]) -> list[str]:
    """Why these tools cannot run here, one line per problem; empty when they all can."""
    problems = [f"unknown tool '{n}' (known: {', '.join(sorted(REGISTRY))})" for n in names if n not in REGISTRY]
    if names and not installed():
        problems.append('agent tools need the tools extra: pip install "idea-refiner[tools]"')
    for n in sorted({n for n in names if n in REGISTRY}):
        problems += [
            f"tool '{n}' needs {v} in the environment or .env" for v in REGISTRY[n].env if not os.environ.get(v)
        ]
    return problems


def public_url(url: str) -> bool:
    """True when every address the host resolves to is public."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        infos = socket.getaddrinfo(u.hostname, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


URL = re.compile(r"https?://[^\s\"'<>)\]}]+")
BREAK_AFTER = 2  # failures of one tool before it is switched off for the rest of the job


def urls_in(text: str) -> set[str]:
    return {u.rstrip(".,;:") for u in URL.findall(text)}


@dataclass
class Budget:
    """Shared by one job's tools. Prompts alone do not stop a model from calling a failing tool 30 times,
    so the cap and the breaker are enforced here."""

    limit: int
    used: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    broken: dict[str, str] = field(default_factory=dict)  # tool name -> last error


def build(names: list[str], budget: int = 6, extra: list[tuple[str, Any]] | None = None) -> list:
    """Instantiate the named tools for one job, plus ``extra`` (name, ready-made tool) pairs (MCP): output capped,
    ``scrape`` URL-guarded, at most ``budget`` calls in total across all of them, and a tool that failed
    twice answers with a stop message instead of being called."""
    from crewai.tools import BaseTool

    shared = Budget(budget)

    class Capped(BaseTool):
        inner: Any
        guard_urls: bool = False

        def _run(self, **kwargs: Any) -> str:
            if self.name in shared.broken:
                return (
                    f"{self.name} is unavailable ({shared.broken[self.name]}). Do not call it again. Answer with "
                    "what you have and say plainly what you could not verify."
                )
            if shared.used >= shared.limit:
                return (
                    f"Tool budget used up ({shared.limit} calls). Answer now with what you have and say plainly "
                    "what you could not verify."
                )
            shared.used += 1
            if self.guard_urls:
                url = str(kwargs.get("website_url") or kwargs.get("url") or "")
                if not public_url(url):
                    return f"Refused: {url!r} is not a public http(s) address."
            try:
                out = str(self.inner.run(**kwargs))
            except Exception as e:
                shared.failures[self.name] = shared.failures.get(self.name, 0) + 1
                if shared.failures[self.name] >= BREAK_AFTER:
                    shared.broken[self.name] = f"{type(e).__name__}: {str(e)[:120]}"
                raise
            return out if len(out) <= OUTPUT_LIMIT else out[:OUTPUT_LIMIT] + "\n[output truncated]"

    logging.getLogger("crewai_tools").setLevel(logging.CRITICAL)  # failures reach us as tool events already
    inners = [(i.name, i, n == "scrape") for n in names for i in [REGISTRY[n].factory()]]
    inners += [(name, t, False) for name, t in extra or []]
    return [
        Capped(name=name, description=i.description, args_schema=i.args_schema, inner=i, guard_urls=guard)
        for name, i, guard in inners
    ]


# --- MCP servers -------------------------------------------------------------------------------------

_VAR = re.compile(r"\$\{(\w+)\}")


def interpolate(value: str) -> str:
    """Replace ``${VAR}`` with the environment variable; a missing one is an error, not an empty string."""

    def sub(m: re.Match) -> str:
        if (v := os.environ.get(m.group(1))) is None:
            raise ToolError(f"environment variable {m.group(1)} is not set")
        return v

    return _VAR.sub(sub, value)


def mcp_problems(name: str, spec, from_file: bool, allow_commands: bool) -> list[str]:
    """Why an MCP server cannot be used, checked before a run starts."""
    problems = []
    if spec.command:
        if not from_file and not allow_commands:
            problems.append(
                f"MCP server '{name}' runs a local command, which is only allowed for boards and projects read "
                "from disk (set REFINER_ALLOW_MCP_COMMANDS=true to allow it for inline boards)"
            )
        elif not shutil.which(spec.command):
            problems.append(f"MCP server '{name}': command '{spec.command}' not found on PATH")
    for v in [*spec.env.values(), *spec.headers.values(), spec.url or ""]:
        try:
            interpolate(v)
        except ToolError as e:
            problems.append(f"MCP server '{name}': {e}")
    return problems


def mcp_config(spec):
    from crewai.mcp.config import MCPServerHTTP, MCPServerSSE, MCPServerStdio

    if spec.command:
        env = {**os.environ, **{k: interpolate(v) for k, v in spec.env.items()}} if spec.env else None
        return MCPServerStdio(command=spec.command, args=[interpolate(a) for a in spec.args], env=env)
    headers = {k: interpolate(v) for k, v in spec.headers.items()} or None
    url = interpolate(spec.url)
    if spec.transport == "sse":
        return MCPServerSSE(url=url, headers=headers)
    return MCPServerHTTP(url=url, headers=headers, streamable=True)


def resolve_mcp(spec) -> list:
    """Connect to an MCP server, list its tools and return them as CrewAI tools (not yet capped)."""
    from crewai.events.event_listener import event_listener
    from crewai.mcp.tool_resolver import MCPToolResolver
    from crewai.utilities.logger import Logger

    # CrewAI's console listener starts verbose and only a running Crew turns it off; without this every
    # connection prints "MCP Connection" panels into the CLI and the server log.
    event_listener.formatter.verbose = False
    return MCPToolResolver(agent=None, logger=Logger(verbose=False)).resolve([mcp_config(spec)])


def mcp_tool_name(tool) -> str:
    """The name the server itself uses for a tool (CrewAI may prefix it with the server name)."""
    return getattr(tool, "original_tool_name", None) or tool.name


def mcp_display_name(server: str, tool) -> str:
    """``<server>_<tool>``, the name the model and the reports see; limited to what LLM APIs accept."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}_{mcp_tool_name(tool)}")[:64]


# --- routing CrewAI tool events to the seat that made the call --------------------------------------

_routes: dict[str, Callable[[str, Any], None]] = {}  # crewai agent id -> handler(kind, event)
_lock = threading.Lock()
_listening = False


def _listen() -> None:
    global _listening
    with _lock:
        if _listening:
            return
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.tool_usage_events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        def handler(kind: str):
            def on(_source, event) -> None:
                if (fn := _routes.get(str(event.agent_id))) is not None:
                    fn(kind, event)

            return on

        crewai_event_bus.on(ToolUsageStartedEvent)(handler("started"))
        crewai_event_bus.on(ToolUsageFinishedEvent)(handler("finished"))
        crewai_event_bus.on(ToolUsageErrorEvent)(handler("error"))
        _listening = True


def route(agent_id: str, fn: Callable[[str, Any], None]) -> Callable[[], None]:
    """Send tool events of ``agent_id`` to ``fn`` until the returned callable is called."""
    _listen()
    _routes[agent_id] = fn
    return lambda: _routes.pop(agent_id, None)

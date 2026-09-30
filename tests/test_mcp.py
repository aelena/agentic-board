from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import verdict_block, wants_verdict

from idea_refiner import boards, tools
from idea_refiner.boards import BoardError
from idea_refiner.engine import RunError, run_board
from idea_refiner.models import McpServerSpec

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."
KB = Path(__file__).parent / "mcp_kb_server.py"
KB_SPEC = {"command": sys.executable, "args": [str(KB)]}


def board_yaml(tmp_path, allow="[lookup]", extra="") -> Path:
    f = tmp_path / "kb-board.yaml"
    exe = sys.executable.replace("\\", "/")
    f.write_text(
        f"""
mcp_servers:
  kb: {{command: "{exe}", args: ["{str(KB).replace(chr(92), "/")}"]}}
agents:
  - {{id: vc, role: Hardened Venture Capitalist, focus: market fit, mcp: [{{server: kb, allow: {allow}}}]}}
  - {{id: cto, role: Scaling CTO, focus: scalability}}
phases: [hostile]
{extra}
""",
        encoding="utf-8",
    )
    return f


def test_specs_validate():
    with pytest.raises(ValueError, match="exactly one of"):
        McpServerSpec(command="x", url="https://y")
    with pytest.raises(ValueError, match="exactly one of"):
        McpServerSpec()
    with pytest.raises(BoardError, match="unknown MCP server"):
        boards.parse_board("agents:\n  - {id: a, role: R, focus: F, mcp: [nope]}\n")
    b = boards.parse_board(
        "mcp_servers: {kb: {url: 'https://kb.example/mcp'}}\nagents:\n  - {id: a, role: R, focus: F, mcp: [kb]}\n"
    )
    assert b.agents[0].mcp[0].server == "kb" and b.agents[0].mcp[0].allow is None


def test_resolve_a_real_stdio_server_and_call_its_tool():
    found = tools.resolve_mcp(McpServerSpec(**KB_SPEC))
    assert {tools.mcp_tool_name(t) for t in found} == {"lookup", "delete_all"}
    lookup = next(t for t in found if tools.mcp_tool_name(t) == "lookup")
    (capped,) = tools.build([], budget=2, extra=[("kb_lookup", lookup)])
    assert capped.name == "kb_lookup"
    assert "churn is 4% a month" in capped._run(topic="pricing")


def test_seat_gets_only_allowed_mcp_tools(tmp_path, settings):
    board = boards.load_board(str(board_yaml(tmp_path)))
    seen = {}

    def exe(agent, task):
        seen[agent.role] = ([t.name for t in agent.tools or []], task.description)
        return "critique" + (verdict_block() if wants_verdict(task) else "")

    run_board(board, IDEA, settings=settings, execute=exe)
    names, desc = seen["Hardened Venture Capitalist"]
    assert names == ["kb_lookup"] and "You have tools (mcp:kb)" in desc  # delete_all filtered out
    assert seen["Scaling CTO"][0] == []


def test_command_servers_need_a_board_from_disk(settings, monkeypatch):
    text = board_yaml(Path(settings.runs_dir.parent)).read_text(encoding="utf-8")
    inline = boards.parse_board(text)  # as if it came in over the API: no source file
    assert inline.source is None
    with pytest.raises(RunError, match="only allowed for boards and projects read from disk"):
        run_board(inline, IDEA, settings=settings, execute=lambda a, t: "x")
    settings.allow_mcp_commands = True
    run_board(inline, IDEA, settings=settings, execute=lambda a, t: "x" + (verdict_block() if wants_verdict(t) else ""))


def test_preflight_catches_missing_command_and_env(tmp_path, settings, monkeypatch):
    monkeypatch.delenv("KB_TOKEN", raising=False)
    f = tmp_path / "b.yaml"
    f.write_text(
        "mcp_servers:\n  kb: {command: no-such-binary-xyz}\n  crm: {url: 'https://crm.example/mcp', headers: {Authorization: 'Bearer ${KB_TOKEN}'}}\n"
        "agents:\n  - {id: a, role: R, focus: F, mcp: [kb, crm]}\nphases: [hostile]\n",
        encoding="utf-8",
    )
    with pytest.raises(RunError) as e:
        run_board(boards.load_board(str(f)), IDEA, settings=settings, execute=lambda a, t: "x")
    assert "command 'no-such-binary-xyz' not found" in str(e.value) and "KB_TOKEN is not set" in str(e.value)


def test_unreachable_server_costs_the_seat_a_tool_not_the_run(tmp_path, settings):
    f = tmp_path / "down.yaml"
    f.write_text(
        "mcp_servers:\n  down: {url: 'http://127.0.0.1:9/mcp'}\n"
        "agents:\n  - {id: a, role: R, focus: F, mcp: [down]}\nphases: [hostile]\n",
        encoding="utf-8",
    )
    events = []
    r = run_board(
        boards.load_board(str(f)),
        IDEA,
        settings=settings,
        emit=events.append,
        execute=lambda a, t: "x" + verdict_block(),
    )
    assert r.phase("hostile").outputs[0].verdict
    err = next(e for e in events if e.type == "tool")
    assert err.data["status"] == "error" and "MCP server 'down' unavailable" in err.text

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from conftest import verdict_block, wants_verdict

from idea_refiner import boards, tools
from idea_refiner.boards import BoardError
from idea_refiner.engine import RunError, run_board

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


def test_unknown_tool_is_a_board_error():
    with pytest.raises(BoardError, match="unknown tool"):
        boards.parse_board("agents:\n  - {id: a, role: R, focus: F, tools: [shell]}\n")
    b = boards.parse_board("agents:\n  - {id: a, role: R, focus: F, tools: [web_search, scrape]}\n")
    assert b.agents[0].tools == ["web_search", "scrape"]


def test_missing_key_fails_before_any_llm_call(startup, settings, monkeypatch):
    import crewai_tools  # noqa: F401 - its import runs load_dotenv(); delete the key after that

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    startup.agents[0].tools = ["web_search"]
    called = []
    with pytest.raises(RunError, match="needs SERPER_API_KEY"):
        run_board(startup, IDEA, settings=settings, execute=lambda a, t: called.append(1) or "x")
    assert not called
    # tools only matter in tool_phases: a coaching-only run does not need the key
    run_board(startup, IDEA, settings=settings, execute=lambda a, t: "x", phases=["coaching"])


def test_public_url_guard():
    assert not tools.public_url("http://127.0.0.1:8000/admin")
    assert not tools.public_url("http://169.254.169.254/latest/meta-data")
    assert not tools.public_url("http://10.0.0.5/")
    assert not tools.public_url("file:///etc/passwd")
    assert not tools.public_url("http://localhost/")


def test_capped_tool_truncates_and_refuses_private_urls(monkeypatch):
    class Fake:
        name, description, args_schema = "Read website content", "fake", None

        def run(self, **kw):
            return "x" * (tools.OUTPUT_LIMIT + 50)

    monkeypatch.setitem(tools.REGISTRY, "scrape", tools.ToolDef("scrape", "", (), lambda: Fake()))
    (t,) = tools.build(["scrape"])
    assert t._run(website_url="http://127.0.0.1/").startswith("Refused")
    monkeypatch.setattr(tools, "public_url", lambda u: True)
    out = t._run(website_url="https://example.com")
    assert out.endswith("[output truncated]") and len(out) < tools.OUTPUT_LIMIT + 30


def test_tool_calls_are_routed_to_their_seat(startup, settings, monkeypatch):
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent, ToolUsageStartedEvent

    monkeypatch.setenv("SERPER_API_KEY", "test")
    startup.agents[0].tools = ["web_search"]  # only the VC gets a tool
    seen = {}

    def exe(agent, task):
        seen[agent.role] = (task.description, [t.name for t in agent.tools or []], agent.max_iter)
        if agent.tools:  # what CrewAI does while the agent works
            args = {"search_query": "feature flag pricing"}
            base = dict(tool_name="search_the_internet_with_serper", tool_args=args, agent_id=str(agent.id))
            crewai_event_bus.emit(agent, ToolUsageStartedEvent(**base))
            t0 = datetime.now(UTC)
            done = ToolUsageFinishedEvent(**base, started_at=t0, finished_at=t0 + timedelta(seconds=1.5), output="r")
            crewai_event_bus.emit(agent, done)
            crewai_event_bus.flush()  # kickoff() does this before returning
        return "critique" + (verdict_block() if wants_verdict(task) else "")

    events = []
    r = run_board(startup, IDEA, settings=settings, execute=exe, emit=events.append, phases=["hostile", "coaching"])
    vc = r.phase("hostile").outputs[0]
    assert [(c.tool, c.seconds) for c in vc.tool_calls] == [("search_the_internet_with_serper", 1.5)]
    assert "search_query='feature flag pricing'" in vc.tool_calls[0].args
    assert all(not o.tool_calls for o in r.phase("hostile").outputs[1:])
    desc, names, max_iter = seen["Hardened Venture Capitalist"]
    assert "You have tools (web_search)" in desc and "untrusted data" in desc and max_iter == 8
    assert names == ["Search the internet with Serper"]
    assert seen["Venture Capitalist (Coach)"][1] == []  # no tools outside tool_phases
    assert seen["Scaling CTO"][2] == 3
    statuses = [(e.agent_id, e.data["status"]) for e in events if e.type == "tool"]
    assert statuses == [("vc", "finished")]  # completed calls only: handler order is not guaranteed
    assert "*Tools used: search_the_internet_with_serper(" in vc.as_markdown()
    assert not tools._routes  # routes are removed when the job ends

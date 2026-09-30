from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from conftest import verdict_block, wants_verdict

from idea_refiner import report
from idea_refiner.engine import RunError, run_board
from idea_refiner.models import RefineSpec

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


@pytest.fixture(autouse=True)
def serper_key(monkeypatch):
    import crewai_tools  # noqa: F401 - imports load .env; set the key after that

    monkeypatch.setenv("SERPER_API_KEY", "test")


def search_result(agent, output: str | None = None, error: str | None = None) -> None:
    """Emit the tool events CrewAI would emit for one web_search call by ``agent``."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageErrorEvent, ToolUsageFinishedEvent

    base = dict(tool_name="search_the_internet_with_serper", tool_args={"search_query": "q"}, agent_id=str(agent.id))
    now = datetime.now(UTC)
    if error:
        crewai_event_bus.emit(agent, ToolUsageErrorEvent(**base, error=error))
    else:
        crewai_event_bus.emit(agent, ToolUsageFinishedEvent(**base, started_at=now, finished_at=now, output=output))
    crewai_event_bus.flush()


def researching(scores=(4, 8), found="https://example.com/report"):
    seen: list[tuple[str, str, list[str]]] = []

    def exe(agent, task):
        seen.append((agent.role, task.description, [t.name for t in agent.tools or []]))
        if task.description.startswith("Research this idea"):
            concern = re.search(r"whose concern is (.+?)\.\n", task.description).group(1)
            search_result(agent, output=f'{{"organic": [{{"link": "{found}"}}]}}')
            return f"- fact about {concern} ({found})"
        if wants_verdict(task):
            m = re.search(r"This is revision (\d+)", task.description)
            return "critique" + verdict_block("pivot", scores[int(m.group(1)) - 1 if m else 0])
        return "Pitch.\n\n## Revised idea\n" + "A sharper revised idea with every detail kept. " * 2

    return exe, seen


def test_research_is_opt_in(startup, settings):
    exe, seen = researching()
    r = run_board(startup, IDEA, settings=settings, execute=exe)
    assert r.phase("research") is None and not any(d.startswith("Research this idea") for _, d, _ in seen)


def test_one_researcher_per_seat_and_each_critic_gets_its_own_briefing(startup, settings):
    exe, seen = researching()
    r = run_board(startup, IDEA, settings=settings, execute=exe, research=True)
    research = r.phase("research")
    assert [o.agent_id for o in research.outputs] == ["vc", "cto", "pm", "gc", "ciso"]
    assert research.outputs[0].role == "Research Analyst for the Hardened Venture Capitalist"
    tasks = [(role, d, names) for role, d, names in seen if d.startswith("Research this idea")]
    assert len(tasks) == 5 and all(
        names == ["Search the internet with Serper", "Read website content"] for *_, names in tasks
    )
    vc = next(d for role, d, _ in seen if role == "Hardened Venture Capitalist")
    assert "## Research briefing prepared for you\n- fact about lack of defensibility and market fit" in vc
    assert "technical debt" not in vc.split("## Idea")[0]  # the CTO's briefing stays with the CTO
    coach = next(d for role, d, _ in seen if role == "Venture Capitalist (Coach)")
    assert "fact about lack of defensibility" in coach
    assert "## Research briefings" in report.to_markdown(r)


def test_research_runs_once_across_revisions(startup, settings):
    exe, seen = researching(scores=(4, 8))
    r = run_board(startup, IDEA, settings=settings, execute=exe, research=True, refine=RefineSpec(max_iterations=2))
    assert len(r.iterations) == 2 and len(r.all("research")) == 1
    rev2 = next(d for role, d, _ in seen if role == "Hardened Venture Capitalist" and "revision 2" in d)
    assert "Research briefing prepared for you" in rev2  # briefings carry over to later revisions


def test_research_needs_its_tools(startup, settings, monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY")
    with pytest.raises(RunError, match="needs SERPER_API_KEY"):
        run_board(startup, IDEA, settings=settings, execute=lambda a, t: "x", research=True)
    startup.research.tools = ["scrape"]  # a board can research with other tools
    exe, _ = researching()
    assert run_board(startup, IDEA, settings=settings, execute=exe, research=True).phase("research")


def test_briefing_from_failed_searches_is_discarded(startup, settings):
    def exe(agent, task):
        if task.description.startswith("Research this idea"):
            search_result(agent, error="403 Client Error: Forbidden")
            return "- LaunchDarkly costs $8 per user (https://launchdarkly.com/pricing)"  # invented
        return "critique" + (verdict_block() if wants_verdict(task) else "")

    seen = []
    r = run_board(
        startup, IDEA, settings=settings, execute=lambda a, t: seen.append(t.description) or exe(a, t), research=True
    )
    o = r.phase("research").outputs[0]
    assert o.text.startswith("Research unavailable") and "403 Client Error" in o.text and "$8" not in o.text
    assert o.sources == [] and o.tool_calls[0].error
    assert not any("$8 per user" in d for d in seen[5:])  # the critics never see the invented fact


def test_citations_no_tool_returned_are_flagged(startup, settings):
    def exe(agent, task):
        if task.description.startswith("Research this idea"):
            search_result(agent, output='{"organic": [{"link": "https://real.example/a"}]}')
            return "- real (https://real.example/a/)\n- made up ([src](https://invented.example/x))"
        return "critique" + (verdict_block() if wants_verdict(task) else "")

    r = run_board(startup, IDEA, settings=settings, execute=exe, research=True, phases=["hostile"])
    o = r.phase("research").outputs[0]
    assert o.sources == ["https://real.example/a"]
    assert "**Unverified citations**" in o.text and "- https://invented.example/x" in o.text
    assert "- https://real.example/a" not in o.text.split("**Unverified citations**")[1]


def test_tool_budget_and_breaker(monkeypatch):
    from idea_refiner import tools

    calls = []

    class Flaky:
        name, description, args_schema = "Search", "fake", None

        def run(self, **kw):
            calls.append(kw)
            raise RuntimeError("403 Forbidden")

    class Fine(Flaky):
        name = "Read"

        def run(self, **kw):
            calls.append(kw)
            return "page"

    monkeypatch.setitem(tools.REGISTRY, "web_search", tools.ToolDef("web_search", "", (), lambda: Flaky()))
    monkeypatch.setitem(tools.REGISTRY, "scrape", tools.ToolDef("scrape", "", (), lambda: Fine()))
    monkeypatch.setattr(tools, "public_url", lambda u: True)
    search, read = tools.build(["web_search", "scrape"], budget=4)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            search._run(search_query="x")
    assert search._run(search_query="x").startswith("Search is unavailable (RuntimeError: 403 Forbidden)")
    assert len(calls) == 2  # the broken tool is not called again
    assert read._run(website_url="https://a.example") == "page"
    assert read._run(website_url="https://b.example") == "page"
    assert read._run(website_url="https://c.example").startswith("Tool budget used up (4 calls)")
    assert len(calls) == 4

from __future__ import annotations

import pytest

from idea_refiner import report
from idea_refiner.engine import RunError, run_board
from idea_refiner.llm import LlmSpec

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


def test_full_run_shares_llm_objects_across_agents(startup, settings, execute):
    from idea_refiner.llm import _LLM_CACHE, clear_cache

    clear_cache()
    run_board(startup, IDEA, settings=settings, execute=execute)
    assert len(_LLM_CACHE) == 1  # 11 agents, one resolved spec, one client


def test_full_run_collects_every_agent(startup, settings, execute):
    events = []
    r = run_board(startup, IDEA, settings=settings, emit=events.append, execute=execute)
    assert [p.phase for p in r.phases] == ["hostile", "coaching", "synthesis"]
    assert len(r.phase("hostile").outputs) == 5 and len(r.phase("coaching").outputs) == 5
    assert r.phase("hostile").outputs[4].role == "CISO / Grey-Hat Hacker"
    assert r.phase("coaching").outputs[0].role == "Venture Capitalist (Coach)"
    assert r.pitch.startswith("Startup Founder says")
    assert r.model == "ollama/fake-model @ http://localhost:11434"
    types = [e.type for e in events]
    assert types[0] == "run_start" and types[-1] == "run_done"
    assert types.count("agent_start") == 11 and types.count("agent_done") == 11
    assert types.count("phase_start") == 3 == types.count("phase_done")


def test_coaching_receives_hostile_feedback_and_synthesis_receives_both(startup, settings):
    seen = []

    def spy(agents, tasks):
        seen.append(tasks[0].description)
        return [f"{a.role}-out" for a in agents]

    run_board(startup, IDEA, settings=settings, execute=spy)
    hostile, coaching, synth = seen
    assert IDEA in hostile and "Criticisms" not in hostile
    assert "### Hardened Venture Capitalist\nHardened Venture Capitalist-out" in coaching
    assert "Hardened Venture Capitalist-out" in synth and "Venture Capitalist (Coach)-out" in synth


def test_phase_subset(startup, settings, execute):
    r = run_board(startup, IDEA, settings=settings, execute=execute, phases=["hostile"])
    assert [p.phase for p in r.phases] == ["hostile"] and r.pitch is None
    r = run_board(startup, IDEA, settings=settings, execute=execute, phases=["synthesis"])
    assert r.pitch and len(r.phases) == 1
    with pytest.raises(RunError):
        run_board(startup, IDEA, settings=settings, execute=execute, phases=[])


def test_request_llm_overrides_settings_but_not_agent(startup, settings, execute):
    startup.agents[0].llm = LlmSpec(provider="mistral", model="mistral-small-latest")
    r = run_board(
        startup, IDEA, settings=settings, execute=execute, request_llm=LlmSpec(provider="openai", model="gpt-4o")
    )
    assert r.model == "openai/gpt-4o"
    outs = r.phase("hostile").outputs
    assert outs[0].model == "mistral/mistral-small-latest" and outs[1].model == "openai/gpt-4o"


def test_output_count_mismatch_is_an_error_event(startup, settings):
    events = []
    with pytest.raises(RunError):
        run_board(startup, IDEA, settings=settings, emit=events.append, execute=lambda a, t: ["only one"])
    assert events[-1].type == "error" and "expected 5" in events[-1].text


def test_report_roundtrip(startup, settings, execute):
    r = run_board(startup, IDEA, settings=settings, execute=execute, title="Legal diff")
    d = report.save(r, settings.runs_dir)
    assert (d / "result.json").exists() and (d / "report.md").exists()
    md = report.to_markdown(report.load(settings.runs_dir, r.id))
    assert md.startswith("# Legal diff") and "### Scaling CTO" in md and "## Refined pitch" in md
    assert [x.id for x in report.list_runs(settings.runs_dir)] == [r.id]

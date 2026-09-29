from __future__ import annotations

import re

from conftest import verdict_block, wants_verdict

from idea_refiner import report
from idea_refiner.engine import run_board
from idea_refiner.models import RefineSpec

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


def scripted(scores: list[int]):
    """Every critic gives revision n the score scores[n-1]; the n-th synthesis returns 'pitch n'."""
    calls: list[tuple[str, str]] = []

    def exe(agent, task):
        m = re.search(r"This is revision (\d+)", task.description)
        n = int(m.group(1)) if m else 1
        calls.append((agent.role, task.description))
        if wants_verdict(task):
            s = scores[n - 1]
            return f"{agent.role} on r{n}" + verdict_block("proceed" if s >= 7 else "pivot", s, [f"gap in r{n}"])
        if agent.role == "Startup Founder":  # the synthesis task names no revision: count them instead
            return f"pitch {sum(role == agent.role for role, _ in calls)}"
        return f"{agent.role} advice"

    return exe, calls


def _run(startup, settings, scores, **refine):
    exe, calls = scripted(scores)
    events = []
    r = run_board(startup, IDEA, settings=settings, execute=exe, emit=events.append, refine=RefineSpec(**refine))
    decisions = [(e.iteration, e.data["action"]) for e in events if e.type == "decision" and e.phase is None]
    return r, calls, decisions


def test_revision_that_clears_the_bar_ends_the_loop_without_more_synthesis(startup, settings):
    r, calls, decisions = _run(startup, settings, [4, 8], max_iterations=3)
    assert [(i.n, i.pitch) for i in r.iterations] == [(1, "pitch 1"), (2, None)]
    assert r.iterations[1].idea == "pitch 1" and r.pitch == "pitch 1"
    assert "revision 2 cleared the board (8.0/10" in r.iterations[1].outcome
    assert r.verdict.mean_score == 8.0 and r.verdict.decision == "proceed"
    assert decisions == [(1, "iterate"), (2, "stop")]
    assert [p.phase for p in r.phases if p.iteration == 2] == ["hostile"]  # no coaching/synthesis spent on it
    rev2 = next(d for role, d in calls if role == "Hardened Venture Capitalist" and "revision 2" in d)
    assert "gap in r1" in rev2 and IDEA in rev2 and "## Revision 2, the one you are judging" in rev2
    assert rev2.index("the one you are judging") < rev2.index("pitch 1")


def test_no_improvement_stops_the_loop(startup, settings):
    r, _, _ = _run(startup, settings, [5, 5, 9], max_iterations=3)
    assert len(r.iterations) == 2 and "did not improve the board's score (5.0 -> 5.0)" in r.iterations[1].outcome
    assert r.pitch == "pitch 1"


def test_iteration_limit_keeps_the_last_unjudged_pitch(startup, settings):
    r, _, decisions = _run(startup, settings, [3, 4, 5], max_iterations=3)
    assert [i.pitch for i in r.iterations] == ["pitch 1", "pitch 2", "pitch 3"] and r.pitch == "pitch 3"
    assert "iteration limit reached (3)" in r.iterations[2].outcome
    assert decisions == [(1, "iterate"), (2, "iterate"), (3, "stop")]
    md = report.to_markdown(r)
    assert "## Refine loop" in md and "# Revision 3" in md and "| 2 | pivot (4.0/10) | sent back to the board |" in md


def test_first_pass_that_clears_does_not_loop(startup, settings):
    r, _, _ = _run(startup, settings, [8], max_iterations=3)
    assert len(r.iterations) == 1 and r.iterations[0].outcome == "the idea cleared the board on the first pass"
    assert r.pitch == "pitch 1"


def test_target_and_kills_are_configurable(startup, settings):
    r, _, _ = _run(startup, settings, [4, 5], max_iterations=2, target_score=5)
    assert "cleared" in r.iterations[1].outcome


def test_single_pass_by_default(startup, settings):
    exe, _ = scripted([4, 5])
    r = run_board(startup, IDEA, settings=settings, execute=exe)
    assert len(r.iterations) == 1 and r.iterations[0].outcome is None and r.pitch == "pitch 1"
    assert all(p.iteration == 1 for p in r.phases)
    startup.refine = RefineSpec(max_iterations=2)  # the board's default applies when the request has none
    r = run_board(startup, IDEA, settings=settings, execute=exe)
    assert len(r.iterations) == 2


def test_next_revision_is_the_self_contained_brief(startup, settings):
    brief = "A feature-flag service for regulated teams, 40 paying customers, SOC2 from day one."
    seen = []

    def exe(agent, task):
        seen.append((agent.role, task.description))
        m = re.search(r"This is revision (\d+)", task.description)
        if wants_verdict(task):
            return "ok" + verdict_block("pivot", 4 if not m else 8)
        if agent.role == "Startup Founder":
            return f"Short pitch.\n\n## Revised idea\n{brief}"
        return "advice"

    r = run_board(startup, IDEA, settings=settings, execute=exe, refine=RefineSpec(max_iterations=2))
    assert r.iterations[1].idea == brief and r.pitch.startswith("Short pitch.")
    synthesis_task = next(d for role, d in seen if role == "Startup Founder")
    assert "## Revised idea" in synthesis_task  # asked for only because the loop is on
    r = run_board(startup, IDEA, settings=settings, execute=exe)
    assert "## Revised idea" not in [d for role, d in seen if role == "Startup Founder"][-1]


def test_revised_idea_falls_back_to_the_whole_pitch():
    from idea_refiner.engine import revised_idea

    assert revised_idea("Pitch.\n### Revised Idea\n" + "x" * 50) == "x" * 50
    assert revised_idea("Pitch without a brief.") == "Pitch without a brief."
    assert revised_idea("Pitch.\n## Revised idea\ntoo short") == "Pitch.\n## Revised idea\ntoo short"

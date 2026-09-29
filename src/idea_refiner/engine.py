"""Run a board against an idea: hostile round, deliberation, coaching round, synthesis.

The deliberation is where the board acts on its own: members rebut each other, and after each round a
chair decides whether another round is worth it and whom to press, so the number of rounds depends on
how the debate goes.

Orchestration on top of CrewAI. Every agent reply is one ``Job``; the jobs of a step are independent
and run in parallel (``Settings.concurrency``). The only side effects are LLM calls, which go through
``execute`` so tests can swap in a fake. Progress is reported through an ``emit`` callback so the CLI
(rich) and the API (SSE) share the same engine. ``emit`` may be called from worker threads.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import PROVIDERS, Settings
from .config import settings as default_settings
from .llm import LlmSpec, build_llm, describe, resolve
from .models import (
    ALL_PHASES,
    AgentOutput,
    AgentSpec,
    BoardSpec,
    ChairNote,
    Event,
    Mode,
    Phase,
    PhaseResult,
    RunResult,
)
from .verdicts import extract_json, parse_verdict, tally

if TYPE_CHECKING:
    from crewai import Agent, Task

Emit = Callable[[Event], None]
Execute = Callable[["Agent", "Task"], str]  # one agent working one task -> its raw reply


class RunError(Exception):
    pass


def _noop(_: Event) -> None:
    pass


def warm_up(settings: Settings = default_settings, spec: LlmSpec | None = None) -> None:
    """Pay the one-off costs up front: importing crewai (seconds) and building the default LLM client
    (loads the CA store). Errors are ignored; a misconfigured default provider fails loudly at run time."""
    import contextlib

    import crewai  # noqa: F401

    with contextlib.suppress(Exception):
        build_llm(spec or LlmSpec(), settings)


def effective_spec(*layers: LlmSpec | None) -> LlmSpec:
    """Merge specs, first argument has the highest priority."""
    out = LlmSpec()
    for layer in reversed(layers):
        if layer:
            out = layer.merged(out)
    return out


def crew_execute(agent: Agent, task: Task) -> str:
    """Default executor: a one-task CrewAI crew."""
    from crewai import Crew, Process

    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff().raw


@dataclass
class Job:
    """One agent answering one task."""

    seat: str  # agent id, used in events and results
    role: str
    goal: str
    backstory: str
    description: str
    expected: str
    llm: LlmSpec | None = None  # the seat's own spec; request and board layers are added at run time
    max_iter: int | None = None
    verdict: bool = False  # reply must end in a JSON verdict, parsed into AgentOutput.verdict


@dataclass
class Ctx:
    """Everything a run needs, passed down instead of eight positional arguments."""

    board: BoardSpec
    request_llm: LlmSpec | None
    settings: Settings
    emit: Emit
    execute: Execute
    run_id: str
    concurrency: int = 1

    def spec(self, seat_llm: LlmSpec | None) -> LlmSpec:
        return effective_spec(seat_llm, self.request_llm, self.board.llm)

    def event(self, type_: str, **kw) -> None:
        self.emit(Event(type=type_, run_id=self.run_id, **kw))


def default_concurrency(settings: Settings, spec: LlmSpec) -> int:
    """``REFINER_CONCURRENCY`` if set; otherwise 1 for local servers (one model in RAM) and 4 for clouds."""
    if settings.concurrency:
        return settings.concurrency
    try:
        return 1 if PROVIDERS[resolve(spec, settings).provider].local else 4
    except ValueError:
        return 1


def run_jobs(ctx: Ctx, phase: Phase, jobs: list[Job], round_: int | None = None) -> list[AgentOutput]:
    """Run independent jobs, in parallel up to ``ctx.concurrency``; outputs keep the jobs' order."""
    from crewai import Agent, Task

    t = ctx.board.prompts

    def one(j: Job) -> AgentOutput:
        ctx.event("agent_start", phase=phase, round=round_, agent_id=j.seat, role=j.role)
        t0 = time.perf_counter()
        spec = ctx.spec(j.llm)
        extra = {"max_iter": j.max_iter} if j.max_iter else {}
        agent = Agent(
            role=j.role,
            goal=j.goal,
            backstory=j.backstory,
            llm=build_llm(spec, ctx.settings),
            verbose=False,
            allow_delegation=False,
            **extra,
        )
        description, expected = j.description, j.expected
        if j.verdict:
            description += "\n\n" + t.verdict_format
            expected += " It ends with the fenced JSON verdict block."
        raw = ctx.execute(agent, Task(description=description, expected_output=expected, agent=agent))
        text, verdict = parse_verdict(raw) if j.verdict else (raw.strip(), None)
        out = AgentOutput(
            agent_id=j.seat,
            role=j.role,
            text=text,
            verdict=verdict,
            model=describe(spec, ctx.settings),
            seconds=round(time.perf_counter() - t0, 2),
        )
        data = {"verdict": verdict.model_dump()} if verdict else None
        ctx.event("agent_done", phase=phase, round=round_, agent_id=j.seat, role=j.role, text=text, data=data)
        return out

    if ctx.concurrency <= 1 or len(jobs) == 1:
        return [one(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=min(ctx.concurrency, len(jobs)), thread_name_prefix="agent") as pool:
        return list(pool.map(one, jobs))


def seat_job(ctx: Ctx, a: AgentSpec, mode: Mode, description: str, verdict: bool = False) -> Job:
    p, t = a.persona(mode), ctx.board.prompts
    role, focus = p.role or a.role, p.focus or a.focus
    fmt = dict(role=role, focus=focus, tone=t.hostile_tone if mode == "hostile" else t.coaching_tone)
    return Job(
        seat=a.id,
        role=role,
        goal=(p.goal or t.goal).format(**fmt),
        backstory=(p.backstory or t.backstory).format(**fmt),
        description=description,
        expected=t.expected_output.format(sentences=t.sentences),
        llm=a.llm,
        max_iter=a.max_iter,
        verdict=verdict,
    )


def _phase(ctx: Ctx, phase: Phase, jobs: list[Job], verdicts: bool, round_: int | None = None) -> PhaseResult:
    ctx.event("phase_start", phase=phase, round=round_)
    t0 = time.perf_counter()
    outputs = run_jobs(ctx, phase, jobs, round_)
    result = PhaseResult(
        phase=phase,
        outputs=outputs,
        round=round_,
        tally=tally(outputs) if verdicts else None,
        seconds=round(time.perf_counter() - t0, 2),
    )
    data = {"tally": result.tally.model_dump()} if result.tally else None
    ctx.event("phase_done", phase=phase, round=round_, text=result.as_markdown(), data=data)
    return result


def run_phase(ctx: Ctx, mode: Mode, idea: str, feedback: str | None = None) -> PhaseResult:
    t = ctx.board.prompts
    template = t.hostile_task if mode == "hostile" else t.coaching_task
    description = template.format(idea=idea, feedback=feedback or "", sentences=t.sentences)
    verdicts = mode == "hostile" and ctx.board.verdicts
    return _phase(ctx, mode, [seat_job(ctx, a, mode, description, verdicts) for a in ctx.board.agents], verdicts)


def _positions(p: PhaseResult) -> dict[str, str | None]:
    return {o.agent_id: o.verdict.decision if o.verdict else None for o in p.outputs}


def _movement(before: PhaseResult, after: PhaseResult) -> str:
    was, now = _positions(before), _positions(after)
    lines = [f"- {k}: {was.get(k) or '?'} -> {v or '?'}" for k, v in now.items() if was.get(k) != v]
    return "\n".join(lines) or "Nobody changed their verdict."


def run_chair(ctx: Ctx, idea: str, n: int, before: PhaseResult, this: PhaseResult) -> ChairNote:
    """Ask the chair whether another round is worth it. An unusable reply closes the debate: when in
    doubt, stop spending tokens."""
    t, c = ctx.board.prompts, ctx.board.deliberation.chair
    ids = [a.id for a in ctx.board.agents]
    job = Job(
        seat="chair",
        role=c.role,
        goal=c.goal,
        backstory=c.backstory,
        description=t.chair_task.format(
            idea=idea,
            round=n,
            rounds=ctx.board.deliberation.rounds,
            positions="\n\n".join(f"{o.as_markdown()}\n(member id: {o.agent_id})" for o in this.outputs),
            standing=this.tally.as_text() if this.tally else "(no verdicts)",
            movement=_movement(before, this),
            ids=", ".join(ids),
        )
        + "\n\n"
        + t.chair_format,
        expected="A short summary of where the board stands, ending with the fenced JSON decision block.",
        llm=c.llm,
    )
    raw = run_jobs(ctx, "deliberation", [job], n)[0].text
    summary, obj = extract_json(raw)
    action = str((obj or {}).get("action", "")).strip().lower()
    if action not in ("continue", "close"):
        return ChairNote(action="close", reason="the chair gave no usable decision", summary=summary)
    questions = (obj or {}).get("questions") or {}
    questions = (
        {k: str(v) for k, v in questions.items() if k in ids and str(v).strip()} if isinstance(questions, dict) else {}
    )
    return ChairNote(action=action, reason=str(obj.get("reason", "")), summary=summary, questions=questions)


def run_deliberation(ctx: Ctx, idea: str, opening: PhaseResult) -> list[PhaseResult]:
    """Critics read each other's positions and rebut or concede, round after round, until the board is
    unanimous, the chair closes the debate (or, without a chair, nobody moves), or the round limit hits."""
    spec, t = ctx.board.deliberation, ctx.board.prompts
    rounds: list[PhaseResult] = []
    if spec.stop_on_consensus and opening.tally and opening.tally.unanimous:
        reason = "the board was unanimous from the opening round"
        ctx.event("decision", phase="deliberation", round=0, text=reason, data={"action": "skip", "by": "rule"})
        return rounds
    prev, questions = opening, {}
    for n in range(1, spec.rounds + 1):
        jobs = []
        for a in ctx.board.agents:
            own = next(o for o in prev.outputs if o.agent_id == a.id)
            q = questions.get(a.id)
            description = t.deliberation_task.format(
                idea=idea,
                round=n,
                own=own.as_markdown(),
                others="\n\n".join(o.as_markdown() for o in prev.outputs if o.agent_id != a.id),
                standing=prev.tally.as_text() if prev.tally else "(no verdicts)",
                question=f"## The chair asks you\n{q}\n\n" if q else "",
                sentences=t.sentences,
            )
            jobs.append(seat_job(ctx, a, "hostile", description, verdict=ctx.board.verdicts))
        this = _phase(ctx, "deliberation", jobs, verdicts=ctx.board.verdicts, round_=n)
        rounds.append(this)
        if spec.stop_on_consensus and this.tally and this.tally.unanimous:
            this.closed, by = "the board is unanimous", "rule"
        elif n >= spec.rounds:
            this.closed, by = "round limit reached", "rule"
        elif spec.chair:
            this.chair = run_chair(ctx, idea, n, prev, this)
            questions, by = this.chair.questions, "chair"
            if this.chair.action == "close":
                this.closed = this.chair.reason or "closed by the chair"
        elif _positions(this) == _positions(prev):
            this.closed, by = "positions are stable", "rule"
        else:
            questions, by = {}, "rule"
        action = "close" if this.closed else "continue"
        reason = this.closed or (this.chair.reason if this.chair else "positions are still moving")
        ctx.event("decision", phase="deliberation", round=n, text=reason, data={"action": action, "by": by})
        if this.closed:
            break
        prev = this
    return rounds


def deliberation_digest(rounds: list[PhaseResult]) -> str:
    """What later phases see of the debate: the final positions and the chair's reading of each round."""
    if not rounds:
        return "(none)"
    notes = [f"- Round {r.round}: {r.chair.summary}" for r in rounds if r.chair and r.chair.summary]
    chair = "\n\n**Chair's notes:**\n" + "\n".join(notes) if notes else ""
    return f"Final positions after {len(rounds)} round(s):\n\n{rounds[-1].as_markdown()}{chair}"


def run_synthesis(
    ctx: Ctx,
    idea: str,
    hostile: PhaseResult | None,
    coaching: PhaseResult | None,
    standing: str,
    deliberation: str = "(none)",
) -> PhaseResult:
    t, s = ctx.board.prompts, ctx.board.synthesizer
    job = Job(
        seat="synthesizer",
        role=s.role,
        goal=s.goal,
        backstory=s.backstory,
        description=t.synthesis_task.format(
            idea=idea,
            hostile_feedback=hostile.as_markdown() if hostile else "(none)",
            coaching_advice=coaching.as_markdown() if coaching else "(none)",
            deliberation=deliberation,
            verdict=standing,
            sentences=t.synthesis_sentences,
        ),
        expected=t.synthesis_expected,
        llm=s.llm,
    )
    return _phase(ctx, "synthesis", [job], verdicts=False)


def run_board(
    board: BoardSpec,
    idea: str,
    *,
    request_llm: LlmSpec | None = None,
    phases: list[Phase] | None = None,
    title: str | None = None,
    settings: Settings = default_settings,
    emit: Emit = _noop,
    execute: Execute = crew_execute,
    run_id: str | None = None,
) -> RunResult:
    """Run the requested phases (default: the board's) and return a full RunResult."""
    wanted = [p for p in ALL_PHASES if p in (board.phases if phases is None else phases)]
    if not wanted:
        raise RunError("no phases selected")
    if "deliberation" in wanted and "hostile" not in wanted:
        raise RunError("deliberation needs the hostile phase: members debate their opening critiques")
    run_spec = effective_spec(request_llm, board.llm)
    result = RunResult(board=board.name, idea=idea.strip(), title=title, model=describe(run_spec, settings))
    if run_id:
        result.id = run_id
    ctx = Ctx(board, request_llm, settings, emit, execute, result.id, default_concurrency(settings, run_spec))
    ctx.event("run_start", text=result.model)
    warm_up(settings, run_spec)  # keep import/SSL setup out of the timings
    t0 = time.perf_counter()
    try:
        hostile = coaching = None
        rounds: list[PhaseResult] = []
        if "hostile" in wanted:
            hostile = run_phase(ctx, "hostile", result.idea)
            result.phases.append(hostile)
            result.verdict = hostile.tally
        if "deliberation" in wanted and hostile:
            rounds = run_deliberation(ctx, result.idea, hostile)
            result.phases.extend(rounds)
            if rounds:
                result.verdict = rounds[-1].tally or result.verdict
        digest = deliberation_digest(rounds)
        if "coaching" in wanted:
            feedback = hostile.as_markdown() if hostile else None
            if feedback and rounds:
                feedback += f"\n\n## After deliberation\n{digest}"
            coaching = run_phase(ctx, "coaching", result.idea, feedback)
            result.phases.append(coaching)
        if "synthesis" in wanted:
            standing = result.verdict.as_text() if result.verdict else "(none)"
            synthesis = run_synthesis(ctx, result.idea, hostile, coaching, standing, digest)
            result.phases.append(synthesis)
            result.pitch = synthesis.outputs[0].text
    except Exception as e:  # noqa: BLE001 - surface every failure as an event, then re-raise
        ctx.event("error", text=f"{type(e).__name__}: {e}")
        raise
    result.seconds = round(time.perf_counter() - t0, 2)
    ctx.event("run_done", text=result.pitch)
    return result

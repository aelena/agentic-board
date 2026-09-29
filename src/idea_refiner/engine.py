"""Run a board against an idea: hostile round, coaching round, synthesis.

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
from .models import ALL_PHASES, AgentOutput, AgentSpec, BoardSpec, Event, Mode, Phase, PhaseResult, RunResult
from .verdicts import parse_verdict, tally

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


def run_jobs(ctx: Ctx, phase: Phase, jobs: list[Job]) -> list[AgentOutput]:
    """Run independent jobs, in parallel up to ``ctx.concurrency``; outputs keep the jobs' order."""
    from crewai import Agent, Task

    t = ctx.board.prompts

    def one(j: Job) -> AgentOutput:
        ctx.event("agent_start", phase=phase, agent_id=j.seat, role=j.role)
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
        ctx.event("agent_done", phase=phase, agent_id=j.seat, role=j.role, text=text, data=data)
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


def _phase(ctx: Ctx, phase: Phase, jobs: list[Job], verdicts: bool) -> PhaseResult:
    ctx.event("phase_start", phase=phase)
    t0 = time.perf_counter()
    outputs = run_jobs(ctx, phase, jobs)
    result = PhaseResult(
        phase=phase,
        outputs=outputs,
        tally=tally(outputs) if verdicts else None,
        seconds=round(time.perf_counter() - t0, 2),
    )
    data = {"tally": result.tally.model_dump()} if result.tally else None
    ctx.event("phase_done", phase=phase, text=result.as_markdown(), data=data)
    return result


def run_phase(ctx: Ctx, mode: Mode, idea: str, feedback: str | None = None) -> PhaseResult:
    t = ctx.board.prompts
    template = t.hostile_task if mode == "hostile" else t.coaching_task
    description = template.format(idea=idea, feedback=feedback or "", sentences=t.sentences)
    verdicts = mode == "hostile" and ctx.board.verdicts
    return _phase(ctx, mode, [seat_job(ctx, a, mode, description, verdicts) for a in ctx.board.agents], verdicts)


def run_synthesis(
    ctx: Ctx, idea: str, hostile: PhaseResult | None, coaching: PhaseResult | None, standing: str
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
        if "hostile" in wanted:
            hostile = run_phase(ctx, "hostile", result.idea)
            result.phases.append(hostile)
            result.verdict = hostile.tally
        if "coaching" in wanted:
            coaching = run_phase(ctx, "coaching", result.idea, hostile.as_markdown() if hostile else None)
            result.phases.append(coaching)
        if "synthesis" in wanted:
            standing = result.verdict.as_text() if result.verdict else "(none)"
            synthesis = run_synthesis(ctx, result.idea, hostile, coaching, standing)
            result.phases.append(synthesis)
            result.pitch = synthesis.outputs[0].text
    except Exception as e:  # noqa: BLE001 - surface every failure as an event, then re-raise
        ctx.event("error", text=f"{type(e).__name__}: {e}")
        raise
    result.seconds = round(time.perf_counter() - t0, 2)
    ctx.event("run_done", text=result.pitch)
    return result

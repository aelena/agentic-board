"""Run a board against an idea: hostile round, coaching round, synthesis.

Pure orchestration on top of CrewAI. The only side effects are LLM calls, which go through
``execute`` so tests can swap in a fake. Progress is reported through an ``emit`` callback so
the CLI (rich) and the API (SSE) share the same engine.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from crewai import Agent, Crew, Process, Task

from .config import Settings
from .config import settings as default_settings
from .llm import LlmSpec, build_llm, describe
from .models import ALL_PHASES, AgentOutput, AgentSpec, BoardSpec, Event, Mode, Phase, PhaseResult, RunResult

Emit = Callable[[Event], None]
Execute = Callable[[list[Agent], list[Task]], list[str]]  # returns one raw output per task, in order


class RunError(Exception):
    pass


def _noop(_: Event) -> None:
    pass


def effective_spec(*layers: LlmSpec | None) -> LlmSpec:
    """Merge specs, first argument has the highest priority."""
    out = LlmSpec()
    for layer in reversed(layers):
        if layer:
            out = layer.merged(out)
    return out


def make_agent(spec: AgentSpec, mode: Mode, board: BoardSpec, request_llm: LlmSpec | None, settings: Settings) -> Agent:
    p, t = spec.persona(mode), board.prompts
    role, focus = p.role or spec.role, p.focus or spec.focus
    tone = t.hostile_tone if mode == "hostile" else t.coaching_tone
    fmt = dict(role=role, focus=focus, tone=tone)
    return Agent(
        role=role,
        goal=(p.goal or t.goal).format(**fmt),
        backstory=(p.backstory or t.backstory).format(**fmt),
        llm=build_llm(effective_spec(spec.llm, request_llm, board.llm), settings),
        verbose=False,
        allow_delegation=False,
        max_iter=spec.max_iter,
    )


def make_synthesizer(board: BoardSpec, request_llm: LlmSpec | None, settings: Settings) -> Agent:
    s = board.synthesizer
    return Agent(
        role=s.role,
        goal=s.goal,
        backstory=s.backstory,
        llm=build_llm(effective_spec(s.llm, request_llm, board.llm), settings),
        verbose=False,
        allow_delegation=False,
    )


def crew_execute(agents: list[Agent], tasks: list[Task]) -> list[str]:
    """Default executor: one sequential CrewAI crew, every task's raw output returned."""
    out = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
    return [t.raw for t in out.tasks_output]


def run_phase(
    board: BoardSpec,
    mode: Mode,
    idea: str,
    feedback: str | None,
    run_id: str,
    request_llm: LlmSpec | None,
    settings: Settings,
    emit: Emit,
    execute: Execute,
) -> PhaseResult:
    t = board.prompts
    template = t.hostile_task if mode == "hostile" else t.coaching_task
    description = template.format(idea=idea, feedback=feedback or "", sentences=t.sentences)
    expected = t.expected_output.format(sentences=t.sentences)
    agents = [make_agent(a, mode, board, request_llm, settings) for a in board.agents]
    roles = [a.role for a in agents]
    ids = [a.id for a in board.agents]

    def on_done(i: int):
        def cb(output):
            emit(
                Event(
                    type="agent_done",
                    run_id=run_id,
                    phase=mode,
                    agent_id=ids[i],
                    role=roles[i],
                    text=output.raw,
                )
            )
            if i + 1 < len(agents):
                emit(Event(type="agent_start", run_id=run_id, phase=mode, agent_id=ids[i + 1], role=roles[i + 1]))

        return cb

    tasks = [
        Task(description=description, expected_output=expected, agent=ag, callback=on_done(i))
        for i, ag in enumerate(agents)
    ]
    emit(Event(type="phase_start", run_id=run_id, phase=mode))
    emit(Event(type="agent_start", run_id=run_id, phase=mode, agent_id=ids[0], role=roles[0]))
    t0 = time.perf_counter()
    raws = execute(agents, tasks)
    if len(raws) != len(agents):
        raise RunError(f"{mode}: expected {len(agents)} outputs, got {len(raws)}")
    result = PhaseResult(
        phase=mode,
        outputs=[
            AgentOutput(
                agent_id=a.id,
                role=r,
                text=raw,
                model=describe(effective_spec(a.llm, request_llm, board.llm), settings),
            )
            for a, r, raw in zip(board.agents, roles, raws, strict=True)
        ],
        seconds=round(time.perf_counter() - t0, 2),
    )
    emit(Event(type="phase_done", run_id=run_id, phase=mode, text=result.as_markdown()))
    return result


def run_synthesis(
    board: BoardSpec,
    idea: str,
    hostile: PhaseResult | None,
    coaching: PhaseResult | None,
    run_id: str,
    request_llm: LlmSpec | None,
    settings: Settings,
    emit: Emit,
    execute: Execute,
) -> tuple[str, float]:
    t = board.prompts
    agent = make_synthesizer(board, request_llm, settings)
    task = Task(
        description=t.synthesis_task.format(
            idea=idea,
            hostile_feedback=hostile.as_markdown() if hostile else "(none)",
            coaching_advice=coaching.as_markdown() if coaching else "(none)",
            sentences=t.synthesis_sentences,
        ),
        expected_output=t.synthesis_expected,
        agent=agent,
    )
    emit(Event(type="phase_start", run_id=run_id, phase="synthesis"))
    emit(Event(type="agent_start", run_id=run_id, phase="synthesis", agent_id="synthesizer", role=agent.role))
    t0 = time.perf_counter()
    pitch = execute([agent], [task])[0]
    secs = round(time.perf_counter() - t0, 2)
    emit(
        Event(
            type="agent_done",
            run_id=run_id,
            phase="synthesis",
            agent_id="synthesizer",
            role=agent.role,
            text=pitch,
        )
    )
    emit(Event(type="phase_done", run_id=run_id, phase="synthesis", text=pitch))
    return pitch, secs


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
    result = RunResult(
        board=board.name,
        idea=idea.strip(),
        title=title,
        model=describe(effective_spec(request_llm, board.llm), settings),
    )
    if run_id:
        result.id = run_id
    emit(Event(type="run_start", run_id=result.id, text=result.model))
    t0 = time.perf_counter()
    try:
        hostile = coaching = None
        if "hostile" in wanted:
            hostile = run_phase(board, "hostile", result.idea, None, result.id, request_llm, settings, emit, execute)
            result.phases.append(hostile)
        if "coaching" in wanted:
            fb = hostile.as_markdown() if hostile else None
            coaching = run_phase(board, "coaching", result.idea, fb, result.id, request_llm, settings, emit, execute)
            result.phases.append(coaching)
        if "synthesis" in wanted:
            pitch, secs = run_synthesis(
                board, result.idea, hostile, coaching, result.id, request_llm, settings, emit, execute
            )
            result.pitch = pitch
            out = AgentOutput(agent_id="synthesizer", role=board.synthesizer.role, text=pitch)
            result.phases.append(PhaseResult(phase="synthesis", outputs=[out], seconds=secs))
    except Exception as e:  # noqa: BLE001 - surface every failure as an event, then re-raise
        emit(Event(type="error", run_id=result.id, text=f"{type(e).__name__}: {e}"))
        raise
    result.seconds = round(time.perf_counter() - t0, 2)
    emit(Event(type="run_done", run_id=result.id, text=result.pitch))
    return result

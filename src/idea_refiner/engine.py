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

import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import tools
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
    IterationSummary,
    McpRef,
    Mode,
    Phase,
    PhaseResult,
    ProjectContext,
    RefineSpec,
    RunResult,
    Tally,
    ToolCall,
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
    tools: list[str] = field(default_factory=list)  # tool names, see idea_refiner.tools
    tool_budget: int = 4  # hard cap on tool calls for this job
    mcp: list[McpRef] = field(default_factory=list)  # MCP servers (and allowed tools) for this job


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
    iteration: int = 1  # revision of the idea being worked on; stamped on every event and phase
    context: ProjectContext | None = None  # project guidelines and per-seat memory
    briefings: dict[str, str] = field(default_factory=dict)  # seat id -> research briefing
    _mcp: dict[str, list] = field(default_factory=dict)  # server name -> resolved tools, once per run
    _mcp_lock: threading.Lock = field(default_factory=threading.Lock)

    def mcp_tools(self, ref: McpRef) -> list:
        """Tools of one MCP server, resolved once per run (connecting lists its tools), filtered by
        ``ref.allow``. Raises when the server cannot be reached."""
        with self._mcp_lock:
            if ref.server not in self._mcp:
                self._mcp[ref.server] = tools.resolve_mcp(self.board.mcp_servers[ref.server])
        found = self._mcp[ref.server]
        return found if ref.allow is None else [t for t in found if tools.mcp_tool_name(t) in ref.allow]

    def preamble(self, seat: str) -> str:
        """Project brief plus the seat's own notes from earlier runs, in front of every task."""
        if not self.context:
            return ""
        t = self.board.prompts
        out = t.project_context.format(name=self.context.name, brief=self.context.brief) if self.context.brief else ""
        notes = self.context.memory.get(seat if seat in {a.id for a in self.board.agents} else "board")
        return out + (t.memory_context.format(notes=notes) if notes else "")

    def spec(self, seat_llm: LlmSpec | None) -> LlmSpec:
        return effective_spec(seat_llm, self.request_llm, self.board.llm)

    def event(self, type_: str, **kw) -> None:
        self.emit(Event(type=type_, run_id=self.run_id, iteration=self.iteration, **kw))


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
        native = []
        for ref in j.mcp:
            try:
                native += [(tools.mcp_display_name(ref.server, t), t) for t in ctx.mcp_tools(ref)]
            except Exception as e:  # noqa: BLE001 - a server that is down costs the seat a tool, not the run
                err = f"MCP server '{ref.server}' unavailable: {type(e).__name__}: {str(e)[:200]}"
                data = {"status": "error", "tool": f"mcp:{ref.server}", "args": "", "error": err}
                ctx.event("tool", phase=phase, round=round_, agent_id=j.seat, role=j.role, text=err, data=data)
        if j.tools or native:
            extra["tools"] = tools.build(j.tools, j.tool_budget, native)
        agent = Agent(
            role=j.role,
            goal=j.goal,
            backstory=j.backstory,
            llm=build_llm(spec, ctx.settings),
            verbose=False,
            allow_delegation=False,
            **extra,
        )
        description, expected = ctx.preamble(j.seat) + j.description, j.expected
        if "tools" in extra:
            names = j.tools + [f"mcp:{r.server}" for r in j.mcp]
            description += "\n\n" + t.tool_guidance.format(tools=", ".join(names))
        if j.verdict:
            description += "\n\n" + t.verdict_format
            expected += " It ends with the fenced JSON verdict block."
        calls: list[ToolCall] = []
        sources: set[str] = set()
        reporter = _tool_reporter(ctx, phase, round_, j, calls, sources)
        unroute = tools.route(str(agent.id), reporter) if "tools" in extra else None
        try:
            raw = ctx.execute(agent, Task(description=description, expected_output=expected, agent=agent))
        finally:
            if unroute:
                unroute()
        text, verdict = parse_verdict(raw) if j.verdict else (raw.strip(), None)
        out = AgentOutput(
            agent_id=j.seat,
            role=j.role,
            text=text,
            verdict=verdict,
            tool_calls=calls,
            sources=sorted(sources),
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


def _clip_args(args) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items()) if isinstance(args, dict) else str(args)
    return text if len(text) <= 160 else text[:157] + "..."


_NO_RESULT = ("Refused:", "Tool budget used up")


def _tool_reporter(ctx: Ctx, phase: Phase, round_: int | None, j: Job, calls: list[ToolCall], sources: set[str]):
    """Turn CrewAI tool events for one agent into ``tool`` events, ToolCall records, and the set of URLs
    the tools really returned (search result links, successfully scraped pages)."""

    def on(kind: str, e) -> None:
        if kind == "started":  # CrewAI runs handlers in a pool, so a start can arrive after its finish
            return
        call = ToolCall(tool=e.tool_name, args=_clip_args(e.tool_args))
        if kind == "finished":
            call.seconds = round((e.finished_at - e.started_at).total_seconds(), 2)
            out = str(e.output or "")
            call.empty = not out.strip() or out.startswith(_NO_RESULT) or " is unavailable (" in out[:200]
            if not call.empty:
                sources.update(tools.urls_in(out))
                if isinstance(e.tool_args, dict) and (url := e.tool_args.get("website_url")):
                    sources.add(str(url))
        else:
            call.error = str(e.error)[:300]
        calls.append(call)
        data = {"status": kind, **call.model_dump(exclude_none=True)}
        ctx.event("tool", phase=phase, round=round_, agent_id=j.seat, role=j.role, text=call.tool, data=data)

    return on


def seat_job(
    ctx: Ctx, a: AgentSpec, mode: Mode, description: str, verdict: bool = False, phase: Phase | None = None
) -> Job:
    """``phase`` (default: ``mode``) decides whether the seat gets its tools, see BoardSpec.tool_phases."""
    p, t = a.persona(mode), ctx.board.prompts
    in_tool_phase = (phase or mode) in ctx.board.tool_phases
    seat_tools, seat_mcp = (a.tools, a.mcp) if in_tool_phase else ([], [])
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
        max_iter=a.max_iter or (8 if seat_tools or seat_mcp else 3),
        tool_budget=a.tool_budget,
        verdict=verdict,
        tools=list(seat_tools),
        mcp=list(seat_mcp),
    )


def _phase(ctx: Ctx, phase: Phase, jobs: list[Job], verdicts: bool, round_: int | None = None) -> PhaseResult:
    ctx.event("phase_start", phase=phase, round=round_)
    t0 = time.perf_counter()
    outputs = run_jobs(ctx, phase, jobs, round_)
    result = PhaseResult(
        phase=phase,
        outputs=outputs,
        iteration=ctx.iteration,
        round=round_,
        tally=tally(outputs) if verdicts else None,
        seconds=round(time.perf_counter() - t0, 2),
    )
    data = {"tally": result.tally.model_dump()} if result.tally else None
    ctx.event("phase_done", phase=phase, round=round_, text=result.as_markdown(), data=data)
    return result


def run_phase(ctx: Ctx, mode: Mode, idea: str, feedback: str | None = None, preamble: str = "") -> PhaseResult:
    t = ctx.board.prompts
    template = t.hostile_task if mode == "hostile" else t.coaching_task
    description = preamble + template.format(idea=idea, feedback=feedback or "", sentences=t.sentences)
    verdicts = mode == "hostile" and ctx.board.verdicts

    def briefed(seat: str) -> str:
        b = ctx.briefings.get(seat)
        return (t.briefing_context.format(briefing=b) if b else "") + description

    jobs = [seat_job(ctx, a, mode, briefed(a.id), verdicts) for a in ctx.board.agents]
    return _phase(ctx, mode, jobs, verdicts)


def run_research(ctx: Ctx, idea: str) -> PhaseResult:
    """One researcher per seat, in parallel, each looking for evidence on that seat's concern. The
    briefings are kept on ``ctx`` and prepended to that seat's hostile and coaching tasks."""
    r, t = ctx.board.research, ctx.board.prompts
    jobs = [
        Job(
            seat=a.id,
            role=f"{r.role} for the {a.role}",
            goal=r.goal,
            backstory=r.backstory,
            description=t.research_task.format(idea=idea, role=a.role, focus=a.focus, points=t.research_points),
            expected=t.research_expected,
            llm=r.llm,
            max_iter=r.max_iter,
            tools=list(r.tools),
            tool_budget=r.tool_budget,
            mcp=list(r.mcp),
        )
        for a in ctx.board.agents
    ]
    result = _phase(ctx, "research", jobs, verdicts=False)
    for o in result.outputs:
        o.text = ground(o)
    ctx.briefings = {o.agent_id: o.text for o in result.outputs if o.text.strip()}
    return result


def _norm(url: str) -> str:
    return url.rstrip("/").lower()


def ground(o: AgentOutput) -> str:
    """Check a research briefing against what its tools returned. With no tool results at all the draft is
    discarded: models happily invent sourced-looking facts when every search failed. Otherwise citations
    no tool returned are listed as unverified."""
    if not any(not c.error and not c.empty for c in o.tool_calls):
        errors = [c.error for c in o.tool_calls if c.error]
        why = f" (last error: {errors[-1][:160]})" if errors else ""
        return (
            f"Research unavailable: no tool call returned results{why}. The researcher's draft was discarded "
            "because nothing in it could be verified. Rely on your own judgement and say where facts are missing."
        )
    known = {_norm(u) for u in o.sources}
    cited = sorted(tools.urls_in(o.text))
    unverified = [u for u in cited if not any(_norm(u) == k or _norm(u).startswith(k + "/") for k in known)]
    if not unverified:
        return o.text
    listed = "\n".join(f"- {u}" for u in unverified)
    return f"{o.text}\n\n**Unverified citations** (no tool returned these; treat the facts behind them as claims):\n{listed}"


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
            jobs.append(seat_job(ctx, a, "hostile", description, ctx.board.verdicts, phase="deliberation"))
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
    brief: bool = False,
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
        )
        + (f"\n\n{t.revision_brief}" if brief else ""),
        expected=t.synthesis_expected,
        llm=s.llm,
    )
    return _phase(ctx, "synthesis", [job], verdicts=False)


def _previous_view(prev: IterationSummary, phases: list[PhaseResult]) -> str:
    """What the board said about the last revision: its standing and every seat's open issues."""
    last = next((p for p in reversed(phases) if p.iteration == prev.n and p.tally), None)
    lines = [f"Board verdict: {prev.verdict.as_text()}" if prev.verdict else "Board verdict: none"]
    for o in last.outputs if last else []:
        if o.verdict:
            lines.append(f"- {o.role} ({o.verdict.decision}, {o.verdict.score}/10): {'; '.join(o.verdict.issues)}")
    return "\n".join(lines)


def cleared(refine: RefineSpec, verdict: Tally | None) -> bool:
    return bool(
        verdict
        and verdict.mean_score is not None
        and verdict.mean_score >= refine.target_score
        and verdict.votes.get("kill", 0) <= refine.max_kills
    )


def _judge(ctx: Ctx, wanted: list[Phase], idea: str, preamble: str, result: RunResult) -> tuple:
    """Hostile round plus deliberation on one revision. Returns (hostile, rounds, verdict)."""
    hostile, rounds, verdict = None, [], None
    if "hostile" in wanted:
        hostile = run_phase(ctx, "hostile", idea, preamble=preamble)
        result.phases.append(hostile)
        verdict = hostile.tally
    if "deliberation" in wanted and hostile:
        rounds = run_deliberation(ctx, idea, hostile)
        result.phases.extend(rounds)
        if rounds:
            verdict = rounds[-1].tally or verdict
    return hostile, rounds, verdict


_REVISED = re.compile(r"^#{1,4}\s*revised idea\s*$", re.I | re.M)


def revised_idea(pitch: str) -> str:
    """The self-contained brief the synthesizer wrote under '## Revised idea', else the whole pitch."""
    m = _REVISED.search(pitch)
    brief = pitch[m.end() :].strip() if m else ""
    return brief if len(brief) >= 40 else pitch


def _improve(
    ctx: Ctx, wanted: list[Phase], idea: str, hostile, rounds, verdict, result: RunResult, brief: bool = False
) -> str | None:
    """Coaching plus synthesis on one revision. Returns the new pitch (None without synthesis)."""
    coaching = None
    digest = deliberation_digest(rounds)
    if "coaching" in wanted:
        feedback = hostile.as_markdown() if hostile else None
        if feedback and rounds:
            feedback += f"\n\n## After deliberation\n{digest}"
        coaching = run_phase(ctx, "coaching", idea, feedback)
        result.phases.append(coaching)
    if "synthesis" not in wanted:
        return None
    standing = verdict.as_text() if verdict else "(none)"
    synthesis = run_synthesis(ctx, idea, hostile, coaching, standing, digest, brief)
    result.phases.append(synthesis)
    return synthesis.outputs[0].text


def run_board(
    board: BoardSpec,
    idea: str,
    *,
    request_llm: LlmSpec | None = None,
    phases: list[Phase] | None = None,
    title: str | None = None,
    refine: RefineSpec | None = None,
    research: bool = False,
    context: ProjectContext | None = None,
    settings: Settings = default_settings,
    emit: Emit = _noop,
    execute: Execute = crew_execute,
    run_id: str | None = None,
) -> RunResult:
    """Run the requested phases (default: the board's) and return a full RunResult.

    With ``refine.max_iterations > 1`` this is a loop: each iteration's pitch becomes the next revision
    and goes back in front of the board, which checks it against its own earlier issues. The loop stops
    as soon as a revision clears ``target_score``, a revision fails to improve on the previous score, or
    the iteration limit is reached. A judged revision the loop stops on gets no coaching or synthesis:
    the pitch that was just judged is the result.
    """
    chosen = set(board.phases if phases is None else phases) | ({"research"} if research else set())
    wanted = [p for p in ALL_PHASES if p in chosen]
    if not wanted:
        raise RunError("no phases selected")
    if "deliberation" in wanted and "hostile" not in wanted:
        raise RunError("deliberation needs the hostile phase: members debate their opening critiques")
    needed = {n for a in board.agents for n in a.tools if set(board.tool_phases) & set(wanted)}
    needed = sorted(needed | (set(board.research.tools) if "research" in wanted else set()))
    problems = tools.missing(needed)  # before any LLM call, not halfway through a run
    servers = {r.server for a in board.agents for r in a.mcp if set(board.tool_phases) & set(wanted)}
    servers |= {r.server for r in board.research.mcp} if "research" in wanted else set()
    for name in sorted(servers):
        spec = board.mcp_servers[name]
        problems += tools.mcp_problems(name, spec, bool(board.source), settings.allow_mcp_commands)
    if problems:
        raise RunError("; ".join(problems))
    refine = refine or board.refine
    loop = refine.max_iterations > 1 and board.verdicts and {"hostile", "synthesis"} <= set(wanted)
    max_n = refine.max_iterations if loop else 1
    run_spec = effective_spec(request_llm, board.llm)
    result = RunResult(
        board=board.name,
        project=context.name if context else None,
        idea=idea.strip(),
        title=title,
        model=describe(run_spec, settings),
    )
    if run_id:
        result.id = run_id
    ctx = Ctx(board, request_llm, settings, emit, execute, result.id, default_concurrency(settings, run_spec))
    ctx.context = context
    ctx.event("run_start", text=result.model)
    warm_up(settings, run_spec)  # keep import/SSL setup out of the timings
    t0 = time.perf_counter()

    def stop(it: IterationSummary, why: str) -> None:
        it.outcome = why
        ctx.event("decision", text=why, data={"action": "stop", "by": "rule"})

    try:
        current, prev = result.idea, None
        if "research" in wanted:  # once per run: the market does not change between revisions
            result.phases.append(run_research(ctx, current))
        for n in range(1, max_n + 1):
            ctx.iteration = n
            it = IterationSummary(n=n, idea=current)
            result.iterations.append(it)
            preamble = ""
            if prev:
                view = _previous_view(prev, result.phases)
                preamble = board.prompts.revision_context.format(n=n, previous=view, original=result.idea)
            hostile, rounds, it.verdict = _judge(ctx, wanted, current, preamble, result)
            result.verdict = it.verdict or result.verdict
            if prev:  # judge the revision before spending coaching + synthesis on it
                before, after = prev.verdict.mean_score, it.verdict.mean_score if it.verdict else None
                if cleared(refine, it.verdict):
                    stop(it, f"revision {n} cleared the board ({after}/10, target {refine.target_score})")
                    break
                if after is None or after <= before:
                    stop(it, f"revision {n} did not improve the board's score ({before} -> {after})")
                    break
            it.pitch = _improve(ctx, wanted, current, hostile, rounds, it.verdict, result, brief=loop) or None
            result.pitch = it.pitch or result.pitch
            if not loop:
                break
            if n == 1 and cleared(refine, it.verdict):
                stop(it, "the idea cleared the board on the first pass")
                break
            if it.verdict is None or it.verdict.mean_score is None:
                stop(it, "no verdicts to judge a revision by")
                break
            if n == max_n:
                stop(it, f"iteration limit reached ({max_n}); the last pitch has not been judged")
                break
            ctx.event(
                "decision",
                text=f"revision {n} scored {it.verdict.mean_score}/10, below the target {refine.target_score}; "
                "the refined pitch goes back to the board",
                data={"action": "iterate", "by": "rule"},
            )
            current, prev = revised_idea(it.pitch), it
    except Exception as e:  # noqa: BLE001 - surface every failure as an event, then re-raise
        ctx.event("error", text=f"{type(e).__name__}: {e}")
        raise
    result.seconds = round(time.perf_counter() - t0, 2)
    ctx.event("run_done", text=result.pitch)
    return result

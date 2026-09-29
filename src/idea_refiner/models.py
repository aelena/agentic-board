"""Pydantic models: board definitions (what you write in YAML) and run results (what you get back)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .llm import LlmSpec

Mode = Literal["hostile", "coaching"]
Phase = Literal["hostile", "deliberation", "coaching", "synthesis"]
ALL_PHASES: tuple[Phase, ...] = ("hostile", "deliberation", "coaching", "synthesis")
Decision = Literal["kill", "pivot", "proceed"]
DECISIONS: tuple[Decision, ...] = ("kill", "pivot", "proceed")  # most severe first


def _parse_llm(v):
    return LlmSpec.parse(v)


class Persona(BaseModel):
    """Optional per-mode overrides for an agent. Anything left None falls back to templates."""

    role: str | None = None
    focus: str | None = None
    goal: str | None = None
    backstory: str | None = None


class AgentSpec(BaseModel):
    """One seat on the board. ``role`` and ``focus`` are enough; the rest is generated."""

    id: str
    role: str
    focus: str = Field(description="What this expert has watched projects die from, e.g. 'technical debt'")
    hostile: Persona = Persona()
    coach: Persona = Persona()
    llm: LlmSpec | None = None
    tools: list[str] = []  # reserved for CrewAI tools, resolved by name later
    max_iter: int = 3

    parse_llm = field_validator("llm", mode="before")(_parse_llm)

    def persona(self, mode: Mode) -> Persona:
        return self.hostile if mode == "hostile" else self.coach


class SynthesizerSpec(BaseModel):
    role: str = "Startup Founder"
    goal: str = "Synthesize all feedback into a concise, investor-ready value proposition"
    backstory: str = "You are a resilient founder who turns brutal feedback into breakthrough strategy."
    llm: LlmSpec | None = None

    parse_llm = field_validator("llm", mode="before")(_parse_llm)


class ChairSpec(BaseModel):
    """Runs the deliberation: after each round decides whether another one is worth it, and whom to press."""

    role: str = "Board Chair"
    goal: str = (
        "Run a sharp deliberation: surface the real disagreements, press weak arguments, "
        "and close the debate as soon as another round would add nothing"
    )
    backstory: str = (
        "You have chaired hundreds of investment committees and design reviews. You are neutral, impatient with "
        "repetition, and you make members defend their positions instead of restating them."
    )
    llm: LlmSpec | None = None

    parse_llm = field_validator("llm", mode="before")(_parse_llm)


class DeliberationSpec(BaseModel):
    """After the hostile round, critics read each other and rebut or concede, for up to ``rounds`` rounds.
    With a chair, the chair decides after each round whether to continue; without one, the debate stops
    once positions stop moving. A unanimous board always stops early when ``stop_on_consensus``."""

    rounds: int = Field(2, ge=1, le=6)
    chair: ChairSpec | None = ChairSpec()
    stop_on_consensus: bool = True


VERDICT_FORMAT = """End your reply with your verdict as a fenced JSON block in exactly this shape:
```json
{"decision": "kill | pivot | proceed", "score": 0-10, "issues": ["most blocking issue", "..."]}
```
Your critique can be as harsh as you like, but your verdict must be calibrated, as if your reputation rode on
it: kill = the flaws are fatal and no change to the idea would fix them; pivot = viable only after a
substantial change you can name; proceed = viable as stated, despite the risks you raised.
score = how strong the idea is from your seat (0 hopeless, 5 average for ideas you see, 10 exceptional).
At most 3 issues, most severe first."""

CHAIR_FORMAT = """End your reply with your decision as a fenced JSON block in exactly this shape:
```json
{"action": "continue | close", "reason": "one sentence", "questions": {"<member id>": "your question"}}
```
Leave questions empty when you close."""


class Prompts(BaseModel):
    """All prompt templates, overridable per board. Placeholders: {role} {focus} {tone} {idea}
    {feedback} {hostile_feedback} {coaching_advice} {deliberation} {verdict} {sentences};
    deliberation: {round} {own} {others} {standing} {question}; chair: {round} {rounds} {positions}
    {standing} {movement} {ids}."""

    hostile_tone: str = "ruthlessly critical, blunt, and skeptical"
    coaching_tone: str = "constructive, supportive, and solution-oriented"
    goal: str = "Provide {tone} feedback on ideas from the perspective of a world-class {role}."
    backstory: str = (
        "You are a top-tier {role} with 15+ years of experience in best-of-breed companies at the bleeding edge "
        "of technology in very competitive environments. You have seen hundreds of projects and investments fail "
        "due to {focus}. You speak plainly, cut through the hype, and focus on hard truths, even if unpleasant."
    )
    hostile_task: str = (
        "Critique this idea with extreme skepticism from your area of expertise. Find the weaknesses that could "
        "sink it and explain how each one would. Be explicit about which flaws are fatal and which are serious "
        "but fixable; calling everything fatal is as lazy as calling everything fine:\n\n{idea}"
    )
    coaching_task: str = (
        "Here is an idea and the criticisms it received. From your area of expertise, give actionable, "
        "constructive advice to turn it into an enterprise-ready product.\n\n"
        "## Idea\n{idea}\n\n## Criticisms\n{feedback}"
    )
    deliberation_task: str = (
        "Round {round} of the board's deliberation on this idea.\n\n## Idea\n{idea}\n\n"
        "## Your position so far\n{own}\n\n## The other members' positions\n{others}\n\n"
        "## Board verdict so far\n{standing}\n\n{question}"
        "Engage with the strongest points the other members made. Say explicitly where you concede and where "
        "you hold, and why. Change your verdict only if you were persuaded, never to be agreeable. "
        "Do not repeat your earlier critique. Up to {sentences} sentences."
    )
    chair_task: str = (
        "You chair this board. Round {round} of at most {rounds} of deliberation just ended.\n\n"
        "## Idea\n{idea}\n\n## Positions after this round\n{positions}\n\n## Board verdict\n{standing}\n\n"
        "## How positions moved this round\n{movement}\n\n"
        "In 2-3 sentences, say where the board stands and what the real disagreement is. Then decide. Another round "
        "is worth it only if a specific disagreement could still be resolved or a member dodged a point. Close if "
        "positions are stable, the disagreement is irreducible, or another round would repeat itself. If you "
        "continue, put one pointed question to each member who must defend or reconsider a position. "
        "Member ids: {ids}."
    )
    synthesis_task: str = (
        "## Idea\n{idea}\n\n## Brutal critiques\n{hostile_feedback}\n\n## Board deliberation\n{deliberation}\n\n"
        "## Board verdict\n{verdict}\n\n## Expert advice\n{coaching_advice}\n\n"
        "Write a {sentences}-sentence investor-ready value proposition that addresses all concerns, "
        "followed by a short bullet list of the concrete changes made to the original idea."
    )
    expected_output: str = (
        "A focused, expert-level critique or advice in up to {sentences} sentences, specific to this idea."
    )
    synthesis_expected: str = (
        "A compelling, concise pitch that shows defensibility, security, compliance, scalability, and user value."
    )
    sentences: int = 5
    synthesis_sentences: int = 3
    verdict_format: str = VERDICT_FORMAT  # appended verbatim (not a template) to tasks that end in a verdict
    chair_format: str = CHAIR_FORMAT  # appended verbatim to the chair's task


class BoardSpec(BaseModel):
    """A board: a named set of agents plus the prompts and phases that drive them."""

    name: str
    description: str = ""
    agents: list[AgentSpec] = Field(min_length=1)
    synthesizer: SynthesizerSpec = SynthesizerSpec()
    prompts: Prompts = Prompts()
    phases: list[Phase] = list(ALL_PHASES)
    verdicts: bool = True  # critics end with a structured kill / pivot / proceed verdict
    deliberation: DeliberationSpec = DeliberationSpec()
    llm: LlmSpec | None = None  # board-wide default, overrides Settings, overridden by agent.llm
    source: str | None = None  # file path it was loaded from

    parse_llm = field_validator("llm", mode="before")(_parse_llm)

    @field_validator("agents")
    @classmethod
    def _unique_ids(cls, v: list[AgentSpec]):
        ids = [a.id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate agent ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        return v


# --- results -------------------------------------------------------------------------------------


class Verdict(BaseModel):
    decision: Decision
    score: int = Field(ge=0, le=10)
    issues: list[str] = []

    def as_text(self) -> str:
        issues = f" | issues: {'; '.join(self.issues)}" if self.issues else ""
        return f"{self.decision} ({self.score}/10){issues}"


class Tally(BaseModel):
    """Where the board stands after a round of verdicts, and who disagrees."""

    votes: dict[Decision, int] = {}
    mean_score: float | None = None
    decision: Decision | None = None  # majority; ties go to the more severe decision
    unanimous: bool = False
    dissent: list[str] = []  # agent ids voting against the majority
    missing: list[str] = []  # agent ids whose reply had no parseable verdict

    def as_text(self) -> str:
        if self.decision is None:
            return "No verdicts were given."
        votes = ", ".join(f"{n} {d}" for d, n in self.votes.items())
        out = f"{votes} (mean score {self.mean_score}/10). Majority: {self.decision}"
        out += ", unanimous." if self.unanimous else "."
        if self.dissent:
            out += f" Dissent: {', '.join(self.dissent)}."
        if self.missing:
            out += f" No verdict: {', '.join(self.missing)}."
        return out


class AgentOutput(BaseModel):
    agent_id: str
    role: str
    text: str
    verdict: Verdict | None = None
    model: str | None = None
    seconds: float | None = None

    def as_markdown(self) -> str:
        v = f"\n\n**Verdict:** {self.verdict.as_text()}" if self.verdict else ""
        return f"### {self.role}\n{self.text.strip()}{v}"


class ChairNote(BaseModel):
    action: Literal["continue", "close"]
    reason: str = ""
    summary: str = ""  # the chair's prose: where the board stands
    questions: dict[str, str] = {}  # agent id -> question for the next round


class PhaseResult(BaseModel):
    phase: Phase
    outputs: list[AgentOutput]
    round: int | None = None  # deliberation round, 1-based
    tally: Tally | None = None
    chair: ChairNote | None = None
    closed: str | None = None  # set on the last deliberation round: why the debate ended
    seconds: float | None = None

    def as_markdown(self) -> str:
        body = "\n\n".join(o.as_markdown() for o in self.outputs)
        return body + (f"\n\n**Board verdict:** {self.tally.as_text()}" if self.tally else "")


class RunRequest(BaseModel):
    """What the API/CLI accept to start a run."""

    idea: str = Field(min_length=10)
    board: str = "startup"
    llm: LlmSpec | None = None
    phases: list[Phase] | None = None
    title: str | None = None


class RunResult(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    title: str | None = None
    board: str
    idea: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phases: list[PhaseResult] = []
    pitch: str | None = None
    verdict: Tally | None = None  # the board's final standing on the idea
    seconds: float | None = None

    def phase(self, name: Phase) -> PhaseResult | None:
        return next((p for p in self.phases if p.phase == name), None)

    def all(self, name: Phase) -> list[PhaseResult]:
        """Every result of a phase, e.g. all deliberation rounds."""
        return [p for p in self.phases if p.phase == name]


EventType = Literal[
    "run_start", "phase_start", "agent_start", "agent_done", "phase_done", "decision", "run_done", "error"
]


class Event(BaseModel):
    """Progress event emitted during a run (CLI progress, SSE stream). ``decision`` marks a control-flow
    choice the board made (close or continue a deliberation), with the reason in ``text``."""

    type: EventType
    run_id: str
    phase: Phase | None = None
    round: int | None = None
    agent_id: str | None = None
    role: str | None = None
    text: str | None = None
    data: dict | None = None  # structured payload: {"verdict"} on agent_done, {"tally"} on phase_done
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

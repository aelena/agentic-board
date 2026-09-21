"""Pydantic models: board definitions (what you write in YAML) and run results (what you get back)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .llm import LlmSpec

Mode = Literal["hostile", "coaching"]
Phase = Literal["hostile", "coaching", "synthesis"]
ALL_PHASES: tuple[Phase, ...] = ("hostile", "coaching", "synthesis")


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


class Prompts(BaseModel):
    """All prompt templates, overridable per board. Placeholders: {role} {focus} {tone} {idea}
    {feedback} {hostile_feedback} {coaching_advice} {sentences}."""

    hostile_tone: str = "ruthlessly critical, blunt, and skeptical"
    coaching_tone: str = "constructive, supportive, and solution-oriented"
    goal: str = "Provide {tone} feedback on ideas from the perspective of a world-class {role}."
    backstory: str = (
        "You are a top-tier {role} with 15+ years of experience in best-of-breed companies at the bleeding edge "
        "of technology in very competitive environments. You have seen hundreds of projects and investments fail "
        "due to {focus}. You speak plainly, cut through the hype, and focus on hard truths, even if unpleasant."
    )
    hostile_task: str = (
        "Critique this idea with extreme skepticism from your area of expertise. "
        "Identify fatal flaws and explain why they are fatal:\n\n{idea}"
    )
    coaching_task: str = (
        "Here is an idea and the criticisms it received. From your area of expertise, give actionable, "
        "constructive advice to turn it into an enterprise-ready product.\n\n"
        "## Idea\n{idea}\n\n## Criticisms\n{feedback}"
    )
    synthesis_task: str = (
        "## Idea\n{idea}\n\n## Brutal critiques\n{hostile_feedback}\n\n## Expert advice\n{coaching_advice}\n\n"
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


class BoardSpec(BaseModel):
    """A board: a named set of agents plus the prompts and phases that drive them."""

    name: str
    description: str = ""
    agents: list[AgentSpec] = Field(min_length=1)
    synthesizer: SynthesizerSpec = SynthesizerSpec()
    prompts: Prompts = Prompts()
    phases: list[Phase] = list(ALL_PHASES)
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


class AgentOutput(BaseModel):
    agent_id: str
    role: str
    text: str
    model: str | None = None
    seconds: float | None = None


class PhaseResult(BaseModel):
    phase: Phase
    outputs: list[AgentOutput]
    seconds: float | None = None

    def as_markdown(self) -> str:
        return "\n\n".join(f"### {o.role}\n{o.text.strip()}" for o in self.outputs)


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
    seconds: float | None = None

    def phase(self, name: Phase) -> PhaseResult | None:
        return next((p for p in self.phases if p.phase == name), None)


class Event(BaseModel):
    """Progress event emitted during a run (CLI progress bar, SSE stream)."""

    type: Literal["run_start", "phase_start", "agent_start", "agent_done", "phase_done", "run_done", "error"]
    run_id: str
    phase: Phase | None = None
    agent_id: str | None = None
    role: str | None = None
    text: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

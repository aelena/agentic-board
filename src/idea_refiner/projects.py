"""Projects: one directory per idea, refined over many runs.

    projects/<name>/
      project.yaml        board, llm, refine defaults, and which seats sit on the board
      idea.md             the current statement of the idea
      guidelines.md       optional, injected into every agent's prompt
      voice.md            optional, ditto
      style.md            optional, ditto
      agents/*.yaml       optional custom agents (one AgentSpec each), added to the board
      memory/<seat>.md    notes each seat carries into the next run (appended after every run)
      memory/board.md     the board's outcomes, read by the chair and the synthesizer
      runs/<id>/          result.json, report.md, board.json (the exact board that ran)
      history/            earlier versions of idea.md, archived by `adopt`

Memory is plain Markdown written deterministically from each run's verdicts and advice: no LLM call,
nothing hidden, and you can edit or delete it by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from . import boards, report
from .boards import BoardError
from .config import Settings
from .config import settings as default_settings
from .llm import LlmSpec
from .models import AgentSpec, BoardSpec, ProjectContext, RefineSpec, RunResult

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")  # also keeps API-supplied names inside projects_dir
CONTEXT_FILES = ("guidelines", "voice", "style")
DOCS = ("idea", *CONTEXT_FILES)  # the Markdown files a user edits
MEMORY_ENTRIES = 4  # most recent entries per seat injected into prompts; the files keep everything
SEP = "\n\n---\n\n"


class ProjectError(Exception):
    pass


class ProjectSpec(BaseModel):
    """``project.yaml``. ``agents`` picks seats: a string keeps that seat from the board, a mapping is a
    custom agent (replacing a board seat with the same id). Omitted = every board seat. Agents from
    ``agents/*.yaml`` are always added."""

    name: str
    description: str = ""
    board: str = "startup"
    agents: list[str | AgentSpec] | None = None
    llm: LlmSpec | None = None
    refine: RefineSpec | None = None
    memory: bool = True

    @field_validator("llm", mode="before")
    @classmethod
    def _llm(cls, v):
        return LlmSpec.parse(v)


@dataclass
class Project:
    path: Path
    spec: ProjectSpec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def idea(self) -> str:
        f = self.path / "idea.md"
        return f.read_text(encoding="utf-8").strip() if f.is_file() else ""

    @property
    def runs_dir(self) -> Path:
        return self.path / "runs"

    def text(self, name: str) -> str:
        f = self.path / f"{name}.md"
        return f.read_text(encoding="utf-8").strip() if f.is_file() else ""

    def memory_file(self, seat: str) -> Path:
        return self.path / "memory" / f"{seat}.md"


def _check_name(name: str) -> str:
    if not NAME.match(name):
        raise ProjectError(f"invalid project name '{name}': use lowercase letters, digits, '-' and '_'")
    return name


def project_path(name: str, settings: Settings = default_settings) -> Path:
    return settings.projects_dir / _check_name(name)


def load_project(name: str, settings: Settings = default_settings) -> Project:
    path = project_path(name, settings)
    f = path / "project.yaml"
    if not f.is_file():
        raise ProjectError(f"unknown project '{name}' (no {f})")
    try:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        data.setdefault("name", name)
        return Project(path, ProjectSpec(**data))
    except (yaml.YAMLError, ValidationError) as e:
        raise ProjectError(f"{f}: {e}") from e


def list_projects(settings: Settings = default_settings) -> list[Project]:
    d = settings.projects_dir
    if not d.is_dir():
        return []
    return [load_project(p.name, settings) for p in sorted(d.iterdir()) if (p / "project.yaml").is_file()]


def init_project(
    name: str, idea: str, board: str = "startup", description: str = "", settings: Settings = default_settings
) -> Project:
    path = project_path(name, settings)
    if (path / "project.yaml").exists():
        raise ProjectError(f"project '{name}' already exists at {path}")
    boards.load_board(board, settings)  # fail early on an unknown board
    for sub in ("agents", "memory", "runs", "history"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    spec = {"name": name, "description": description, "board": board}
    body = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    body += (
        "# agents:            # optional: keep only these board seats and/or add custom ones\n"
        "#   - vc\n#   - {id: devrel, role: Developer Relations Lead, focus: developer tools nobody adopted}\n"
        "# llm: anthropic/claude-sonnet-5\n# refine: {max_iterations: 3, target_score: 7}\n# memory: true\n"
    )
    (path / "project.yaml").write_text(body, encoding="utf-8")
    (path / "idea.md").write_text(idea.strip() + "\n", encoding="utf-8")
    for f in CONTEXT_FILES:
        (path / f"{f}.md").touch()
    return load_project(name, settings)


def project_board(project: Project, settings: Settings = default_settings) -> BoardSpec:
    """The project's board: the base board's seats, filtered and mixed with custom agents."""
    base = boards.load_board(project.spec.board, settings)
    by_id = {a.id: a for a in base.agents}
    if project.spec.agents is None:
        seats = list(base.agents)
    else:
        seats = []
        for entry in project.spec.agents:
            if isinstance(entry, str):
                if entry not in by_id:
                    raise ProjectError(f"project '{project.name}': board '{base.name}' has no seat '{entry}'")
                seats.append(by_id[entry])
            else:
                seats.append(entry)
    agents_dir = project.path / "agents"
    for f in sorted(agents_dir.glob("*.y*ml")) if agents_dir.is_dir() else []:
        try:
            custom = AgentSpec(**(yaml.safe_load(f.read_text(encoding="utf-8")) or {}))
        except (yaml.YAMLError, ValidationError, TypeError) as e:
            raise ProjectError(f"{f}: {e}") from e
        seats = [s for s in seats if s.id != custom.id] + [custom]
    try:
        board = base.model_copy(update={"agents": seats, "llm": project.spec.llm or base.llm})
        return BoardSpec.model_validate(board.model_dump())  # re-run validation (unique ids, min one seat)
    except ValidationError as e:
        raise ProjectError(f"project '{project.name}': {e}") from e


def _recent(notes: str, n: int = MEMORY_ENTRIES) -> str:
    return SEP.join(e for e in notes.strip().split(SEP)[-n:] if e.strip())


def project_context(project: Project, board: BoardSpec) -> ProjectContext:
    parts = [(f, project.text(f)) for f in CONTEXT_FILES]
    brief = "\n\n".join(f"### {f.capitalize()}\n{t}" for f, t in parts if t)
    memory = {}
    if project.spec.memory:
        for seat in [a.id for a in board.agents] + ["board"]:
            f = project.memory_file(seat)
            if f.is_file() and (notes := _recent(f.read_text(encoding="utf-8"))):
                memory[seat] = notes
    return ProjectContext(name=project.name, brief=brief, memory=memory)


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def remember(project: Project, board: BoardSpec, result: RunResult) -> None:
    """Append this run to every seat's notes and to the board's, and snapshot the board that ran."""
    run_dir = project.runs_dir / result.id
    if run_dir.is_dir():
        (run_dir / "board.json").write_text(board.model_dump_json(indent=2), encoding="utf-8")
    if not project.spec.memory:
        return
    (project.path / "memory").mkdir(exist_ok=True)
    stamp = f"{result.created_at:%Y-%m-%d %H:%M} UTC, run {result.id}"
    last = result.iterations[-1] if result.iterations else None
    standing = result.verdict.as_text() if result.verdict else "no verdict"
    outcome = f" Outcome: {last.outcome}." if last and last.outcome else ""
    for a in board.agents:
        mine = [o for p in result.phases for o in p.outputs if o.agent_id == a.id]
        if not mine:
            continue
        verdicts = [o for p in result.phases if p.phase != "coaching" for o in p.outputs if o.agent_id == a.id]
        final = next((o.verdict for o in reversed(verdicts) if o.verdict), None)
        advice = [o for p in result.phases if p.phase == "coaching" for o in p.outputs if o.agent_id == a.id]
        lines = [f"## {stamp}"]
        if last:
            lines.append(f"Judged: {_clip(last.idea, 200)}")
        if final:
            lines.append(f"My verdict: {final.as_text()}")
        if advice:
            lines.append(f"Advice I gave: {_clip(advice[-1].text, 400)}")
        lines.append(f"The board: {standing}{outcome}")
        _append(project.memory_file(a.id), "\n".join(lines))
    board_lines = [f"## {stamp}", f"Board verdict: {standing}{outcome}"]
    if len(result.iterations) > 1:
        scores = " -> ".join(str(i.verdict.mean_score) if i.verdict else "?" for i in result.iterations)
        board_lines.append(f"Refine loop scores: {scores}")
    if result.pitch:
        board_lines.append(f"Latest pitch: {_clip(result.pitch, 600)}")
    _append(project.memory_file("board"), "\n".join(board_lines))


def _append(f: Path, entry: str) -> None:
    old = f.read_text(encoding="utf-8").strip() if f.is_file() else ""
    f.write_text((old + SEP if old else "") + entry + "\n", encoding="utf-8")


def adopt(project: Project, run_id: str | None = None) -> tuple[Path, str]:
    """Make a run's refined idea the project's idea.md; the old one is archived under history/.
    Returns (archived file, new idea)."""
    from .engine import revised_idea

    runs = report.list_runs(project.runs_dir)
    if not runs:
        raise ProjectError(f"project '{project.name}' has no runs yet")
    r = next((x for x in runs if x.id == run_id), None) if run_id else runs[0]
    if r is None:
        raise ProjectError(f"project '{project.name}' has no run '{run_id}'")
    if not r.pitch:
        raise ProjectError(f"run {r.id} has no pitch to adopt (it ran without synthesis)")
    new = revised_idea(r.pitch)
    (project.path / "history").mkdir(exist_ok=True)
    archived = project.path / "history" / f"idea-{datetime.now(UTC):%Y%m%d-%H%M%S}.md"
    archived.write_text(project.idea + "\n", encoding="utf-8")
    (project.path / "idea.md").write_text(new.strip() + "\n", encoding="utf-8")
    return archived, new


@dataclass
class Prepared:
    """Everything needed to run a project: resolved board, idea, context and loop defaults."""

    project: Project
    board: BoardSpec
    idea: str
    context: ProjectContext
    refine: RefineSpec | None


def prepare(name: str, idea: str | None = None, settings: Settings = default_settings) -> Prepared:
    project = load_project(name, settings)
    try:
        board = project_board(project, settings)
    except BoardError as e:
        raise ProjectError(str(e)) from e
    text = (idea or "").strip() or project.idea
    if len(text) < 10:
        raise ProjectError(f"project '{name}' has no idea yet: write it in {project.path / 'idea.md'}")
    return Prepared(project, board, text, project_context(project, board), project.spec.refine)

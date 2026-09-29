"""``refiner`` command line. Runs boards in-process, or against a remote API with ``--server``."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__, boards, projects, report
from .boards import BoardError
from .config import PROVIDERS, Settings
from .config import settings as default_settings
from .engine import RunError, run_board
from .llm import LlmSpec, build_llm, describe
from .models import ALL_PHASES, Event, RefineSpec, RunRequest, RunResult, Tally, Verdict
from .projects import ProjectError

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Agentic AI boardroom: stress-test and refine ideas with configurable agent boards.",
)
boards_app = typer.Typer(no_args_is_help=True, help="List, inspect, validate and scaffold boards.")
runs_app = typer.Typer(no_args_is_help=True, help="Browse saved runs.")
projects_app = typer.Typer(no_args_is_help=True, help="Projects: one idea refined over many runs, with memory.")
app.add_typer(boards_app, name="boards")
app.add_typer(runs_app, name="runs")
app.add_typer(projects_app, name="projects")
con = Console()
err = Console(stderr=True, style="red")

ProviderOpt = Annotated[
    str | None,
    typer.Option(
        "--provider",
        "-p",
        help="openai | anthropic | mistral | gemini | groq | openrouter | ollama | lmstudio | openai-compatible",
    ),
]
ModelOpt = Annotated[
    str | None, typer.Option("--model", "-m", help="Model name for the provider, e.g. gpt-4o, llama3.1")
]
BaseUrlOpt = Annotated[str | None, typer.Option("--base-url", help="Override endpoint (Ollama, LM Studio, vLLM...)")]
ApiKeyOpt = Annotated[
    str | None, typer.Option("--api-key", help="Override API key (prefer env vars)", show_default=False)
]


def _version(v: bool):
    if v:
        con.print(f"idea-refiner {__version__}")
        raise typer.Exit()


@app.callback()
def _root(version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False):
    pass


def _llm(provider, model, base_url, api_key) -> LlmSpec | None:
    spec = LlmSpec(provider=provider, model=model, base_url=base_url, api_key=api_key)
    return spec if spec.model_dump(exclude_none=True) else None


def _refine(iterate: int | None, target: float | None) -> RefineSpec | None:
    if iterate is None and target is None:
        return None
    spec = RefineSpec(max_iterations=iterate or 3)  # --target alone implies a loop
    return spec.model_copy(update={"target_score": target}) if target is not None else spec


def _fail(msg: str, code: int = 1):
    err.print(msg)
    raise typer.Exit(code)


def _read_idea(idea: str | None, file: Path | None) -> str:
    if file:
        return file.read_text(encoding="utf-8")
    if idea and idea != "-":
        return idea
    if not sys.stdin.isatty():
        return sys.stdin.read()
    _fail("Give the idea as an argument, with --file, or on stdin.")
    return ""  # unreachable


class Progress:
    """Renders engine events on the terminal."""

    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.status = None
        self.thinking: dict[str, str] = {}  # agent id -> role; agents may run in parallel
        self.lock = threading.Lock()  # events arrive from worker threads

    def _spin(self):
        if not self.thinking:
            if self.status:
                self.status.stop()
                self.status = None
            return
        label = f"[cyan]{', '.join(self.thinking.values())}[/] thinking..."
        if self.status:
            self.status.update(label)
        else:
            self.status = con.status(label)
            self.status.start()

    def __call__(self, e: Event) -> None:
        if self.quiet:
            return
        with self.lock:
            self._handle(e)

    def _handle(self, e: Event) -> None:
        match e.type:
            case "run_start":
                con.print(f"[dim]run {e.run_id} | model {e.text}[/]")
            case "phase_start":
                rev = f" [dim](revision {e.iteration})[/]" if (e.iteration or 1) > 1 else ""
                con.rule(f"[bold]{e.phase}[/]" + (f" round {e.round}" if e.round else "") + rev)
            case "decision":
                where = f" [dim]({e.phase})[/]" if e.phase else ""
                con.print(f"[bold magenta]{(e.data or {}).get('action', '')}[/]{where} {e.text}")
            case "agent_start":
                self.thinking[e.agent_id or ""] = e.role or ""
                self._spin()
            case "agent_done":
                self.thinking.pop(e.agent_id or "", None)
                self._spin()
                verdict = (e.data or {}).get("verdict")
                sub = Verdict.model_validate(verdict).as_text() if verdict else None
                con.print(Panel(Markdown(e.text or ""), title=f"[cyan]{e.role}[/]", subtitle=sub, border_style="dim"))
            case "phase_done":
                if tally := (e.data or {}).get("tally"):
                    con.print(f"[bold]board verdict:[/] {Tally.model_validate(tally).as_text()}")
            case "error":
                self.thinking.clear()
                self._spin()
                err.print(f"error: {e.text}")


@app.command()
def run(
    idea: Annotated[str | None, typer.Argument(help="The idea text, or '-' for stdin")] = None,
    file: Annotated[
        Path | None, typer.Option("--file", "-f", exists=True, dir_okay=False, help="Read the idea from a file")
    ] = None,
    board: Annotated[
        str | None, typer.Option("--board", "-b", help="Board name or path to a board YAML (default: startup)")
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", "-P", help="Run a project: its idea.md, board, guidelines and memory"),
    ] = None,
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    base_url: BaseUrlOpt = None,
    api_key: ApiKeyOpt = None,
    phases: Annotated[
        list[str] | None,
        typer.Option("--phase", help="Run only these phases: hostile, deliberation, coaching, synthesis (repeatable)"),
    ] = None,
    server: Annotated[
        str | None, typer.Option("--server", "-s", help="Run on a remote API, e.g. http://localhost:8000")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Runs directory (default: REFINER_RUNS_DIR or ./runs)")
    ] = None,
    title: Annotated[str | None, typer.Option("--title", "-t")] = None,
    iterate: Annotated[
        int | None,
        typer.Option(
            "--iterate", "-i", min=1, max=5, help="Refine loop: send the pitch back to the board up to N times"
        ),
    ] = None,
    target: Annotated[
        float | None, typer.Option("--target", min=0, max=10, help="Board mean score that ends the loop (default 7)")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the RunResult as JSON instead of a report")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No live progress, only the final output")] = False,
):
    """Run a board against an idea: hostile critique, deliberation, coaching advice, synthesis."""
    if project and board:
        _fail("--project uses the project's board; set `board:` in its project.yaml instead of --board")
    text = "" if project and idea is None and file is None else _read_idea(idea, file).strip()
    if not project and len(text) < 10:
        _fail("The idea is too short.")
    if bad := [p for p in phases or [] if p not in ALL_PHASES]:
        _fail(f"unknown phase(s) {bad}; choose from {', '.join(ALL_PHASES)}")
    req = RunRequest(
        idea=text,
        board=board,
        project=project,
        llm=_llm(provider, model, base_url, api_key),
        phases=phases or None,
        title=title,
        refine=_refine(iterate, target),
    )
    progress = Progress(quiet or as_json)
    settings = default_settings if out is None else Settings(runs_dir=out)
    try:
        if server:
            from .api.client import ApiError, Client

            try:
                client = Client(server)
                run_id = client.start_run(req)
                for e in client.events(run_id):
                    progress(e)
                result = client.result(run_id)
            except ApiError as e:
                _fail(str(e))
        elif project:
            prep = projects.prepare(project, text, settings)
            result = run_board(
                prep.board,
                prep.idea,
                request_llm=req.llm,
                phases=req.phases,
                title=title,
                refine=req.refine or prep.refine,
                context=prep.context,
                settings=settings,
                emit=progress,
            )
            saved = report.save(result, prep.project.runs_dir)
            projects.remember(prep.project, prep.board, result)
            if not (quiet or as_json):
                con.print(f"[dim]saved to {saved}; memory updated in {prep.project.path / 'memory'}[/]")
        else:
            spec = boards.load_board(board or settings.default_board, settings)
            result = run_board(
                spec,
                text,
                request_llm=req.llm,
                phases=req.phases,
                title=title,
                refine=req.refine,
                settings=settings,
                emit=progress,
            )
            saved = report.save(result, settings.runs_dir)
            if not (quiet or as_json):
                con.print(f"[dim]saved to {saved}[/]")
    except (BoardError, ProjectError, RunError, ValueError) as e:
        _fail(str(e))
    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001 - LLM/provider errors: show a clean line, not a CrewAI trace
        where = describe(req.llm or LlmSpec(), settings)
        hint = f"(while calling {where}; run `refiner check` with the same provider flags)"
        _fail(f"{type(e).__name__}: {e}\n{hint}")
    _show(result, as_json, quiet)


def _show(result: RunResult, as_json: bool, quiet: bool):
    if as_json:
        con.print_json(result.model_dump_json())
    elif quiet:
        con.print(Markdown(report.to_markdown(result)))
    elif result.pitch:
        con.rule("[bold green]refined pitch[/]")
        con.print(Panel(Markdown(result.pitch), border_style="green"))


@app.command()
def check(provider: ProviderOpt = None, model: ModelOpt = None, base_url: BaseUrlOpt = None, api_key: ApiKeyOpt = None):
    """Make one tiny LLM call to confirm the provider, key and endpoint work."""
    spec = _llm(provider, model, base_url, api_key) or LlmSpec()
    try:
        name = describe(spec, default_settings)
        with con.status(f"calling {name}..."):
            reply = build_llm(spec, default_settings).call([{"role": "user", "content": "Reply with exactly: OK"}])
        con.print(f"[green]OK[/] {name} answered: {str(reply).strip()[:80]}")
    except Exception as e:  # noqa: BLE001
        _fail(f"FAILED {type(e).__name__}: {e}")


@app.command()
def providers():
    """Show known providers and whether their API key is present."""
    t = Table("provider", "default model", "key env", "key present", "local endpoint")
    for p in PROVIDERS.values():
        mark = "[green]yes[/]" if p.key_present and p.key_env else ("[dim]n/a[/]" if not p.key_env else "[red]no[/]")
        t.add_row(p.name, p.default_model or "[dim](required)[/]", p.key_env or "", mark, p.base_url or "")
    con.print(t)
    con.print(f"auto-detected default: [bold]{default_settings.resolved_provider()}[/]")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    dev: Annotated[bool, typer.Option(help="Auto-reload and access log")] = False,
    workers: Annotated[int, typer.Option(help="Worker processes (Linux/macOS only)")] = 1,
):
    """Start the REST API (and the web UI if web/dist is built)."""
    from .api.app import serve as _serve

    _serve(host, port, dev, workers)


# --- boards ---------------------------------------------------------------------------------------


@boards_app.command("list")
def boards_list():
    """List available boards (built-in and user-defined)."""
    t = Table("name", "agents", "phases", "description", "source")
    for b in boards.list_boards():
        t.add_row(b.name, str(len(b.agents)), ",".join(b.phases), b.description.strip(), b.source or "")
    con.print(t)


@boards_app.command("show")
def boards_show(name: str):
    """Show a board's agents and how they are configured."""
    try:
        b = boards.load_board(name)
    except BoardError as e:
        _fail(str(e))
    con.print(Panel(b.description.strip() or "[dim]no description[/]", title=f"[bold]{b.name}[/]", subtitle=b.source))
    t = Table("id", "hostile role", "focus", "coach role", "coach focus", "llm")
    for a in b.agents:
        t.add_row(
            a.id,
            a.hostile.role or a.role,
            a.hostile.focus or a.focus,
            a.coach.role or a.role,
            a.coach.focus or a.focus,
            describe(a.llm, default_settings) if a.llm else "[dim]default[/]",
        )
    con.print(t)
    con.print(f"synthesizer: [bold]{b.synthesizer.role}[/] - {b.synthesizer.goal}")
    con.print(f"phases: {', '.join(b.phases)}")


@boards_app.command("validate")
def boards_validate(path: Path):
    """Validate a board YAML file."""
    try:
        b = boards.load_board_file(path)
    except BoardError as e:
        _fail(str(e))
    con.print(f"[green]OK[/] {path}: board '{b.name}' with {len(b.agents)} agents")


@boards_app.command("init")
def boards_init(name: str, directory: Annotated[Path, typer.Option("--dir", "-d")] = Path("boards")):
    """Scaffold a new board YAML in ./boards (picked up automatically)."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.yaml"
    if target.exists():
        _fail(f"{target} already exists")
    target.write_text(boards.board_template(name), encoding="utf-8")
    con.print(f'[green]OK[/] wrote {target}. Edit it, then: refiner run -b {name} "your idea"')


# --- runs -----------------------------------------------------------------------------------------


@runs_app.command("list")
def runs_list(runs_dir: Annotated[Path | None, typer.Option("--dir")] = None):
    """List saved runs."""
    t = Table("id", "created", "board", "model", "title / idea")
    for r in report.list_runs(runs_dir or default_settings.runs_dir):
        t.add_row(r.id, f"{r.created_at:%Y-%m-%d %H:%M}", r.board, r.model, (r.title or r.idea)[:70])
    con.print(t)


@runs_app.command("show")
def runs_show(
    run_id: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    runs_dir: Annotated[Path | None, typer.Option("--dir")] = None,
):
    """Print a saved run as a Markdown report (or JSON)."""
    try:
        r = report.load(runs_dir or default_settings.runs_dir, run_id)
    except FileNotFoundError:
        _fail(f"run {run_id} not found")
    con.print_json(r.model_dump_json()) if as_json else con.print(Markdown(report.to_markdown(r)))


@runs_app.command("pdf")
def runs_pdf(
    run_id: str,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    runs_dir: Annotated[Path | None, typer.Option("--dir")] = None,
):
    """Export a saved run to PDF (needs: pip install idea-refiner[pdf])."""
    d = runs_dir or default_settings.runs_dir
    try:
        path = report.to_pdf(report.load(d, run_id), output or d / run_id / "report.pdf")
    except (FileNotFoundError, RuntimeError) as e:
        _fail(str(e))
    con.print(f"[green]OK[/] {path}")


# --- projects -------------------------------------------------------------------------------------


@projects_app.command("init")
def projects_init(
    name: str,
    idea: Annotated[str | None, typer.Argument(help="The idea text, or '-' for stdin")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", exists=True, dir_okay=False)] = None,
    board: Annotated[str, typer.Option("--board", "-b")] = "startup",
    description: Annotated[str, typer.Option("--description", "-d")] = "",
):
    """Create ./projects/<name> with idea.md, guidelines/voice/style.md, agents/, memory/, runs/."""
    try:
        p = projects.init_project(name, _read_idea(idea, file), board, description)
    except (ProjectError, BoardError) as e:
        _fail(str(e))
    con.print(f"[green]OK[/] {p.path}. Fill in guidelines.md / voice.md / style.md, then: refiner run -P {name}")


@projects_app.command("list")
def projects_list():
    """List projects."""
    t = Table("name", "board", "runs", "idea")
    for p in projects.list_projects():
        t.add_row(p.name, p.spec.board, str(len(report.list_runs(p.runs_dir))), p.idea[:70])
    con.print(t)


@projects_app.command("show")
def projects_show(name: str, memory: Annotated[bool, typer.Option("--memory", "-m", help="Print the notes")] = False):
    """Show a project: its board seats, context files, memory and runs."""
    try:
        p = projects.load_project(name)
        board = projects.project_board(p)
    except (ProjectError, BoardError) as e:
        _fail(str(e))
    con.print(Panel(Markdown(p.idea or "_no idea.md yet_"), title=f"[bold]{p.name}[/]", subtitle=str(p.path)))
    con.print(f"board: [bold]{board.name}[/] | seats: {', '.join(a.id for a in board.agents)}")
    ctx = [f"{f}.md" for f in projects.CONTEXT_FILES if p.text(f)]
    con.print(f"context: {', '.join(ctx) or '[dim]none (guidelines.md, voice.md, style.md are empty)[/]'}")
    notes = sorted((p.path / "memory").glob("*.md")) if (p.path / "memory").is_dir() else []
    con.print(f"memory: {', '.join(f.stem for f in notes) or '[dim]none yet[/]'}")
    for r in report.list_runs(p.runs_dir)[:10]:
        v = r.verdict.as_text() if r.verdict else "no verdict"
        con.print(f"  [dim]{r.created_at:%Y-%m-%d %H:%M}[/] {r.id}  {v}")
    if memory:
        for f in notes:
            con.print(Panel(Markdown(f.read_text(encoding="utf-8")), title=f.stem, border_style="dim"))


@projects_app.command("adopt")
def projects_adopt(name: str, run_id: Annotated[str | None, typer.Argument(help="Default: the latest run")] = None):
    """Make a run's refined idea the project's idea.md (the old one goes to history/)."""
    try:
        archived, _ = projects.adopt(projects.load_project(name), run_id)
    except ProjectError as e:
        _fail(str(e))
    con.print(f"[green]OK[/] idea.md updated; previous version kept as {archived}")


if __name__ == "__main__":
    app()

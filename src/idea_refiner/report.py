"""Render and persist run results: JSON, Markdown, optional PDF."""

from __future__ import annotations

from pathlib import Path

from .models import PhaseResult, RunResult

PHASE_TITLES = {
    "research": "Research briefings",
    "hostile": "Hostile feedback (brutal truth)",
    "deliberation": "Deliberation",
    "coaching": "Coaching advice (actionable fixes)",
    "synthesis": "Refined pitch",
}


def phase_markdown(p: PhaseResult) -> str:
    title = PHASE_TITLES[p.phase] + (f", round {p.round}" if p.round else "")
    body = p.outputs[0].text.strip() if p.phase == "synthesis" else p.as_markdown()
    if p.chair and p.chair.summary:
        body += f"\n\n**Chair:** {p.chair.summary}"
        body += "".join(f"\n- *to {k}:* {q}" for k, q in p.chair.questions.items())
    if p.closed:
        body += f"\n\n*Deliberation closed: {p.closed}.*"
    return f"## {title}\n\n{body}\n\n"


def to_markdown(r: RunResult) -> str:
    head = f"# {r.title or 'Boardroom validation report'}\n\n"
    meta = (
        f"- Board: `{r.board}`\n- Model: `{r.model}`\n- Run: `{r.id}` at {r.created_at:%Y-%m-%d %H:%M} UTC\n"
        f"- Duration: {r.seconds or 0:.0f}s\n"
    )
    meta += f"- Board verdict: {r.verdict.as_text()}\n\n" if r.verdict else "\n"
    idea = "## Your idea\n\n> " + r.idea.strip().replace("\n", "\n> ") + "\n\n"
    if len(r.iterations) <= 1:
        return head + meta + idea + "".join(phase_markdown(p) for p in r.phases)
    log = "## Refine loop\n\n| revision | board verdict | outcome |\n|---|---|---|\n"
    for it in r.iterations:
        v = f"{it.verdict.decision} ({it.verdict.mean_score}/10)" if it.verdict else "-"
        log += f"| {it.n} | {v} | {it.outcome or 'sent back to the board'} |\n"
    body = ""
    for it in r.iterations:
        body += f"# Revision {it.n}\n\n"
        if it.n > 1:
            body += "> " + it.idea.strip().replace("\n", "\n> ") + "\n\n"
        body += "".join(phase_markdown(p) for p in r.phases if p.iteration == it.n)
    return head + meta + idea + log + "\n" + body


def save(r: RunResult, runs_dir: Path) -> Path:
    """Write ``<runs_dir>/<id>/result.json`` and ``report.md``; return the directory."""
    d = runs_dir / r.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
    (d / "report.md").write_text(to_markdown(r), encoding="utf-8")
    return d


def load(runs_dir: Path, run_id: str) -> RunResult:
    return RunResult.model_validate_json((runs_dir / run_id / "result.json").read_text(encoding="utf-8"))


def list_runs(runs_dir: Path) -> list[RunResult]:
    if not runs_dir.is_dir():
        return []
    out = [load(runs_dir, p.name) for p in runs_dir.iterdir() if (p / "result.json").is_file()]
    return sorted(out, key=lambda r: r.created_at, reverse=True)


def to_pdf(r: RunResult, path: Path) -> Path:
    """Needs the ``pdf`` extra: ``pip install idea-refiner[pdf]``."""
    try:
        from markdown import markdown
        from weasyprint import HTML
    except ImportError as e:
        raise RuntimeError("PDF export needs the pdf extra: pip install idea-refiner[pdf]") from e
    css = (
        "<style>body{font-family:sans-serif;max-width:800px;margin:2em auto;line-height:1.5}"
        "blockquote{border-left:3px solid #999;padding-left:1em;color:#444}</style>"
    )
    HTML(string=css + markdown(to_markdown(r), extensions=["extra"])).write_pdf(str(path))
    return path

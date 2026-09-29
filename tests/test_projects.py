from __future__ import annotations

import time

import pytest
from conftest import verdict_block, wants_verdict
from sanic_testing.reusable import ReusableClient

from idea_refiner import projects, report
from idea_refiner.api.app import create_app
from idea_refiner.engine import run_board
from idea_refiner.projects import ProjectError

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


@pytest.fixture
def psettings(settings, tmp_path):
    settings.projects_dir = tmp_path / "projects"
    return settings


def recording():
    seen: list[tuple[str, str]] = []

    def exe(agent, task):
        seen.append((agent.role, task.description))
        if wants_verdict(task):
            return f"{agent.role} critique" + verdict_block(
                "pivot" if "Counsel" in agent.role else "kill", 3, ["no moat"]
            )
        return f"{agent.role} output"

    return exe, seen


def test_init_layout_and_name_safety(psettings):
    p = projects.init_project("legal-diff", IDEA, settings=psettings)
    for f in ("project.yaml", "idea.md", "guidelines.md", "voice.md", "style.md"):
        assert (p.path / f).is_file()
    for d in ("agents", "memory", "runs", "history"):
        assert (p.path / d).is_dir()
    assert p.idea == IDEA and p.spec.board == "startup"
    assert [x.name for x in projects.list_projects(psettings)] == ["legal-diff"]
    with pytest.raises(ProjectError, match="already exists"):
        projects.init_project("legal-diff", IDEA, settings=psettings)
    for bad in ("../evil", "Upper", "a/b", ""):
        with pytest.raises(ProjectError, match="invalid project name"):
            projects.init_project(bad, IDEA, settings=psettings)


def test_board_mixes_builtin_and_custom_seats(psettings):
    p = projects.init_project("mix", IDEA, settings=psettings)
    (p.path / "project.yaml").write_text(
        "board: startup\nagents:\n  - vc\n  - ciso\n  - {id: devrel, role: DevRel Lead, focus: tools nobody adopted}\n",
        encoding="utf-8",
    )
    (p.path / "agents" / "ciso.yaml").write_text("id: ciso\nrole: Red Team Lead\nfocus: breaches\n", encoding="utf-8")
    board = projects.project_board(projects.load_project("mix", psettings), psettings)
    assert [(a.id, a.role) for a in board.agents] == [
        ("vc", "Hardened Venture Capitalist"),
        ("devrel", "DevRel Lead"),
        ("ciso", "Red Team Lead"),  # agents/*.yaml replaces the seat with the same id
    ]
    (p.path / "project.yaml").write_text("agents: [nope]\n", encoding="utf-8")
    with pytest.raises(ProjectError, match="has no seat 'nope'"):
        projects.project_board(projects.load_project("mix", psettings), psettings)


def _project_run(psettings, exe, name="legal"):
    prep = projects.prepare(name, settings=psettings)
    r = run_board(prep.board, prep.idea, context=prep.context, settings=psettings, execute=exe, refine=prep.refine)
    report.save(r, prep.project.runs_dir)
    projects.remember(prep.project, prep.board, r)
    return prep, r


def test_guidelines_reach_every_agent_and_memory_carries_over(psettings):
    p = projects.init_project("legal", IDEA, settings=psettings)
    (p.path / "guidelines.md").write_text("Target EU law firms only.", encoding="utf-8")
    (p.path / "voice.md").write_text("Plain English, no hype.", encoding="utf-8")
    exe, seen = recording()
    prep, r1 = _project_run(psettings, exe)
    assert r1.project == "legal" and prep.context.memory == {}
    assert all("Target EU law firms only." in d and "Plain English, no hype." in d for _, d in seen)
    assert "notes from earlier sessions" not in seen[0][1]
    vc_notes = (p.path / "memory" / "vc.md").read_text(encoding="utf-8")
    assert f"run {r1.id}" in vc_notes and "My verdict: kill (3/10) | issues: no moat" in vc_notes
    assert "Advice I gave: Venture Capitalist (Coach) output" in vc_notes
    assert "Latest pitch: Startup Founder output" in (p.path / "memory" / "board.md").read_text(encoding="utf-8")
    assert (p.runs_dir / r1.id / "board.json").is_file()

    seen.clear()
    _, r2 = _project_run(psettings, exe)
    vc_task = next(d for role, d in seen if role == "Hardened Venture Capitalist")
    assert "## Your notes from earlier sessions on this project" in vc_task and f"run {r1.id}" in vc_task
    founder_task = next(d for role, d in seen if role == "Startup Founder")
    assert "Latest pitch: Startup Founder output" in founder_task  # the synthesizer reads the board's notes
    assert (p.path / "memory" / "vc.md").read_text(encoding="utf-8").count("## ") == 2


def test_memory_injection_keeps_only_recent_entries(psettings):
    p = projects.init_project("legal", IDEA, settings=psettings)
    notes = projects.SEP.join(f"## entry {i}" for i in range(10))
    (p.path / "memory" / "vc.md").write_text(notes, encoding="utf-8")
    ctx = projects.prepare("legal", settings=psettings).context
    assert ctx.memory["vc"].count("## entry") == projects.MEMORY_ENTRIES and "entry 9" in ctx.memory["vc"]
    (p.path / "project.yaml").write_text("memory: false\n", encoding="utf-8")
    assert projects.prepare("legal", settings=psettings).context.memory == {}


def test_adopt_archives_the_old_idea(psettings):
    p = projects.init_project("legal", IDEA, settings=psettings)
    with pytest.raises(ProjectError, match="no runs yet"):
        projects.adopt(p)

    def exe(agent, task):
        if agent.role == "Startup Founder":
            return "Pitch.\n\n## Revised idea\nA local-first contract diff for EU firms, priced per seat, SOC2 ready."
        return "x" + (verdict_block() if wants_verdict(task) else "")

    _project_run(psettings, exe)
    archived, new = projects.adopt(p)
    assert new.startswith("A local-first contract diff") and p.idea == new
    assert archived.read_text(encoding="utf-8").strip() == IDEA


def test_project_api(psettings, execute):
    app = create_app(psettings, execute=execute, name=f"test_proj_{time.time_ns()}")
    with ReusableClient(app, host="127.0.0.1", port=48250) as c:
        _, r = c.post("/api/projects", json={"name": "../x", "idea": IDEA})
        assert r.status == 422 and "invalid project name" in r.json["error"]
        _, r = c.post("/api/projects", json={"name": "legal", "idea": IDEA, "board": "architecture"})
        assert r.status == 201 and r.json["board"] == "architecture"
        _, r = c.put("/api/projects/legal/docs/guidelines", json={"text": "EU only."})
        assert r.json["ok"]
        _, r = c.put("/api/projects/legal/docs/project", json={"text": "x"})
        assert r.status == 422
        _, r = c.post("/api/runs", json={"project": "nope"})
        assert r.status == 422 and "unknown project" in r.json["error"]
        _, r = c.post("/api/runs", json={"project": "legal"})
        assert r.status == 202
        run_id = r.json["id"]
        for _ in range(100):
            _, r = c.get(f"/api/runs/{run_id}")
            if r.json["status"] in ("done", "error"):
                break
            time.sleep(0.2)
        assert r.json["status"] == "done" and r.json["project"] == "legal"
        assert r.json["result"]["board"] == "architecture"
        assert (psettings.projects_dir / "legal" / "runs" / run_id / "result.json").is_file()
        _, r = c.get("/api/projects/legal")
        assert r.json["docs"]["guidelines"] == "EU only." and r.json["run_ids"] == [run_id]
        assert "sre" in r.json["memory"] and "board" in r.json["memory"]
        _, r = c.post("/api/projects/legal/adopt", json={})
        assert r.status == 200 and r.json["archived"].startswith("idea-")
        _, r = c.delete(f"/api/runs/{run_id}")
        assert r.json["deleted"] and not (psettings.projects_dir / "legal" / "runs" / run_id).exists()

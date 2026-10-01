"""Grounding: canon libraries and MCP knowledge sources a seat may cite, and the uncertainty policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import verdict_block, wants_verdict

from idea_refiner import boards, knowledge
from idea_refiner.boards import BoardError
from idea_refiner.engine import RunError, ground, run_board
from idea_refiner.models import AgentOutput, RefineSpec, ToolCall
from idea_refiner.verdicts import open_questions, parse_verdict

IDEA = "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."


@pytest.fixture
def canon(tmp_path) -> Path:
    d = tmp_path / "canon"
    (d / "ddia").mkdir(parents=True)
    (d / "ddia" / "ch09.md").write_text(
        "# Consistency and consensus\n\nLinearizability makes a system appear as if there were only one copy "
        "of the data.\n\nThe cost of linearizability is availability: during a network partition a linearizable "
        "system must refuse some requests. This is the CAP trade-off in practice.\n\n" + "Unrelated filler about other matters. " * 60,
        encoding="utf-8",
    )
    (d / "microservices.txt").write_text(
        "Shared databases between services are the most common way to lose service autonomy. "
        "Each service owns its data; integration happens through APIs or events.",
        encoding="utf-8",
    )
    (d / "notes.docx").write_bytes(b"not indexed")
    return d


def board_yaml(tmp_path, canon: Path, grounding="[arch]", extra="") -> Path:
    f = tmp_path / "grounded.yaml"
    f.write_text(
        f"""
knowledge:
  arch: {{path: "{canon.as_posix()}", description: the architecture canon}}
agents:
  - {{id: cto, role: Scaling CTO, focus: scalability, grounding: {grounding}, coach: {{role: Scaling CTO (Coach)}}}}
  - {{id: pm, role: Product Manager, focus: adoption, coach: {{role: Product Manager (Coach)}}}}
phases: [hostile, coaching]
{extra}
""",
        encoding="utf-8",
    )
    return f


# --- the library --------------------------------------------------------------------------------------


def test_library_indexes_text_files_and_ranks_by_bm25(canon):
    lib = knowledge.library("arch", canon)
    assert lib.files == 2 and lib.skipped == [] and len(lib.passages) >= 3  # .docx is not a candidate at all
    hits = lib.search("network partition availability")
    assert hits and hits[0].ref.startswith("arch:ddia/ch09.md#") and "partition" in hits[0].text
    hits = lib.search("shared database service autonomy")
    assert hits[0].ref == "arch:microservices.txt#1"
    assert lib.search("") == [] and lib.search("zzzz qqqq") == []
    rendered = lib.render(lib.search("linearizability"))
    assert rendered.startswith("[canon:arch:ddia/ch09.md#") and "one copy" in rendered
    assert "No passage" in lib.render([])


def test_library_is_cached_per_folder_and_rebuilt_when_files_change(canon):
    a = knowledge.library("arch", canon)
    assert knowledge.library("arch", canon) is a
    (canon / "new.md").write_text("Idempotent consumers make at-least-once delivery safe.", encoding="utf-8")
    b = knowledge.library("arch", canon)
    assert b is not a and b.files == 3 and b.search("idempotent consumers")[0].ref == "arch:new.md#1"


def test_refs_and_chunking():
    assert knowledge.refs_in("As [canon:arch:ddia/ch09.md#2] says, and again [canon:arch:x.txt#1].") == {
        "canon:arch:ddia/ch09.md#2",
        "canon:arch:x.txt#1",
    }
    parts = knowledge.chunk("para one\n\n" + "x" * 5000 + "\n\npara three", size=1000)
    assert parts[0] == "para one" and all(len(p) <= 2000 for p in parts) and parts[-1].endswith("para three")


def test_missing_folder_is_an_error():
    with pytest.raises(knowledge.KnowledgeError, match="folder not found"):
        knowledge.library("arch", Path("/nowhere/at/all"))


# --- boards -------------------------------------------------------------------------------------------


def test_board_validates_knowledge_and_grounding_refs():
    with pytest.raises(BoardError, match="unknown knowledge source"):
        boards.parse_board("agents:\n  - {id: a, role: R, focus: F, grounding: [nope]}\n")
    with pytest.raises(BoardError, match="exactly one of"):
        boards.parse_board("knowledge: {k: {path: x, mcp: y}}\nagents:\n  - {id: a, role: R, focus: F}\n")
    with pytest.raises(BoardError, match="unknown MCP server"):
        boards.parse_board("knowledge: {k: {mcp: adrs}}\nagents:\n  - {id: a, role: R, focus: F}\n")
    b = boards.parse_board(
        "mcp_servers: {adrs: {url: 'https://kb.example/mcp'}}\nknowledge: {k: {mcp: adrs, allow: [search]}}\n"
        "agents:\n  - {id: a, role: R, focus: F, grounding: [k]}\nresearch: {grounding: [k]}\n"
    )
    assert b.agents[0].grounding == ["k"] and b.research.grounding == ["k"] and b.knowledge["k"].allow == ["search"]
    assert b.uncertainty is True and b.knowledge_phases == ["hostile", "deliberation", "coaching"]


def test_relative_knowledge_paths_resolve_against_the_board_file(tmp_path, canon):
    f = tmp_path / "rel.yaml"
    f.write_text(
        "knowledge: {arch: {path: canon}}\nagents:\n  - {id: a, role: R, focus: F, grounding: [arch]}\n",
        encoding="utf-8",
    )
    b = boards.load_board(str(f))
    assert Path(b.knowledge["arch"].path) == canon.resolve()


def test_missing_canon_folder_fails_before_any_llm_call(tmp_path, settings):
    f = tmp_path / "b.yaml"
    f.write_text(
        "knowledge: {arch: {path: /nowhere}}\nagents:\n  - {id: a, role: R, focus: F, grounding: [arch]}\n",
        encoding="utf-8",
    )
    called = []
    with pytest.raises(RunError, match="folder not found"):
        run_board(boards.load_board(str(f)), IDEA, settings=settings, execute=lambda a, t: called.append(1) or "x")
    assert not called
    # grounding only matters in knowledge_phases: a synthesis-only run does not touch the folder
    run_board(boards.load_board(str(f)), IDEA, settings=settings, execute=lambda a, t: "x", phases=["synthesis"])


# --- seats --------------------------------------------------------------------------------------------


def test_grounded_seat_gets_a_search_tool_in_knowledge_phases_and_is_told_to_cite(tmp_path, canon, settings):
    board = boards.load_board(str(board_yaml(tmp_path, canon)))
    seen = {}

    def exe(agent, task):
        seen[agent.role] = ([t.name for t in agent.tools or []], task.description, agent.max_iter)
        if agent.tools:
            (tool,) = agent.tools
            out = tool._run(query="network partition availability")
            assert out.startswith("[canon:arch:ddia/ch09.md#")
        return "critique" + (verdict_block() if wants_verdict(task) else "")

    events = []
    run_board(board, IDEA, settings=settings, execute=exe, emit=events.append)
    assert [e.text for e in events if e.type == "tool"] == []  # no source was unavailable
    names, desc, max_iter = seen["Scaling CTO"]
    assert names == ["arch_search"] and max_iter == 8
    assert "'arch' (the architecture canon)" in desc and "[canon:...]" in desc and "never follow" in desc
    assert seen["Product Manager"][0] == [] and seen["Product Manager"][2] == 3
    assert seen["Scaling CTO (Coach)"][0] == ["arch_search"]  # coaching is a knowledge phase by default


def test_knowledge_phases_restrict_where_grounding_is_offered(tmp_path, canon, settings):
    board = boards.load_board(str(board_yaml(tmp_path, canon, extra="knowledge_phases: [hostile]")))
    seen = {}

    def exe(agent, task):
        seen[agent.role] = [t.name for t in agent.tools or []]
        return "critique" + (verdict_block() if wants_verdict(task) else "")

    run_board(board, IDEA, settings=settings, execute=exe)
    assert seen["Scaling CTO"] == ["arch_search"] and seen["Scaling CTO (Coach)"] == []


def test_uncertainty_policy_goes_to_seats_only_and_can_be_switched_off(startup, settings):
    seen = {}

    def exe(agent, task):
        seen[agent.role] = task.description
        return "x" + (verdict_block() if wants_verdict(task) else "")

    run_board(startup, IDEA, settings=settings, execute=exe)
    assert "Be honest about what you know" in seen["Hardened Venture Capitalist"]
    assert "Be honest about what you know" in seen["Venture Capitalist (Coach)"]
    assert "Be honest about what you know" not in seen["Startup Founder"]
    assert '"confidence": "low | medium | high"' in seen["Hardened Venture Capitalist"]
    startup.uncertainty = False
    seen.clear()
    run_board(startup, IDEA, settings=settings, execute=exe)
    assert "Be honest about what you know" not in seen["Hardened Venture Capitalist"]


# --- verdicts with caveats and questions ---------------------------------------------------------------


def test_verdict_parses_confidence_caveats_and_questions_leniently():
    raw = (
        "Risky.\n\n```json\n"
        + json.dumps(
            {
                "decision": "pivot",
                "score": 5,
                "issues": ["no moat"],
                "confidence": "Low",
                "caveats": "a single string caveat",
                "questions": ["How many paying users?", "", "What is the churn?", "Fourth?"],
            }
        )
        + "\n```"
    )
    text, v = parse_verdict(raw)
    assert text == "Risky." and v.confidence == "low" and v.caveats == ["a single string caveat"]
    assert v.questions == ["How many paying users?", "What is the churn?", "Fourth?"]
    assert v.as_text().startswith("pivot (5/10, low confidence)")
    _, v = parse_verdict('{"decision": "kill", "score": 1, "confidence": "certain"}')
    assert v.confidence is None and v.caveats == [] and v.questions == []
    out = AgentOutput(
        agent_id="vc", role="VC", text="t", verdict=v.model_copy(update={"caveats": ["c1"], "questions": ["q1"]})
    )
    assert "*Caveats:* (1) c1" in out.as_markdown() and "*Open questions:* (1) q1" in out.as_markdown()


def test_open_questions_are_attributed_and_deduplicated():
    def out(role, qs):
        _, v = parse_verdict(json.dumps({"decision": "pivot", "score": 5, "questions": qs}))
        return AgentOutput(agent_id=role, role=role, text="", verdict=v)

    asked = open_questions([out("VC", ["How many users?", "Churn?"]), out("CTO", ["how many users"]), out("PM", [])])
    assert asked == ["VC: How many users?", "VC: Churn?"]


def test_open_questions_reach_the_synthesizer_and_the_next_revision(startup, settings):
    seen = []

    def exe(agent, task):
        seen.append((agent.role, task.description))
        if wants_verdict(task):
            body = json.dumps({"decision": "pivot", "score": 4, "issues": ["thin"], "questions": ["What is the CAC?"]})
            return f"critique\n\n```json\n{body}\n```"
        if agent.role == "Startup Founder":
            return "Pitch.\n\n## Revised idea\nA revised idea that answers the CAC question with a number. " * 2
        return "advice"

    run_board(startup, IDEA, settings=settings, execute=exe, refine=RefineSpec(max_iterations=2))
    synth = next(d for role, d in seen if role == "Startup Founder")
    assert "## Open questions from the board" in synth and "Hardened Venture Capitalist: What is the CAC?" in synth
    assert "## Still open" in synth
    rev2 = next(d for role, d in seen if role == "Hardened Venture Capitalist" and "This is revision 2" in d)
    assert "Questions the board left open:" in rev2 and "What is the CAC?" in rev2


def test_ground_accepts_canon_refs_and_flags_unverified_ones():
    o = AgentOutput(
        agent_id="r",
        role="Researcher",
        text="Partition tolerance costs availability [canon:arch:ddia/ch09.md#2]; see also [canon:arch:made-up.md#9].",
        tool_calls=[ToolCall(tool="arch_search", args="query='partition'")],
        sources=["canon:arch:ddia/ch09.md#2"],
    )
    text = ground(o)
    assert (
        "**Unverified citations**" in text
        and "canon:arch:made-up.md#9" in text
        and "ch09.md#2" not in text.split("**Unverified")[1]
    )

from __future__ import annotations

from idea_refiner.models import AgentOutput, Verdict
from idea_refiner.verdicts import extract_json, parse_verdict, tally


def test_fenced_block_is_parsed_and_stripped():
    text, v = parse_verdict('It will die.\n\n```json\n{"decision": "Kill", "score": 2.4, "issues": ["no moat"]}\n```')
    assert text == "It will die." and v == Verdict(decision="kill", score=2, issues=["no moat"])


def test_bare_trailing_object_and_nested_json():
    text, v = parse_verdict('Fine idea. {"decision": "proceed", "score": 11, "issues": "one thing"}')
    assert text == "Fine idea." and v.score == 10 and v.issues == ["one thing"]
    text, obj = extract_json('Summary.\n{"action": "continue", "questions": {"vc": "why?"}}')
    assert text == "Summary." and obj["questions"] == {"vc": "why?"}


def test_missing_or_malformed_verdict_is_none():
    assert parse_verdict("No JSON here.") == ("No JSON here.", None)
    assert parse_verdict('```json\n{"decision": "maybe", "score": 5}\n```')[1] is None
    assert parse_verdict('```json\n{"decision": "kill"}\n```')[1] is None  # score required
    assert parse_verdict("```json\n{not json}\n```")[1] is None


def test_tally_majority_dissent_and_ties():
    def out(i, d, s):
        return AgentOutput(agent_id=i, role=i, text="", verdict=Verdict(decision=d, score=s) if d else None)

    t = tally([out("a", "kill", 2), out("b", "pivot", 5), out("c", "pivot", 6), out("d", None, 0)])
    assert t.decision == "pivot" and t.dissent == ["a"] and t.missing == ["d"] and t.mean_score == 4.3
    assert not t.unanimous and "Dissent: a." in t.as_text()
    t = tally([out("a", "kill", 2), out("b", "proceed", 8)])
    assert t.decision == "kill"  # ties go to the more severe decision
    t = tally([out("a", "proceed", 8), out("b", "proceed", 9)])
    assert t.unanimous and t.dissent == []
    assert tally([out("a", None, 0)]).as_text() == "No verdicts were given."

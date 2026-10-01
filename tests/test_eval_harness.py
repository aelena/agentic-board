"""The board-vs-single evaluation harness, exercised with fakes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import board_vs_single as ev  # noqa: E402


def test_judgement_parsing_is_lenient_and_bounded():
    raw = 'Fine.\n```json\n{"A": {"specificity": 7.6, "coverage": 11, "actionability": 0, "honesty": 5, "overall": 6}, "B": {"specificity": 4, "coverage": 4, "actionability": 4, "honesty": 4, "overall": 4}, "winner": "a", "reason": "x"}\n```'
    j = ev.parse_judgement(raw)
    assert j["A"] == {"specificity": 8, "coverage": 10, "actionability": 1, "honesty": 5, "overall": 6}
    assert j["winner"] == "A"
    assert ev.parse_judgement("no json") is None
    assert ev.parse_judgement('{"A": {"specificity": 1}, "B": {}, "winner": "A"}') is None  # incomplete scores
    assert ev.parse_judgement('{"A": {c: 5 for c in []}, "winner": "B"}'.replace("{c: 5 for c in []}", "{}")) is None


def test_decision_from_two_position_swapped_judgements():
    assert ev.decide([{"winner": "board"}, {"winner": "board"}]) == "board"
    assert ev.decide([{"winner": "board"}, {"winner": "single"}]) == "split"
    assert ev.decide([{"winner": "tie"}, {"winner": "single"}]) == "single"
    assert ev.decide([{"winner": "tie"}, {"winner": "tie"}]) == "tie"
    assert ev.decide([{"error": "x"}, {"winner": "board"}]) == "board"
    assert ev.decide([{"error": "x"}, {"error": "y"}]) == "error"


def test_judge_pair_maps_a_b_back_to_contenders(monkeypatch):
    calls = []

    def call(llm, messages):
        text = messages[-1]["content"]
        calls.append(text)
        # A is always the winner; mapped back, that is board first then single
        return '```json\n{"A": {"specificity": 8, "coverage": 8, "actionability": 8, "honesty": 8, "overall": 8}, "B": {"specificity": 5, "coverage": 5, "actionability": 5, "honesty": 5, "overall": 5}, "winner": "A", "reason": "r"}\n```'

    monkeypatch.setattr(ev, "build_llm", lambda spec, settings: object())
    js = ev.judge_pair("idea", "BOARD TEXT", "SINGLE TEXT", ev.LlmSpec(), ev.Settings(_env_file=None), call)
    assert [j["winner"] for j in js] == ["board", "single"] and ev.decide(js) == "split"
    assert js[0]["scores"]["board"]["overall"] == 8 and js[1]["scores"]["single"]["overall"] == 8
    assert calls[0].index("BOARD TEXT") < calls[0].index("SINGLE TEXT")
    assert calls[1].index("SINGLE TEXT") < calls[1].index("BOARD TEXT")
    assert "## Review A" in calls[0] and "## Review B" in calls[0]


def test_single_prompt_gives_the_lone_model_everything_the_seats_get(startup):
    msgs = ev.single_prompt(startup, "An idea.")
    user = msgs[1]["content"]
    assert "Hardened Venture Capitalist, who has watched projects die from" in msgs[0]["content"]
    assert "Be honest about what you know" in user and '"confidence": "low | medium | high"' in user
    assert "## Critiques" in user and "## Advice" in user and "## Refined pitch" in user
    assert "Venture Capitalist (Coach)" in user


@pytest.mark.parametrize("n", [2])
def test_dry_run_writes_per_idea_json_and_a_summary(tmp_path, n):
    out = ev.main(["--dry", "--max-ideas", str(n), "--out", str(tmp_path / "res"), "--seed", "1"])
    files = sorted(p.name for p in out.iterdir())
    assert "summary.md" in files and "summary.json" in files and len(files) == n + 2
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["ideas"] == n and sum(s["wins"].values()) == n and s["dry_run"] is True
    assert s["mean_board_agent_calls"] >= 11  # five seats, hostile + coaching, plus synthesis
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "# Board vs single model" in md and "| specificity |" in md
    row = json.loads(next(out.glob("legal-diff.json")).read_text(encoding="utf-8"))
    assert row["board"]["review"].startswith("## Critiques") and "## Refined pitch" in row["board"]["review"]
    assert len(row["judgements"]) == 2 and row["verdict"] in {"board", "single", "tie", "split"}

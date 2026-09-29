"""Structured verdicts pulled out of free-text agent replies, and board-level tallies.

Agents end their reply with a small JSON block. Parsing is lenient on purpose: it has to work with
local 3B models as well as frontier ones, so a missing or malformed block yields ``None`` rather than
failing the run, and the tally reports which seats did not vote.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from pydantic import ValidationError

from .models import DECISIONS, AgentOutput, Tally, Verdict

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def extract_json(text: str) -> tuple[str, dict | None]:
    """Split ``text`` into (prose, trailing JSON object). Prefers the last fenced block, then a bare
    object at the very end of the reply. Returns (text, None) when there is none."""
    for m in reversed(list(_FENCE.finditer(text))):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return (text[: m.start()] + text[m.end() :]).strip(), obj
    tail = text.rstrip()
    if tail.endswith("}"):
        window = max(0, len(tail) - 4000)
        for i in (i for i, ch in enumerate(tail) if ch == "{" and i >= window):  # outermost object first
            try:
                obj = json.loads(tail[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return tail[:i].strip(), obj
    return text.strip(), None


def parse_verdict(text: str) -> tuple[str, Verdict | None]:
    prose, obj = extract_json(text)
    if obj is None:
        return prose, None
    try:
        issues = obj.get("issues") or []
        if isinstance(issues, str):
            issues = [issues]
        return prose, Verdict(
            decision=str(obj.get("decision", "")).strip().lower(),
            score=max(0, min(10, round(float(obj.get("score"))))),
            issues=[str(i).strip() for i in issues if str(i).strip()][:3],
        )
    except (TypeError, ValueError, ValidationError):
        return prose, None


def tally(outputs: list[AgentOutput]) -> Tally:
    voted = [o for o in outputs if o.verdict]
    votes = Counter(o.verdict.decision for o in voted)
    # majority; a tie goes to the more severe decision, a board that is split on killing it has not cleared it
    decision = max(DECISIONS, key=lambda d: (votes[d], -DECISIONS.index(d))) if voted else None
    return Tally(
        votes={d: votes[d] for d in DECISIONS if votes[d]},
        mean_score=round(sum(o.verdict.score for o in voted) / len(voted), 1) if voted else None,
        decision=decision,
        unanimous=bool(voted) and len(votes) == 1 and len(voted) == len(outputs),
        dissent=[o.agent_id for o in voted if o.verdict.decision != decision],
        missing=[o.agent_id for o in outputs if not o.verdict],
    )

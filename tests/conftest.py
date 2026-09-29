from __future__ import annotations

import json

import pytest

from idea_refiner.boards import load_board
from idea_refiner.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    # _env_file=None: tests must not pick up the developer's .env (keys, providers, dirs)
    return Settings(_env_file=None, provider="ollama", model="fake-model", runs_dir=tmp_path / "runs", boards_dir=None)


@pytest.fixture
def startup():
    return load_board("startup")


def verdict_block(decision: str = "pivot", score: int = 5, issues: list[str] | None = None) -> str:
    body = json.dumps({"decision": decision, "score": score, "issues": issues or ["weak moat"]})
    return f"\n\n```json\n{body}\n```"


def wants_verdict(task) -> bool:
    return '"decision"' in task.description


def chair_block(action: str = "close", reason: str = "done", questions: dict | None = None) -> str:
    body = json.dumps({"action": action, "reason": reason, "questions": questions or {}})
    return f"\n\n```json\n{body}\n```"


def is_chair(task) -> bool:
    return '"action"' in task.description


def fake_execute(agent, task) -> str:
    """Stand-in for a CrewAI crew: answers from the agent's role, with a verdict or a chair decision when
    the task asks for one."""
    raw = f"{agent.role} says: {task.description[:40]}"
    if is_chair(task):
        return raw + chair_block()
    return raw + verdict_block() if wants_verdict(task) else raw


@pytest.fixture
def execute():
    return fake_execute

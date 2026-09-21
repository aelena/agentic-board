from __future__ import annotations

import pytest
from crewai.tasks.task_output import TaskOutput

from idea_refiner.boards import load_board
from idea_refiner.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(provider="ollama", model="fake-model", runs_dir=tmp_path / "runs", boards_dir=None)


@pytest.fixture
def startup():
    return load_board("startup")


def fake_execute(agents, tasks):
    """Stand-in for a CrewAI crew: answers each task from its agent's role and fires task callbacks."""
    outs = []
    for agent, task in zip(agents, tasks, strict=True):
        raw = f"{agent.role} says: {task.description[:40]}"
        if task.callback:
            task.callback(TaskOutput(description=task.description, raw=raw, agent=agent.role))
        outs.append(raw)
    return outs


@pytest.fixture
def execute():
    return fake_execute

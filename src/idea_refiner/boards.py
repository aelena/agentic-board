"""Discover, load and validate board YAML files.

Search order (later wins on name clash): built-in boards shipped with the package,
``~/.idea-refiner/boards``, ``./boards`` in the current directory, ``REFINER_BOARDS_DIR``.
A board can also be given as an explicit path to a ``.yaml`` file.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .config import Settings, user_boards_dirs
from .config import settings as default_settings
from .models import BoardSpec

BUILTIN_DIR = Path(__file__).parent / "boards"


class BoardError(Exception):
    pass


def _yaml_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix in (".yaml", ".yml"))


def board_files(settings: Settings = default_settings) -> dict[str, Path]:
    """Map board name -> file, user dirs overriding built-ins."""
    found: dict[str, Path] = {}
    for d in [BUILTIN_DIR, *user_boards_dirs(settings)]:
        for f in _yaml_files(d):
            found[f.stem] = f
    return found


def load_board_file(path: Path) -> BoardSpec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise BoardError(f"{path}: invalid YAML: {e}") from e
    data.setdefault("name", path.stem)
    try:
        return BoardSpec(**data, source=str(path))
    except ValidationError as e:
        raise BoardError(f"{path}: {e}") from e


def parse_board(text: str, name: str = "inline") -> BoardSpec:
    """Validate a YAML document held in memory (API ``POST /boards/validate``)."""
    try:
        data = yaml.safe_load(text) or {}
        data.setdefault("name", name)
        return BoardSpec(**data)
    except (yaml.YAMLError, ValidationError) as e:
        raise BoardError(str(e)) from e


def load_board(name_or_path: str, settings: Settings = default_settings) -> BoardSpec:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.is_file():
        if not p.is_file():
            raise BoardError(f"board file not found: {p}")
        return load_board_file(p)
    files = board_files(settings)
    if name_or_path not in files:
        raise BoardError(f"unknown board '{name_or_path}'. Available: {', '.join(sorted(files))}")
    return load_board_file(files[name_or_path])


def list_boards(settings: Settings = default_settings) -> list[BoardSpec]:
    return [load_board_file(f) for f in board_files(settings).values()]


def board_template(name: str) -> str:
    """Starter YAML for ``refiner boards init``."""
    return f"""# Board definition for idea-refiner. Every agent runs in hostile mode, then coaching mode.
name: {name}
description: Describe what this board is for.

# Optional board-wide LLM (overrides env/CLI). Agents may override again with their own `llm:`.
# llm: ollama/llama3.1
# llm: {{provider: openai-compatible, model: my-model, base_url: http://localhost:8080/v1}}

agents:
  - id: expert_a
    role: Domain Expert A
    focus: the typical way projects fail in this domain
    # coach: {{role: Domain Expert A (Coach), focus: how to do it right}}
  - id: expert_b
    role: Domain Expert B
    focus: another failure mode
    # llm: anthropic/claude-sonnet-5

synthesizer:
  role: Project Lead
  goal: Turn the critique and advice into a refined, defensible proposal
  backstory: You turn harsh feedback into a sharper plan.

# phases: [hostile, deliberation, coaching, synthesis]   # drop phases you do not want
# deliberation: {{rounds: 2}}               # critics debate; add `chair: null` to run without a chair
# prompts:                                  # override any template, see README
#   sentences: 5
"""

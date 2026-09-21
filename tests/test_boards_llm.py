from __future__ import annotations

import pytest

from idea_refiner import boards
from idea_refiner.boards import BoardError
from idea_refiner.config import PROVIDERS, Settings
from idea_refiner.engine import effective_spec
from idea_refiner.llm import LlmSpec, build_llm, litellm_model


def test_builtin_boards_load():
    names = {b.name for b in boards.list_boards(Settings(boards_dir=None))}
    assert {"startup", "architecture"} <= names
    b = boards.load_board("startup")
    assert [a.id for a in b.agents] == ["vc", "cto", "pm", "gc", "ciso"]
    assert b.agents[0].coach.role == "Venture Capitalist (Coach)"


def test_template_validates_and_init_roundtrip(tmp_path):
    text = boards.board_template("demo")
    b = boards.parse_board(text)
    assert b.name == "demo" and len(b.agents) == 2


def test_user_dir_overrides_builtin(tmp_path):
    (tmp_path / "startup.yaml").write_text("agents:\n  - {id: only, role: Solo, focus: nothing}\n", encoding="utf-8")
    s = Settings(boards_dir=tmp_path)
    b = boards.load_board("startup", s)
    assert len(b.agents) == 1 and b.source.startswith(str(tmp_path))


def test_board_errors(tmp_path):
    with pytest.raises(BoardError, match="unknown board"):
        boards.load_board("nope", Settings(boards_dir=None))
    with pytest.raises(BoardError, match="duplicate agent ids"):
        boards.parse_board("agents:\n  - {id: a, role: r, focus: f}\n  - {id: a, role: r2, focus: f}\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("agents: [\n", encoding="utf-8")
    with pytest.raises(BoardError, match="invalid YAML"):
        boards.load_board_file(bad)


def test_llm_spec_parsing_and_layering():
    assert LlmSpec.parse("ollama/llama3.1") == LlmSpec(provider="ollama", model="llama3.1")
    assert LlmSpec.parse("gpt-4o") == LlmSpec(model="gpt-4o")
    assert LlmSpec.parse({"provider": "mistral"}) == LlmSpec(provider="mistral")
    agent = LlmSpec(model="agent-model")
    request = LlmSpec(provider="openai", model="req-model")
    board = LlmSpec(provider="anthropic", temperature=0.1)
    merged = effective_spec(agent, request, board)
    assert merged == LlmSpec(provider="openai", model="agent-model", temperature=0.1)


def test_litellm_model_strings():
    assert litellm_model(LlmSpec(provider="mistral")) == "mistral/mistral-large-latest"
    assert litellm_model(LlmSpec(provider="ollama", model="qwen2.5:14b")) == "ollama/qwen2.5:14b"
    assert litellm_model(LlmSpec(provider="lmstudio")) == "openai/local-model"
    assert (
        litellm_model(LlmSpec(provider="openrouter", model="anthropic/claude-sonnet-5"))
        == "openrouter/anthropic/claude-sonnet-5"
    )
    with pytest.raises(ValueError, match="explicit model"):
        litellm_model(LlmSpec(provider="openai-compatible"))


def test_build_llm_local_and_compatible():
    s = Settings(provider="ollama", model="llama3.1")
    llm = build_llm(LlmSpec(), s)
    # CrewAI 1.x routes ollama/openai natively and strips the prefix; the endpoint must survive
    assert llm.model.endswith("llama3.1") and PROVIDERS["ollama"].base_url in llm.base_url
    llm = build_llm(
        LlmSpec(provider="openai-compatible", model="my-model", base_url="http://x:8080/v1", api_key="k"), s
    )
    assert llm.model.endswith("my-model") and llm.base_url == "http://x:8080/v1"
    with pytest.raises(ValueError, match="requires base_url"):
        build_llm(LlmSpec(provider="openai-compatible", model="m"), s)
    with pytest.raises(ValueError, match="unknown provider"):
        build_llm(LlmSpec(provider="nope"), s)


def test_provider_autodetect(monkeypatch):
    for p in PROVIDERS.values():
        if p.key_env:
            monkeypatch.delenv(p.key_env, raising=False)
    assert Settings(provider=None).resolved_provider() == "ollama"
    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    assert Settings(provider=None).resolved_provider() == "mistral"
    assert Settings(provider="groq").resolved_provider() == "groq"

"""Runtime settings and LLM provider catalogue.

Everything is overridable via environment variables prefixed ``REFINER_`` (or a ``.env`` file),
and again via CLI flags / API request bodies. Provider API keys use their vendors' standard
env var names (``OPENAI_API_KEY``, ``MISTRAL_API_KEY``, ...) so existing setups just work.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# CrewAI phones home unless told otherwise. Off by default; users can re-enable explicitly.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# CrewAI's LLM model validator trips a noisy pydantic UserWarning on every construction.
warnings.filterwarnings("ignore", message="A custom validator is returning a value other than `self`")
warnings.filterwarnings("ignore", message="function callbacks cannot be serialized")


class Provider(BaseModel):
    """A known LLM backend and how to address it through LiteLLM."""

    name: str
    prefix: str  # LiteLLM route prefix, e.g. "ollama" -> "ollama/llama3.1"
    default_model: str | None
    key_env: str | None = None  # env var holding the API key, None = no key needed
    base_url: str | None = None
    api_key: str | None = None  # constant dummy key for local OpenAI-compatible servers
    local: bool = False

    @property
    def key_present(self) -> bool:
        return self.key_env is None or bool(os.environ.get(self.key_env))


PROVIDERS: dict[str, Provider] = {
    p.name: p
    for p in (
        Provider(name="openai", prefix="openai", default_model="gpt-4o", key_env="OPENAI_API_KEY"),
        Provider(name="anthropic", prefix="anthropic", default_model="claude-sonnet-5", key_env="ANTHROPIC_API_KEY"),
        Provider(name="mistral", prefix="mistral", default_model="mistral-large-latest", key_env="MISTRAL_API_KEY"),
        Provider(name="gemini", prefix="gemini", default_model="gemini-2.5-pro", key_env="GEMINI_API_KEY"),
        Provider(name="groq", prefix="groq", default_model="llama-3.3-70b-versatile", key_env="GROQ_API_KEY"),
        Provider(
            name="openrouter",
            prefix="openrouter",
            default_model="openai/gpt-4o",
            key_env="OPENROUTER_API_KEY",
        ),
        Provider(
            name="ollama",
            prefix="ollama",
            default_model="llama3.1",
            base_url="http://localhost:11434",
            local=True,
        ),
        Provider(
            name="lmstudio",
            prefix="openai",
            default_model="local-model",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            local=True,
        ),
        # Any OpenAI-compatible server (vLLM, llama.cpp server, LiteLLM proxy...). Needs --base-url.
        Provider(
            name="openai-compatible",
            prefix="openai",
            default_model=None,
            key_env="OPENAI_API_KEY",
            local=True,
        ),
    )
}

# Order used when auto-detecting a provider from the keys present in the environment.
AUTO_ORDER = ("openai", "anthropic", "mistral", "gemini", "groq", "openrouter", "ollama")


class Settings(BaseSettings):
    """Process-wide defaults. ``REFINER_PROVIDER=ollama REFINER_MODEL=qwen2.5:14b`` etc."""

    model_config = SettingsConfigDict(env_prefix="REFINER_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    provider: str | None = None  # None = auto-detect from available keys, else ollama
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = Field(None, repr=False)
    temperature: float = 0.4
    max_tokens: int = 4096
    timeout: int = 300
    concurrency: int | None = None  # parallel agent calls; None = 1 for local providers, 4 for cloud ones

    boards_dir: Path | None = None  # extra user boards directory
    runs_dir: Path = Path("runs")
    projects_dir: Path = Path("projects")
    default_board: str = "startup"

    host: str = "127.0.0.1"
    port: int = 8000

    def resolved_provider(self) -> str:
        if self.provider:
            return self.provider
        return next((n for n in AUTO_ORDER if n != "ollama" and PROVIDERS[n].key_present), "ollama")


def user_boards_dirs(settings: Settings) -> list[Path]:
    """Directories searched for user-defined boards, in priority order (later wins)."""
    candidates = [Path.home() / ".idea-refiner" / "boards", Path.cwd() / "boards", settings.boards_dir]
    return [p for p in candidates if p and p.is_dir()]


settings = Settings()

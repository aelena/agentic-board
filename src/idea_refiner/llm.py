"""Turn a provider/model description into a CrewAI ``LLM`` instance (LiteLLM underneath)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .config import PROVIDERS, Settings

if TYPE_CHECKING:
    from crewai import LLM

# LLM objects are stateless HTTP clients, and building one costs ~1.7s on Windows (two httpx clients,
# each loading the CA store). Share them across agents and runs, keyed by the fully resolved spec.
_LLM_CACHE: dict[str, LLM] = {}


class LlmSpec(BaseModel):
    """Where and how to call a model. Every field optional; gaps are filled from Settings."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = Field(None, repr=False)
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def parse(cls, value: str | dict | LlmSpec | None) -> LlmSpec | None:
        """Accept ``"ollama/llama3.1"``, ``"gpt-4o"``, a dict, or an LlmSpec."""
        if value is None or isinstance(value, LlmSpec):
            return value
        if isinstance(value, dict):
            return cls(**value)
        head, _, tail = value.partition("/")
        if tail and head in PROVIDERS:
            return cls(provider=head, model=tail)
        return cls(model=value)

    def merged(self, base: LlmSpec | None) -> LlmSpec:
        """Return self with None fields taken from ``base``."""
        if base is None:
            return self
        return LlmSpec(**{**base.model_dump(exclude_none=True), **self.model_dump(exclude_none=True)})


def spec_from_settings(s: Settings) -> LlmSpec:
    return LlmSpec(
        provider=s.resolved_provider(),
        model=s.model,
        base_url=s.base_url,
        api_key=s.api_key,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
    )


def litellm_model(spec: LlmSpec) -> str:
    """Fully qualified LiteLLM model string, e.g. ``mistral/mistral-large-latest``."""
    prov = PROVIDERS[spec.provider or "ollama"]
    model = spec.model or prov.default_model
    if not model:
        raise ValueError(f"provider '{prov.name}' needs an explicit model")
    return model if "/" in model and prov.name not in ("openrouter", "openai-compatible") else f"{prov.prefix}/{model}"


def resolve(spec: LlmSpec, settings: Settings) -> LlmSpec:
    """Fill the gaps in ``spec`` from settings and validate the provider."""
    spec = spec.merged(spec_from_settings(settings))
    prov = PROVIDERS.get(spec.provider or "")
    if prov is None:
        raise ValueError(f"unknown provider '{spec.provider}'. Known: {', '.join(PROVIDERS)}")
    if prov.name == "openai-compatible" and not (spec.base_url or prov.base_url):
        raise ValueError("provider 'openai-compatible' requires base_url")
    return spec


def build_llm(spec: LlmSpec, settings: Settings) -> LLM:
    """CrewAI LLM for ``spec``; identical specs share one instance (see ``_LLM_CACHE``)."""
    from crewai import LLM  # deferred: importing crewai takes seconds and most CLI commands never need it

    spec = resolve(spec, settings)
    key = spec.model_dump_json() + f"|{settings.timeout}"
    if key in _LLM_CACHE:
        return _LLM_CACHE[key]
    prov = PROVIDERS[spec.provider]
    kwargs = dict(
        model=litellm_model(spec), temperature=spec.temperature, max_tokens=spec.max_tokens, timeout=settings.timeout
    )
    if base_url := spec.base_url or prov.base_url:
        kwargs["base_url"] = base_url
    if api_key := spec.api_key or prov.api_key:
        kwargs["api_key"] = api_key
    return _LLM_CACHE.setdefault(key, LLM(**kwargs))


def clear_cache() -> None:
    _LLM_CACHE.clear()


def describe(spec: LlmSpec, settings: Settings) -> str:
    spec = spec.merged(spec_from_settings(settings))
    return litellm_model(spec) + (
        f" @ {spec.base_url or PROVIDERS[spec.provider].base_url or ''}" if PROVIDERS[spec.provider].local else ""
    )

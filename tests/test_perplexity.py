"""The Perplexity research tool: answer plus the URLs it rests on, no crewai-tools needed."""

from __future__ import annotations

import pytest

from idea_refiner import tools


class FakeResponse:
    def __init__(self, data: dict, status: int = 200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class FakeClient:
    last: dict = {}

    def __init__(self, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        FakeClient.last = {"url": url, "json": json, "headers": headers}
        return FakeResponse(
            {
                "choices": [{"message": {"content": "Churn in SMB SaaS runs 3-7% monthly."}}],
                "citations": ["https://a.example/report", "https://b.example/post"],
                "search_results": [{"url": "https://b.example/post"}, {"url": "https://c.example/x"}],
            }
        )


def test_perplexity_answer_appends_deduplicated_sources(monkeypatch):
    import httpx

    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(httpx, "Client", FakeClient)
    out = tools.perplexity_answer("What is typical SMB SaaS churn?")
    assert out.startswith("Churn in SMB SaaS runs 3-7% monthly.")
    assert out.split("Sources:")[1].split() == [
        "-",
        "https://a.example/report",
        "-",
        "https://b.example/post",
        "-",
        "https://c.example/x",
    ]
    assert FakeClient.last["headers"]["Authorization"] == "Bearer pplx-test"
    assert FakeClient.last["json"]["model"] == tools.PERPLEXITY_MODEL
    assert tools.urls_in(out) == {"https://a.example/report", "https://b.example/post", "https://c.example/x"}


def test_perplexity_is_a_registered_tool_that_needs_only_its_key(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setattr(tools, "installed", lambda: False)
    problems = tools.missing(["perplexity_search"])
    assert problems == ["tool 'perplexity_search' needs PERPLEXITY_API_KEY in the environment or .env"]
    monkeypatch.setenv("PERPLEXITY_API_KEY", "x")
    assert tools.missing(["perplexity_search"]) == []
    assert "tools extra" in " ".join(tools.missing(["scrape"]))  # the crewai-tools ones still need the extra


def test_perplexity_tool_is_capped_like_the_others(monkeypatch):
    import httpx

    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(httpx, "Client", FakeClient)
    (t,) = tools.build(["perplexity_search"], budget=1)
    assert t.name == "perplexity_search"
    assert "Sources:" in t._run(query="q")
    assert t._run(query="again").startswith("Tool budget used up")


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}])
def test_empty_answer_is_empty_string(monkeypatch, payload):
    import httpx

    class Empty(FakeClient):
        def post(self, url, json=None, headers=None):
            return FakeResponse(payload)

    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(httpx, "Client", Empty)
    assert tools.perplexity_answer("q") == ""

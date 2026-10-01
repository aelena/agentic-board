"""Knowledge sources a seat may cite (``grounding:`` in board YAML).

Two kinds, declared under a board's ``knowledge:`` and referenced by name from a seat's ``grounding``:

- a **canon library**: a local folder of ``.md``, ``.txt``, ``.rst`` (and ``.pdf`` when pypdf is installed)
  indexed on this machine with BM25. The books, papers and reports you own; the index is built in memory
  per process and never written anywhere, so the texts stay where they are.
- an **MCP knowledge server**: ``mcp: <server>`` from the board's ``mcp_servers`` (ADRs, standards, past
  decisions). Resolved exactly like a seat's other MCP tools.

Either way the seat gets a search tool and is told to cite what it finds as ``[canon:<ref>]``. The refs a
tool actually returned are collected per seat (``AgentOutput.sources``), so a citation that no search
produced can be told apart from one that did.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHUNK_CHARS = 1200  # target size of one indexed passage
TOP_K = 5
TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst")
REF = re.compile(r"\[canon:([^\]\s]+)\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]+")
_STOP_WORDS = (
    "the a an and or of to in on for with by from at as is are was were be been it its this that these "
    "those which who whom what when where why how not no but if then than so such into over under "
    "between about we you they he she i our your their his her them us can could should would may might "
    "will shall do does did done have has had having there here also more most very just only any all "
    "each other some "
)
_STOP = frozenset(_STOP_WORDS.split())


class KnowledgeError(Exception):
    pass


def refs_in(text: str) -> set[str]:
    """``canon:<ref>`` citations in a text, e.g. from a seat's reply or a tool's output."""
    return {f"canon:{m}" for m in REF.findall(text)}


def tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def _read(path: Path) -> str:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        try:
            return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        except Exception:  # noqa: BLE001 - a broken PDF costs one file, not the library
            return ""
    return ""


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-aligned passages of about ``size`` characters."""
    out, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) + 2 > size:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n\n{para}" if buf else para
        while len(buf) > size * 2:  # a single huge paragraph: hard split
            out.append(buf[:size])
            buf = buf[size:]
    if buf:
        out.append(buf)
    return out


@dataclass
class Passage:
    ref: str  # <library>:<relative file>#<n>
    text: str
    score: float = 0.0


@dataclass
class Library:
    """A BM25 index over the passages of one folder."""

    name: str
    root: Path
    passages: list[Passage] = field(default_factory=list)
    files: int = 0
    skipped: list[str] = field(default_factory=list)  # files that yielded no text
    _tf: list[Counter] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avgdl: float = 0.0

    @classmethod
    def build(cls, name: str, root: Path) -> Library:
        lib = cls(name=name, root=root)
        for f in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (*TEXT_SUFFIXES, ".pdf")):
            text = _read(f)
            rel = f.relative_to(root).as_posix()
            if not text.strip():
                lib.skipped.append(rel)
                continue
            lib.files += 1
            for i, c in enumerate(chunk(text), start=1):
                lib.passages.append(Passage(ref=f"{name}:{rel}#{i}", text=c))
        lib._tf = [Counter(tokens(p.text)) for p in lib.passages]
        for tf in lib._tf:
            lib._df.update(tf.keys())
        lib._avgdl = (sum(sum(tf.values()) for tf in lib._tf) / len(lib._tf)) if lib._tf else 0.0
        return lib

    def search(self, query: str, k: int = TOP_K, k1: float = 1.5, b: float = 0.75) -> list[Passage]:
        q = tokens(query)
        if not q or not self.passages:
            return []
        n = len(self.passages)
        scored = []
        for p, tf in zip(self.passages, self._tf, strict=True):
            dl = sum(tf.values())
            s = 0.0
            for term in q:
                if (f := tf.get(term)) is None:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / (self._avgdl or 1)))
            if s > 0:
                scored.append(Passage(ref=p.ref, text=p.text, score=round(s, 3)))
        scored.sort(key=lambda p: -p.score)
        return scored[:k]

    def render(self, hits: list[Passage]) -> str:
        if not hits:
            return f"No passage in the '{self.name}' library matches that query. Try other words, or say the library has nothing on it."
        return "\n\n".join(f"[canon:{h.ref}]\n{h.text.strip()}" for h in hits)


# One index per folder per process: a library of ten books takes a second or two to tokenise.
_CACHE: dict[tuple[str, str, tuple], Library] = {}
_LOCK = threading.Lock()


def _signature(root: Path) -> tuple:
    return tuple(
        sorted(
            (p.relative_to(root).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
            for p in root.rglob("*")
            if p.is_file()
        )
    )


def library(name: str, root: str | Path) -> Library:
    root = Path(root)
    if not root.is_dir():
        raise KnowledgeError(f"knowledge '{name}': folder not found: {root}")
    key = (name, str(root.resolve()), _signature(root))
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = Library.build(name, root)
        return _CACHE[key]


def problems(name: str, spec, mcp_servers: dict) -> list[str]:
    """Why a knowledge source cannot be used, checked before a run starts."""
    if spec.path:
        root = Path(spec.path)
        if not root.is_dir():
            return [f"knowledge '{name}': folder not found: {root}"]
        if not any(p.suffix.lower() in (*TEXT_SUFFIXES, ".pdf") for p in root.rglob("*") if p.is_file()):
            return [f"knowledge '{name}': no .md/.txt/.rst/.pdf files under {root}"]
        return []
    if spec.mcp not in mcp_servers:
        return [f"knowledge '{name}': unknown MCP server '{spec.mcp}'; declare it under mcp_servers"]
    return []


def search_tool(name: str, lib: Library, description: str = "") -> Any:
    """A CrewAI tool that searches one library. Wrapped by ``tools.build`` like every other tool, so it shares
    the seat's call budget and output cap."""
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    class SearchArgs(BaseModel):
        query: str = Field(description="What to look for: a few specific words, not a sentence")

    what = description or f"the '{name}' reference library"
    tool_name = f"{name}_search"  # a class body cannot read `name` once it assigns a field called `name`
    desc = (
        f"Search {what} ({lib.files} documents). Returns the best-matching passages, each headed by a "
        f"[canon:...] reference. Cite a passage by copying its reference verbatim."
    )

    class LibrarySearch(BaseTool):
        name: str = tool_name
        description: str = desc
        args_schema: type[BaseModel] = SearchArgs

        def _run(self, query: str) -> str:
            return lib.render(lib.search(query))

    return LibrarySearch()

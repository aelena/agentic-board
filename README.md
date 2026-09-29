# Idea Refiner: an agentic AI boardroom

Stress-test an idea in front of a board of AI experts, then let the same experts coach you, then get a
refined pitch. Boards are declared in YAML, so the same engine runs a startup board, an architecture
review board, or anything you define. Works with cloud LLMs (OpenAI, Anthropic, Mistral, Gemini, Groq,
OpenRouter) and fully locally (Ollama, LM Studio, any OpenAI-compatible server).

Born from a Colab notebook ("AI Investor Board", CrewAI + Mistral). Now a CLI, a REST API and a web UI
sharing one engine.

> Don't ask AI to build your product. Ask AI to kill your idea, so only the strongest survive.

## How it works

Every board runs up to four phases:

1. **Hostile**: each agent tears the idea apart from its own domain and ends with a verdict.
2. **Deliberation**: the critics read each other and rebut or concede, round after round. After each
   round a **chair** agent decides whether another round is worth it and puts pointed questions to the
   members who dodged a point. The debate ends when the board is unanimous, the chair closes it, or the
   round limit is hit, so how long it runs depends on how the debate goes.
3. **Coaching**: the same agents, now constructive, get the idea, the critiques and where the debate
   landed, and propose fixes.
4. **Synthesis**: one agent distils everything into a refined pitch.

In the hostile round each critic ends with a structured verdict (`kill | pivot | proceed`, a 0-10 score,
up to three blocking issues). The board tallies them: majority, mean score, who dissents. Parsing is
lenient so small local models work too; a seat that returns no valid verdict is listed as missing
rather than failing the run. Set `verdicts: false` on a board to turn this off. If the opening round is already unanimous there is
nothing to debate and deliberation is skipped.

**Refine loop** (`--iterate N`, or `refine:` on a board or in the API body): the synthesizer also writes a
self-contained revised brief, and that brief goes back in front of the board as the next revision. The
critics see what they said about the previous version and must say whether each issue was actually fixed.
The loop stops when a revision clears the target (mean score >= `target_score`, at most `max_kills` kill
votes), when a revision fails to improve on the previous score, or at the iteration limit. A revision the
loop stops on is judged but not re-synthesized, so the pitch you get is the one the board last scored.

```bash
refiner run -f idea.md --iterate 3 --target 7
```

```yaml
refine: { max_iterations: 3, target_score: 7, max_kills: 0 }   # board default; 1 = single pass
```

Agents within a round run in parallel: 4 at a time for cloud providers, 1 for local servers (one model in
RAM). Override with `REFINER_CONCURRENCY`.

The built-in `startup` board is the original one: Hardened VC, Scaling CTO, Product Manager, General
Counsel, CISO / Grey-Hat Hacker, and a Founder who synthesises. The `architecture` board reviews a
technical design (SRE, AppSec, Data, FinOps, Platform, Lead Architect).

## Install

Python 3.11+.

```bash
git clone <this repo> && cd agentic-idea-refiner
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Faster alternative with [uv](https://docs.astral.sh/uv/) (same `pyproject.toml`, resolves CrewAI's
dependency tree in seconds):

```bash
pip install uv && uv pip install -e ".[dev]"
```

## Configure an LLM

Copy `.env.example` to `.env` and set what you use. Vendor keys keep their standard names.

| Provider            | Key env              | Default model              | Notes                                   |
|---------------------|----------------------|----------------------------|-----------------------------------------|
| `openai`            | `OPENAI_API_KEY`     | `gpt-4o`                   |                                         |
| `anthropic`         | `ANTHROPIC_API_KEY`  | `claude-sonnet-5`          |                                         |
| `mistral`           | `MISTRAL_API_KEY`    | `mistral-large-latest`     | what the notebook used                  |
| `gemini`            | `GEMINI_API_KEY`     | `gemini-2.5-pro`           |                                         |
| `groq`              | `GROQ_API_KEY`       | `llama-3.3-70b-versatile`  |                                         |
| `openrouter`        | `OPENROUTER_API_KEY` | `openai/gpt-4o`            |                                         |
| `ollama`            | none                 | `llama3.1`                 | `http://localhost:11434`, local         |
| `lmstudio`          | none                 | `local-model`              | `http://localhost:1234/v1`, local       |
| `openai-compatible` | optional             | required                   | vLLM, llama.cpp, LiteLLM proxy; needs `--base-url` |

The provider is auto-detected from the keys present, falling back to a local Ollama. Force it with
`REFINER_PROVIDER` / `REFINER_MODEL` in `.env`, or per run with `--provider` / `--model` / `--base-url`.

Local models: Ollama 0.3 or newer is needed for Llama 3.2 class models, and a 3B model wants roughly 4 GB
of free RAM. `refiner check` tells you quickly whether the endpoint can actually serve the model.

```bash
refiner providers                       # what is configured
refiner check                           # one tiny call to confirm key + endpoint work
refiner check -p ollama -m qwen2.5:14b
```

## CLI

```bash
refiner run "A two-pass legal document comparison system that runs locally with Qdrant and FastAPI."
refiner run -f idea.md -b architecture -p anthropic
cat idea.md | refiner run - --phase hostile          # only the brutal round
refiner run -f idea.md --server http://localhost:8000  # run on a remote API instead of in-process
refiner run -f idea.md --json > result.json
```

Each run is saved under `runs/<id>/` as `result.json` and `report.md`.

```bash
refiner runs list
refiner runs show <id>
refiner runs pdf <id>          # needs: pip install "idea-refiner[pdf]"
```

## Boards in YAML

```bash
refiner boards list
refiner boards show startup
refiner boards init my-board    # writes ./boards/my-board.yaml, picked up automatically
refiner boards validate ./boards/my-board.yaml
refiner run -b my-board "..."
refiner run -b ./somewhere/else.yaml "..."
```

Boards are searched in this order, later wins on the same name: built-in, `~/.idea-refiner/boards`,
`./boards`, `REFINER_BOARDS_DIR`. Dropping a `startup.yaml` in `./boards` overrides the built-in one.

Minimal board:

```yaml
name: legal-review
description: Contract review from three angles.

agents:
  - id: litigator
    role: Litigation Partner
    focus: clauses that lost in court
    coach: { role: Litigation Partner (Coach), focus: drafting clauses that hold }
  - id: privacy
    role: Data Protection Officer
    focus: GDPR findings nobody saw coming
    llm: ollama/llama3.1            # per-agent model override, optional

synthesizer:
  role: General Counsel
  goal: Rewrite the contract summary so it survives review
  backstory: You have negotiated hundreds of these.

deliberation:                             # optional; this is the default
  rounds: 2                               # upper bound, 1-6
  stop_on_consensus: true
  chair: { role: Board Chair }            # `chair: null` = no chair, stop when nobody changes verdict

phases: [hostile, deliberation, coaching, synthesis]   # drop any you do not want
```

An agent needs only `id`, `role` and `focus` (what this expert has watched projects die from). Everything
else is generated from templates. Overrides available per agent: `hostile: {role, focus, goal,
backstory}`, `coach: {...}`, `llm`, `max_iter`. A board-wide `llm:` applies to all agents.

Prompt templates live under `prompts:` and can be overridden per board (`goal`, `backstory`,
`hostile_task`, `deliberation_task`, `chair_task`, `coaching_task`, `synthesis_task`, `expected_output`,
`synthesis_expected`, `hostile_tone`, `coaching_tone`, `sentences`, `synthesis_sentences`, `verdict_format`,
`chair_format`). Placeholders: `{role}`, `{focus}`, `{tone}`, `{idea}`, `{feedback}`, `{hostile_feedback}`,
`{deliberation}`, `{verdict}`, `{coaching_advice}`, `{sentences}`; deliberation adds `{round}`, `{own}`,
`{others}`, `{standing}`, `{question}`; the chair gets `{round}`, `{rounds}`, `{positions}`, `{standing}`,
`{movement}`, `{ids}`. See
`src/idea_refiner/boards/architecture.yaml` for an example.

LLM resolution order, most specific wins: agent `llm` > request (`--provider`/`--model`, or the API
body) > board `llm` > environment.

## Projects: refine one idea over many sessions

A project is a directory holding one idea and everything the board should know about it:

```
projects/<name>/
  project.yaml       board, llm, refine defaults, which seats sit on the board
  idea.md            the current statement of the idea
  guidelines.md      injected into every agent's prompt (audience, constraints, non-negotiables)
  voice.md           ditto (how outputs should sound)
  style.md           ditto (format, length, terminology)
  agents/*.yaml      custom agents, one AgentSpec per file, added to the board
  memory/<seat>.md   each seat's notes from earlier runs: its verdict, issues and advice
  memory/board.md    the board's outcomes, read by the chair and the synthesizer
  runs/<id>/         result.json, report.md, board.json (the exact board that ran)
  history/           earlier versions of idea.md
```

```bash
refiner projects init flags -f idea.md -b startup
$EDITOR projects/flags/guidelines.md
refiner run -P flags --iterate 3        # idea, board, guidelines and memory come from the project
refiner projects show flags --memory
refiner projects adopt flags            # the latest refined idea becomes idea.md; the old one goes to history/
refiner run -P flags                    # the board remembers what it said last time
```

Mix built-in and custom seats in `project.yaml`: a string keeps that seat from the board, a mapping adds
a custom agent (or replaces the seat with the same id), and files in `agents/` are always added:

```yaml
board: startup
agents:
  - vc
  - ciso
  - {id: devrel, role: Developer Relations Lead, focus: developer tools nobody adopted}
refine: {max_iterations: 3, target_score: 7}
memory: true          # false = no notes written or read
```

Memory is plain Markdown written after each run from the verdicts and advice, with no extra LLM call.
The most recent 4 entries per seat go into its prompts; the files keep everything, and you can edit
or delete them. Guidelines are per project, never global.

## REST API

```bash
refiner serve                 # http://127.0.0.1:8000
refiner serve --dev           # auto-reload
refiner serve --workers 4     # Linux/macOS; Windows always runs single-process
```

| Method | Path                         | Purpose                                              |
|--------|------------------------------|------------------------------------------------------|
| GET    | `/api/health`                | version, resolved provider                           |
| GET    | `/api/providers`             | provider catalogue, whether each key is present      |
| GET    | `/api/boards`                | all boards                                           |
| GET    | `/api/boards/{name}`         | one board                                            |
| POST   | `/api/boards/validate`       | `{yaml}` -> parsed board or 422 with the error       |
| POST   | `/api/runs`                  | `{idea, board?, project?, llm?, phases?, title?, refine?}` -> 202 `{id}` |
| GET    | `/api/runs`                  | run summaries, newest first                          |
| GET    | `/api/runs/{id}`             | status plus full result when done                    |
| GET    | `/api/runs/{id}/events`      | Server-Sent Events: replays history, then streams    |
| GET    | `/api/runs/{id}/report.md`   | Markdown report                                      |
| DELETE | `/api/runs/{id}`             |                                                      |
| GET    | `/api/projects`              | project summaries                                    |
| POST   | `/api/projects`              | `{name, idea, board?, description?}` -> 201          |
| GET    | `/api/projects/{name}`       | spec, docs, resolved board, memory, run ids          |
| PUT    | `/api/projects/{name}/docs/{doc}` | `{text}` for `idea`, `guidelines`, `voice`, `style` |
| POST   | `/api/projects/{name}/adopt` | `{run_id?}`: refined idea becomes `idea.md`           |

Event types: `run_start`, `phase_start`, `agent_start`, `agent_done`, `phase_done`, `decision`, `run_done`,
`error`. Events carry `round` for deliberation and a `data` payload: the verdict on `agent_done`, the tally on
`phase_done`, `{action, by}` on `decision` (closing, continuing or skipping a debate; `iterate` or `stop` for
the refine loop, with no `phase`). Every event carries `iteration`. The chair shows up as `agent_id: chair`.

```bash
curl -s localhost:8000/api/runs -H 'content-type: application/json' \
  -d '{"idea":"...","board":"startup","llm":{"provider":"ollama","model":"llama3.1"}}'
curl -N localhost:8000/api/runs/<id>/events
```

## Docker

```bash
docker compose up --build                    # API on :8000, runs persisted in a volume
docker compose --profile local up --build    # also starts Ollama
docker compose exec ollama ollama pull llama3.1
```

Set `REFINER_PROVIDER=ollama` and `REFINER_BASE_URL=http://ollama:11434` in `.env` for the local
profile. Cloud keys in `.env` are passed through. Board YAMLs in `./boards` are mounted read-only.

## Web UI

Svelte 5 app in `web/`: pick a board, provider and model, paste the idea, and watch each agent's card
fill in live over SSE. Past runs are listed in the sidebar; reports download as Markdown.

```bash
cd web && npm install && npm run build   # produces web/dist, which `refiner serve` picks up at /
npm run dev                              # dev server on :5173 proxying /api to :8000
```

## Development

```bash
pytest -q            # no network: the engine takes an injectable executor
ruff check . && ruff format .
```

Layout:

```
src/idea_refiner/
  config.py     settings + provider catalogue
  llm.py        LlmSpec layering -> crewai.LLM
  models.py     BoardSpec / AgentSpec / Prompts (YAML schema), RunResult, Event
  boards.py     YAML discovery and validation
  engine.py     hostile -> coaching -> synthesis on CrewAI, parallel jobs, emits events
  verdicts.py   parse agent verdicts, tally the board
  projects.py   project directories, board mixing, context and memory
  report.py     markdown / json / pdf
  cli.py        typer CLI
  api/          Sanic app, run store, httpx client
  boards/       built-in boards
```

Design notes:

- The notebook's `crew.kickoff()` returns only the last task's output in sequential mode, so its
  "hostile feedback" was really just the CISO's critique. The engine collects every agent's output.
- CrewAI stays as the orchestration layer so agent tools, delegation and memory can be added later.
- CrewAI telemetry is disabled by default (`CREWAI_TELEMETRY_OPT_OUT`).

## Roadmap

- **Web UI**: board YAML editor and validator, run comparison.
- **Support agents**: a scribe (minutes and notes), a cross-checker (contradictions and unsupported
  claims across agents), a deep researcher (grounded briefings via search and MCP knowledge bases) and
  configurable synthesizers, supporting the main board members.
- **Structured output**: upload a JSON template and get schema-validated agent output.
- **Report templates**: render the report into a user-supplied `.docx` (corporate template).
- **Agent tools**: `crewai_tools` built-ins referenced by name from YAML (scraper, search), then opt-in
  user Python tool scripts loaded from a local plugin directory, disabled by default and never via upload
  without explicit configuration.
- **MCP servers**: user-configured MCP servers (internal tools, APIs, knowledge bases) exposed to agents
  as tools.
- **Voice input**: record in the browser, transcribe, send to the board.
- **Persona history**: version-controlled changes to roles, backstories and goals per project (each run
  already snapshots its exact board in `board.json`).
- **Scribe memory**: an optional LLM scribe that condenses each seat's notes, instead of the verbatim ones.

## License

MIT

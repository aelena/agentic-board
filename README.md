# Idea Refiner: an agentic AI boardroom

Stress-test an idea in front of a board of AI experts, then let the same experts coach you, then get a
refined pitch. Boards are declared in YAML, so the same engine runs a startup board, an architecture
review board, or anything you define. Works with cloud LLMs (OpenAI, Anthropic, Mistral, Gemini, Groq,
OpenRouter) and fully locally (Ollama, LM Studio, any OpenAI-compatible server).

Born from a Colab notebook ("AI Investor Board", CrewAI + Mistral). Now a CLI, a REST API and a web UI
sharing one engine.

> Don't ask AI to build your product. Ask AI to kill your idea, so only the strongest survive.

## How it works

Every board runs three phases:

1. **Hostile**: each agent tears the idea apart from its own domain.
2. **Coaching**: the same agents, now constructive, get the idea *and* the critiques and propose fixes.
3. **Synthesis**: one agent distils everything into a refined pitch.

In the hostile round each critic ends with a structured verdict (`kill | pivot | proceed`, a 0-10 score,
up to three blocking issues). The board tallies them: majority, mean score, who dissents. Parsing is
lenient so small local models work too; a seat that returns no valid verdict is listed as missing
rather than failing the run. Set `verdicts: false` on a board to turn this off.

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

phases: [hostile, coaching, synthesis]   # drop any you do not want
```

An agent needs only `id`, `role` and `focus` (what this expert has watched projects die from). Everything
else is generated from templates. Overrides available per agent: `hostile: {role, focus, goal,
backstory}`, `coach: {...}`, `llm`, `max_iter`. A board-wide `llm:` applies to all agents.

Prompt templates live under `prompts:` and can be overridden per board (`goal`, `backstory`,
`hostile_task`, `coaching_task`, `synthesis_task`, `expected_output`, `synthesis_expected`,
`hostile_tone`, `coaching_tone`, `sentences`, `synthesis_sentences`). Placeholders: `{role}`, `{focus}`,
`{tone}`, `{idea}`, `{feedback}`, `{hostile_feedback}`, `{coaching_advice}`, `{sentences}`. See
`src/idea_refiner/boards/architecture.yaml` for an example.

LLM resolution order, most specific wins: agent `llm` > request (`--provider`/`--model`, or the API
body) > board `llm` > environment.

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
| POST   | `/api/runs`                  | `{idea, board?, llm?, phases?, title?}` -> 202 `{id}` |
| GET    | `/api/runs`                  | run summaries, newest first                          |
| GET    | `/api/runs/{id}`             | status plus full result when done                    |
| GET    | `/api/runs/{id}/events`      | Server-Sent Events: replays history, then streams    |
| GET    | `/api/runs/{id}/report.md`   | Markdown report                                      |
| DELETE | `/api/runs/{id}`             |                                                      |

Event types: `run_start`, `phase_start`, `agent_start`, `agent_done`, `phase_done`, `run_done`, `error`.

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
- **Projects**: one directory/resource per idea holding `idea.md`, project-level `guidelines.md`,
  `voice.md`, `style.md` injected into every agent, a mix of custom and built-in agents, runs and
  history. Guidelines are per project, never global.
- **Board deliberation**: agents read and rebut each other over configurable rounds, optionally chaired,
  ending in scored verdicts so consensus and dissent are explicit, instead of five monologues and a summary.
- **Support agents**: a scribe (minutes and notes), a cross-checker (contradictions and unsupported
  claims across agents), a deep researcher (grounded briefings via search and MCP knowledge bases) and
  configurable synthesizers, supporting the main board members.
- **Agent memory**: each agent's thinking stored as inspectable notes so an idea can be refined over
  several turns.
- **Structured output**: upload a JSON template and get schema-validated agent output.
- **Report templates**: render the report into a user-supplied `.docx` (corporate template).
- **Agent tools**: `crewai_tools` built-ins referenced by name from YAML (scraper, search), then opt-in
  user Python tool scripts loaded from a local plugin directory, disabled by default and never via upload
  without explicit configuration.
- **MCP servers**: user-configured MCP servers (internal tools, APIs, knowledge bases) exposed to agents
  as tools.
- **Voice input**: record in the browser, transcribe, send to the board.
- **Persona history**: version-controlled changes to roles, backstories and goals per project.

## License

MIT

"""Does a board of agents beat one well-prompted model? A blind, position-swapped comparison.

For each idea in ``ideas.yaml``:

- **Board**: the ``startup`` board runs hostile, deliberation, coaching and synthesis (the default phases).
  Its review is the hostile critiques with verdicts, the coaching advice and the refined pitch.
- **Single**: the same model, one call, given the same five roles, the same verdict format and the same
  uncertainty policy, asked for the same three sections. "Well-prompted" means it gets every instruction
  the seats get; what it lacks is separate contexts, a chair, and a second look.
- **Judge**: a model (ideally a different one, ``--judge-model``) sees the two reviews as A and B, blind,
  scores each on five criteria and names a winner. Every pair is judged twice with the order swapped, so
  position bias shows up as a split decision instead of a false win.

Results land in ``evals/results/<timestamp>/``: one JSON per idea, ``summary.md`` and ``summary.json``.
Run ``--dry`` to exercise the harness with fakes and no keys.

    python evals/board_vs_single.py --max-ideas 2          # smoke
    python evals/board_vs_single.py --judge-model gpt-4.1   # the real thing
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_refiner import boards  # noqa: E402
from idea_refiner.config import Settings  # noqa: E402
from idea_refiner.engine import crew_execute, run_board  # noqa: E402
from idea_refiner.llm import LlmSpec, build_llm, describe  # noqa: E402
from idea_refiner.models import UNCERTAINTY_POLICY, VERDICT_FORMAT, BoardSpec, RunResult  # noqa: E402
from idea_refiner.verdicts import extract_json  # noqa: E402

HERE = Path(__file__).parent
CRITERIA = ("specificity", "coverage", "actionability", "honesty", "overall")
CRITERIA_TEXT = """\
- specificity: the critique engages this idea's actual numbers, market and constraints, not generic startup advice.
- coverage: it finds the distinct, real risks across business, technology, product, legal and security.
- actionability: the advice and the pitch contain concrete changes a founder could make next month.
- honesty: claims are calibrated; it separates what it knows from what it infers, states caveats and
  questions, and invents no facts.
- overall: which review would a serious investor or CTO rather have received."""


# --- the two contenders --------------------------------------------------------------------------------


def board_review(result: RunResult) -> str:
    hostile = result.phase("hostile")
    coaching = result.phase("coaching")
    parts = []
    if hostile:
        parts.append("## Critiques\n\n" + hostile.as_markdown())
    if coaching:
        parts.append("## Advice\n\n" + coaching.as_markdown())
    if result.pitch:
        parts.append("## Refined pitch\n\n" + result.pitch.strip())
    return "\n\n".join(parts)


def single_prompt(board: BoardSpec, idea: str) -> list[dict]:
    t = board.prompts
    roles = "\n".join(f"- {a.role}, who has watched projects die from {a.focus}" for a in board.agents)
    coaches = "\n".join(f"- {a.coach.role or a.role}" for a in board.agents)
    each_seat = "End each seat's critique with its verdict"
    system = (
        "You are a complete board of experts reviewing a business idea. The seats are:\n"
        f"{roles}\n\nYou write every seat's contribution yourself, each in its own voice, from its own expertise."
    )
    user = (
        f"## Idea\n{idea}\n\n"
        "Produce three sections.\n\n"
        f"## Critiques\nFor each seat, under a '### <role>' heading: {t.hostile_task.split(':')[0]}. "
        f"Up to {t.sentences} sentences, {t.hostile_tone}. {UNCERTAINTY_POLICY}\n"
        f"{VERDICT_FORMAT.replace('End your reply with your verdict', each_seat)}\n\n"
        f"## Advice\nFor each seat, now {t.coaching_tone}, under a '### <role>' heading using these names:\n{coaches}\n"
        f"give actionable, constructive advice to turn the idea into an enterprise-ready product, up to {t.sentences} sentences.\n\n"
        f"## Refined pitch\n{t.synthesis_task.split('## Expert advice')[-1].split(chr(10))[-1].format(sentences=t.synthesis_sentences)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def single_review(board: BoardSpec, idea: str, spec: LlmSpec, settings: Settings, call) -> str:
    return call(build_llm(spec, settings), single_prompt(board, idea))


# --- the judge -----------------------------------------------------------------------------------------


def judge_prompt(idea: str, a: str, b: str) -> list[dict]:
    system = (
        "You are an experienced investor and CTO judging two independent written reviews of the same business "
        "idea. You do not know who or what wrote them. Judge only what is on the page."
    )
    user = (
        f"## The idea\n{idea}\n\n## Review A\n{a}\n\n## Review B\n{b}\n\n"
        f"Score each review from 1 to 10 on every criterion:\n{CRITERIA_TEXT}\n\n"
        "Then name the winner. Be decisive: 'tie' only when you genuinely cannot separate them. End with a "
        "fenced JSON block in exactly this shape:\n"
        '```json\n{"A": {"specificity": 0, "coverage": 0, "actionability": 0, "honesty": 0, "overall": 0},\n'
        ' "B": {"specificity": 0, "coverage": 0, "actionability": 0, "honesty": 0, "overall": 0},\n'
        ' "winner": "A | B | tie", "reason": "one sentence"}\n```'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judgement(raw: str) -> dict | None:
    _, obj = extract_json(raw)
    if not obj or "winner" not in obj:
        return None
    try:
        out = {side: {c: max(1, min(10, int(round(float(obj[side][c]))))) for c in CRITERIA} for side in ("A", "B")}
    except (KeyError, TypeError, ValueError):
        return None
    winner = str(obj["winner"]).strip().upper()
    out["winner"] = winner if winner in ("A", "B") else "tie"
    out["reason"] = str(obj.get("reason", ""))[:300]
    return out


def judge_pair(idea: str, board_text: str, single_text: str, spec: LlmSpec, settings: Settings, call) -> list[dict]:
    """Two judgements, order swapped, each mapped back to board/single."""
    llm = build_llm(spec, settings)
    verdicts = []
    for order in (("board", "single"), ("single", "board")):
        a, b = (board_text, single_text) if order[0] == "board" else (single_text, board_text)
        j = parse_judgement(call(llm, judge_prompt(idea, a, b)))
        if j is None:
            verdicts.append({"order": order, "error": "no parseable judgement"})
            continue
        mapped = {order[0]: j["A"], order[1]: j["B"]}
        winner = j["winner"] if j["winner"] == "tie" else order[0 if j["winner"] == "A" else 1]
        verdicts.append({"order": order, "scores": mapped, "winner": winner, "reason": j["reason"]})
    return verdicts


def decide(judgements: list[dict]) -> str:
    """board / single / tie / split, from the two position-swapped judgements."""
    wins = [j.get("winner") for j in judgements if "winner" in j]
    if len(wins) < 2:
        return wins[0] if wins else "error"
    if wins[0] == wins[1]:
        return wins[0]
    if "tie" in wins:
        return next(w for w in wins if w != "tie")
    return "split"


# --- the run -------------------------------------------------------------------------------------------


def llm_call(llm, messages: list[dict]) -> str:
    return str(llm.call(messages))


def approx_tokens(text: str) -> int:
    return len(text) // 4


def run(args) -> Path:
    settings = Settings()
    spec = LlmSpec.parse(args.model) or LlmSpec()
    judge_spec = LlmSpec.parse(args.judge_model) or spec
    board = boards.load_board(args.board, settings)
    ideas = yaml.safe_load((HERE / "ideas.yaml").read_text(encoding="utf-8"))["ideas"]
    if args.ids:
        ideas = [i for i in ideas if i["id"] in args.ids]
    ideas = ideas[: args.max_ideas]
    out_dir = Path(args.out) if args.out else HERE / "results" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry:
        execute, call = _fake_execute, _fake_call
    else:
        execute, call = crew_execute, llm_call

    rows = []
    for n, idea in enumerate(ideas, 1):
        text = idea["text"].strip()
        done = out_dir / f"{idea['id']}.json"
        if args.resume and done.is_file():
            rows.append(json.loads(done.read_text(encoding="utf-8")))
            print(f"[{n}/{len(ideas)}] {idea['id']}: already judged, kept")
            continue
        print(f"[{n}/{len(ideas)}] {idea['id']}: board...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = run_board(board, text, request_llm=spec, settings=settings, execute=execute, title=idea["title"])
        board_text = board_review(result)
        board_s = round(time.perf_counter() - t0, 1)
        print(f"{board_s}s; single...", end=" ", flush=True)
        t0 = time.perf_counter()
        single_text = single_review(board, text, spec, settings, call)
        single_s = round(time.perf_counter() - t0, 1)
        print(f"{single_s}s; judging...", end=" ", flush=True)
        judgements = judge_pair(text, board_text, single_text, judge_spec, settings, call)
        verdict = decide(judgements)
        print(verdict)
        row = {
            "id": idea["id"],
            "title": idea["title"],
            "verdict": verdict,
            "judgements": judgements,
            "board": {
                "seconds": board_s,
                "approx_output_tokens": approx_tokens("".join(o.text for p in result.phases for o in p.outputs)),
                "agent_calls": sum(len(p.outputs) for p in result.phases),
                "tally": result.verdict.as_text() if result.verdict else None,
                "review": board_text,
            },
            "single": {"seconds": single_s, "approx_output_tokens": approx_tokens(single_text), "review": single_text},
        }
        rows.append(row)
        (out_dir / f"{idea['id']}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarise(rows, board.name, describe(spec, settings), describe(judge_spec, settings), args.dry)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(summary, rows), encoding="utf-8")
    print(f"\n{summary_markdown(summary, rows)}\nwritten to {out_dir}")
    return out_dir


def summarise(rows: list[dict], board_name: str, model: str, judge: str, dry: bool) -> dict:
    verdicts = [r["verdict"] for r in rows]
    scores = {side: {c: [] for c in CRITERIA} for side in ("board", "single")}
    for r in rows:
        for j in r["judgements"]:
            for side in ("board", "single"):
                for c in CRITERIA:
                    if "scores" in j:
                        scores[side][c].append(j["scores"][side][c])
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else None  # noqa: E731
    return {
        "board": board_name,
        "model": model,
        "judge": judge,
        "dry_run": dry,
        "ideas": len(rows),
        "wins": {k: verdicts.count(k) for k in ("board", "single", "tie", "split", "error")},
        "mean_scores": {side: {c: mean(v) for c, v in scores[side].items()} for side in scores},
        "mean_seconds": {side: mean([r[side]["seconds"] for r in rows]) for side in ("board", "single")},
        "mean_approx_output_tokens": {
            side: mean([r[side]["approx_output_tokens"] for r in rows]) for side in ("board", "single")
        },
        "mean_board_agent_calls": mean([r["board"]["agent_calls"] for r in rows]),
    }


def summary_markdown(s: dict, rows: list[dict]) -> str:
    w = s["wins"]
    head = (
        f"# Board vs single model\n\n"
        f"Board `{s['board']}` vs one call of the same model, {s['ideas']} ideas, judged blind twice each "
        f"(order swapped) by `{s['judge']}`. Model under test: `{s['model']}`.{' DRY RUN with fakes.' if s['dry_run'] else ''}\n\n"
        f"**Board wins {w['board']}, single wins {w['single']}, ties {w['tie']}, split decisions {w['split']}.**\n\n"
    )
    crit = "| criterion | board | single |\n|---|---|---|\n"
    for c in CRITERIA:
        crit += f"| {c} | {s['mean_scores']['board'][c]} | {s['mean_scores']['single'][c]} |\n"
    cost = (
        f"\n| | board | single |\n|---|---|---|\n"
        f"| mean seconds | {s['mean_seconds']['board']} | {s['mean_seconds']['single']} |\n"
        f"| mean output tokens (approx.) | {s['mean_approx_output_tokens']['board']} | {s['mean_approx_output_tokens']['single']} |\n"
        f"| LLM calls | {s['mean_board_agent_calls']} agent calls | 1 |\n\n"
    )
    per = "| idea | verdict | judgement 1 | judgement 2 |\n|---|---|---|---|\n"
    for r in rows:
        js = [
            f"{j.get('winner', 'error')} ({j['scores']['board']['overall']} vs {j['scores']['single']['overall']})"
            if "scores" in j
            else "error"
            for j in r["judgements"]
        ]
        per += f"| {r['title']} | **{r['verdict']}** | {js[0] if js else '-'} | {js[1] if len(js) > 1 else '-'} |\n"
    return head + "## Mean scores (1-10)\n\n" + crit + cost + "## Per idea\n\n" + per


# --- fakes for --dry -----------------------------------------------------------------------------------


def _fake_execute(agent, task) -> str:
    body = {"decision": "pivot", "score": 5, "issues": ["thin moat"], "confidence": "low", "questions": ["CAC?"]}
    tail = f"\n\n```json\n{json.dumps(body)}\n```" if '"decision"' in task.description else ""
    if '"action"' in task.description:
        tail = '\n\n```json\n{"action": "close", "reason": "done", "questions": {}}\n```'
    return f"{agent.role} fake reply.{tail}"


def _fake_call(llm, messages: list[dict]) -> str:
    text = messages[-1]["content"]
    if "## Review A" in text:
        a = {c: random.randint(5, 8) for c in CRITERIA}
        b = {c: random.randint(5, 8) for c in CRITERIA}
        return f"Judged.\n```json\n{json.dumps({'A': a, 'B': b, 'winner': random.choice(['A', 'B', 'tie']), 'reason': 'fake'})}\n```"
    return "### Hardened Venture Capitalist\nfake single-model review\n\n## Refined pitch\nfake pitch"


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--board", default="startup")
    p.add_argument("--model", default=None, help="model under test, e.g. gpt-4o (default: settings)")
    p.add_argument("--judge-model", default=None, help="judge model; ideally a different one (default: --model)")
    p.add_argument("--max-ideas", type=int, default=20)
    p.add_argument("--ids", nargs="*", help="only these idea ids")
    p.add_argument("--out", default=None, help="results directory (default: evals/results/<timestamp>)")
    p.add_argument("--dry", action="store_true", help="fakes instead of LLM calls: exercises the harness")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="keep ideas already judged in --out; run only the rest")
    args = p.parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
    return run(args)


if __name__ == "__main__":
    main()

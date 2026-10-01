"""Render a saved run (runs/<id>/result.json) as a terminal-style animated GIF for the README.

The frames replay the run as the CLI shows it: the idea, each phase, every seat's verdict as it lands,
the board tally, the chair's decisions, the refine-loop decision, and the pitch. Nothing is invented: every
line comes from the saved RunResult, so the GIF is a faithful (sped-up) record of one real run.

    python tools/make_demo_gif.py runs/<id> docs/demo.gif
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea_refiner.models import RunResult  # noqa: E402

W, H = 1000, 600
PAD, LINE = 22, 20
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\CascadiaMono.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
BG, FG, DIM, GREEN, RED, YELLOW, CYAN, MAGENTA = (
    (24, 24, 28),
    (220, 220, 220),
    (130, 130, 140),
    (120, 200, 120),
    (230, 110, 110),
    (230, 200, 100),
    (110, 190, 230),
    (200, 140, 230),
)
COLOUR = {"kill": RED, "pivot": YELLOW, "proceed": GREEN}
TITLES = {
    "research": "research phase",
    "hostile": "hostile round",
    "deliberation": "deliberation",
    "coaching": "coaching round",
    "synthesis": "synthesis",
}


def font(size: int = 15) -> ImageFont.FreeTypeFont:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


class Terminal:
    """A scrolling list of (text, colour) lines rendered into frames."""

    def __init__(self):
        self.lines: list[tuple[str, tuple]] = []
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []
        self.font = font()
        self.rows = (H - 2 * PAD) // LINE

    def say(self, text: str = "", colour=FG, hold: int = 350, wrap: int = 92) -> None:
        for chunk in textwrap.wrap(text, wrap) or [""]:
            self.lines.append((chunk, colour))
        self.snap(hold)

    def snap(self, hold: int) -> None:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        visible = self.lines[-self.rows :]
        for i, (text, colour) in enumerate(visible):
            d.text((PAD, PAD + i * LINE), text, font=self.font, fill=colour)
        # a title bar
        d.rectangle([0, 0, W, 4], fill=MAGENTA)
        self.frames.append(img)
        self.durations.append(hold)

    def save(self, path: Path) -> None:
        self.durations[-1] = 4000  # linger on the last frame
        self.frames[0].save(
            path, save_all=True, append_images=self.frames[1:], duration=self.durations, loop=0, optimize=True
        )


def render(result: RunResult, out: Path) -> None:
    t = Terminal()
    t.say(f"$ refiner run -f idea.md --iterate {len(result.iterations) or 1} --target 7", CYAN, 900)
    t.say(f"board: {result.board}   model: {result.model}", DIM, 600)
    t.say()
    t.say("idea: " + result.idea.strip().replace("\n", " ")[:260] + ("..." if len(result.idea) > 260 else ""), FG, 1600)
    t.say()
    for it in result.iterations or [None]:
        n = it.n if it else 1
        if it and n > 1:
            t.say()
            t.say(f"── revision {n} goes back to the board ──", MAGENTA, 900)
        for p in [p for p in result.phases if p.iteration == n]:
            title = TITLES[p.phase] + (f", round {p.round}" if p.round else "")
            t.say(f"▶ {title}", MAGENTA, 600)
            if p.phase == "synthesis":
                for line in (result.pitch or p.outputs[0].text).strip().split("\n"):
                    if line.strip().startswith("## Revised idea"):
                        break
                    if line.strip():
                        t.say("  " + line.strip(), GREEN, 420)
                continue
            for o in p.outputs:
                if o.verdict:
                    v = o.verdict
                    conf = f", {v.confidence} confidence" if v.confidence else ""
                    t.say(f"  {o.role:<30} {v.decision:<8} {v.score:>2}/10{conf}", COLOUR[v.decision], 420)
                    if v.issues:
                        t.say(f"      issue: {v.issues[0]}", DIM, 260)
                    if v.questions:
                        t.say(f"      asks:  {v.questions[0]}", DIM, 260)
                else:
                    first = o.text.strip().split("\n")[0]
                    t.say(f"  {o.role:<30} {first[:70]}", FG, 380)
            if p.tally:
                t.say(f"  board: {p.tally.as_text()}", CYAN, 1100)
            if p.chair and p.chair.action == "continue" and p.chair.questions:
                t.say("  chair: another round; questions to " + ", ".join(p.chair.questions), YELLOW, 1100)
            if p.closed:
                t.say(f"  deliberation closed: {p.closed}", DIM, 1000)
            t.say()
        if it and it.outcome:
            t.say(f"● {it.outcome}", CYAN, 1600)
            t.say()
    if result.verdict:
        t.say(f"final board verdict: {result.verdict.as_text()}", GREEN, 2000)
    t.say(f"done in {result.seconds:.0f}s · report saved to runs/{result.id}/report.md", DIM, 2500)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.save(out)
    print(f"{len(t.frames)} frames -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    run_dir, gif = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "docs/demo.gif")
    render(RunResult.model_validate_json((run_dir / "result.json").read_text(encoding="utf-8")), gif)

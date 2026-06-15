"""The `openworld` CLI — build, optimize, deploy world models.

Three phases, AutoML-style:
  openworld build  "<description>" --name foo   # Claude Code authors a spec
  openworld optimize specs/foo.json --goal ...  # Claude Code tunes it
  openworld serve  specs/ --allow-code          # FastAPI inference server

Plus helpers: `openworld ls <dir>` and `openworld card <spec>`.

Build/optimize drive Claude Code to author the spec: an interactive, watchable
tmux session when tmux is available, otherwise headless (`claude -p`). If the
`claude` CLI isn't installed, they print the scaffold + prompt to run manually.
This module is optional (click/rich/fastapi) and is never imported by the core.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import _tmux
from .card import render_card
from .spec import from_spec, spec_to_json, to_spec, validate_spec

console = Console()


def _load_spec(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gather(paths) -> list:
    """Expand paths/dirs into a list of (path, spec) pairs."""
    out = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            try:
                out.append((f, json.loads(f.read_text(encoding="utf-8"))))
            except Exception as e:
                console.print(f"[yellow]skip {f}: {e}[/yellow]")
    return out


@click.group(help="Build, optimize, and deploy world models.")
@click.version_option(package_name="openworld", message="openworld %(version)s")
def main():
    pass


# --------------------------------------------------------------------------- #
# deploy
# --------------------------------------------------------------------------- #
@main.command(help="Serve world specs as a FastAPI inference server.")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--allow-code/--no-allow-code", default=True, show_default=True,
              help="Run the worlds' dynamics (required to step/predict). "
                   "Only enable for specs you trust.")
def serve(paths, host, port, allow_code):
    import uvicorn

    from .serve import serve_app
    pairs = _gather(paths)
    if not pairs:
        raise click.ClickException("no specs found")
    specs = [s for _, s in pairs]
    if allow_code:
        console.print("[yellow]--allow-code: executing world dynamics from specs "
                      "(trust your inputs).[/yellow]")
    console.print(Panel.fit(
        "\n".join(f"• [bold]{s['name']}[/bold]  "
                  f"http://{host}:{port}/worlds/{s['name']}/view" for s in specs)
        + f"\n\nindex   http://{host}:{port}/\n"
        f"api     http://{host}:{port}/docs",
        title=f"OpenWorld serving {len(specs)} worlds", border_style="blue"))
    app = serve_app(specs, allow_code=allow_code)
    uvicorn.run(app, host=host, port=port)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
@main.command(help="List world specs in a directory or set of files.")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
def ls(paths):
    pairs = _gather(paths)
    table = Table(title=f"{len(pairs)} world specs")
    for col in ("name", "kind", "actions", "perception", "valid", "size"):
        table.add_column(col)
    for _, s in pairs:
        comp = s.get("composite")
        problems = validate_spec(s)
        table.add_row(
            s.get("name", "?"),
            "composite" if comp else "leaf",
            str(len(s.get("actions", []))),
            "yes" if s.get("perception") else "—",
            "[green]ok[/green]" if not problems else f"[red]{len(problems)}[/red]",
            f"{len(spec_to_json(s)) / 1024:.1f} KB")
    console.print(table)


@main.command(help="Render a world spec to an SVG model card.")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--out", type=click.Path(), default=None, help="output .svg path")
@click.option("--theme", default="light", type=click.Choice(["light", "dark"]))
@click.option("--open", "open_", is_flag=True, help="open the card after rendering")
def card(spec, out, theme, open_):
    s = _load_spec(spec)
    out = out or f"{s['name']}.svg"
    render_card(s, path=out, theme=theme)
    console.print(f"[green]wrote[/green] {out}")
    if open_:
        click.launch(out)


# --------------------------------------------------------------------------- #
# build / optimize (Claude Code in tmux)
# --------------------------------------------------------------------------- #
_BUILD_MD = """# Build an OpenWorld world model

**Goal:** {description}

Author the world in `build/{name}.py`, verify it runs, then run it to emit the
spec. Use the framework:

- `World(name, description, initial_state, actions, rules, transition)` with a
  `CodeTransition(code)` whose `code` defines `def transition(state, action): ...`
  (return the next state dict). Keep dynamics deterministic and verified.
- For multi-part worlds use `CompositeWorld(name, children=..., bridges=[Bridge(...)],
  aggregators=[Aggregator(...)], bindings=[Binding(...)])`.
- For input, attach perception: `world.perceptors = [CodePerceptor(code, produces,
  schema, modality="text")]` where `code` defines `def perceive(data): -> dict`.
  Add outputs with `world.emit = [{{"modality": "report", "fields": [...],
  "report": "..."}}]` and goals with `world.objectives = [{{"name":...,"goal":...}}]`.

The bottom of `build/{name}.py` already calls `to_spec` and writes
`specs/{name}.json`. When the spec is written and `validate_spec` is clean, you
are done.
"""

_SCAFFOLD = '''"""Build script for the `{name}` world. Fill in the world, then run:

    python build/{name}.py   # writes specs/{name}.json
"""
from pathlib import Path
from openworld import (World, CompositeWorld, CodeTransition, CodePerceptor,
                       Bridge, Aggregator, Binding, to_spec, validate_spec,
                       spec_to_json)

# --- TODO: build the world for: {description} ---
TRANSITION = """
def transition(state, action):
    s = dict(state)
    # ... apply action to produce the next state ...
    return s
"""

world = World(
    name="{name}",
    description="{description}",
    initial_state={{}},          # TODO
    actions=[],                   # TODO
    rules=[],                     # TODO (natural-language contract)
    transition=CodeTransition(TRANSITION),
)
# Optional: world.perceptors = [CodePerceptor(...)]; world.emit = [...]

spec = to_spec(world)
problems = validate_spec(spec)
assert not problems, problems
Path("specs").mkdir(exist_ok=True)
Path("specs/{name}.json").write_text(spec_to_json(spec))
print("wrote specs/{name}.json")
'''


def _scaffold(name: str, description: str) -> Path:
    Path("build").mkdir(exist_ok=True)
    (Path("build") / "BUILD.md").write_text(
        _BUILD_MD.format(name=name, description=description), encoding="utf-8")
    scaffold = Path("build") / f"{name}.py"
    if not scaffold.exists():
        scaffold.write_text(_SCAFFOLD.format(name=name, description=description),
                            encoding="utf-8")
    return scaffold


def _claude_phase(session: str, message: str, wait_for: Path, manual_hint: str):
    # No Claude Code at all -> manual mode.
    if not _tmux.claude_available():
        console.print(Panel(
            f"[yellow]claude not found — can't drive Claude Code automatically."
            f"[/yellow]\n\n{manual_hint}", title="manual mode", border_style="yellow"))
        return False
    # tmux present -> interactive, watchable session you can attach to.
    if _tmux.tmux_available():
        console.print(f"[blue]launching Claude Code in tmux session "
                      f"[bold]{session}[/bold]…[/blue]  (attach: tmux attach -t {session})")
        ok = _tmux.drive(session, Path.cwd(), message, wait_for, on_tail=lambda _t: None)
    else:
        # No tmux -> drive Claude Code headlessly (no multiplexer needed).
        console.print("[blue]no tmux — running Claude Code headlessly "
                      "(claude -p)…[/blue]")
        with console.status("[blue]Claude Code is authoring the spec…[/blue]"):
            ok, out = _tmux.claude_headless(message, Path.cwd(), wait_for)
        if out:
            console.print(Panel(out, title="claude output", border_style="dim"))
    if ok:
        console.print(f"[green]✓ {wait_for} produced[/green]")
    else:
        console.print(Panel(
            f"[yellow]Claude Code did not produce {wait_for}.[/yellow]\n\n{manual_hint}",
            title="finish manually", border_style="yellow"))
    return ok


@main.command(help="Build a world from a description, authored by Claude Code.")
@click.argument("description")
@click.option("--name", required=True, help="world name (slug)")
def build(description, name):
    scaffold = _scaffold(name, description)
    spec_path = Path("specs") / f"{name}.json"
    msg = (f"Read build/BUILD.md and follow it to build the '{name}' world in "
           f"{scaffold}, then run `python {scaffold}` to write {spec_path}. "
           f"The world is: {description}")
    hint = (f"1. Edit [bold]{scaffold}[/bold] (see build/BUILD.md)\n"
            f"2. Run [bold]python {scaffold}[/bold] to write {spec_path}\n"
            f"3. Or paste this to Claude Code:\n   {msg}")
    if _claude_phase(f"ow-build-{name}", msg, spec_path, hint):
        s = _load_spec(str(spec_path))
        problems = validate_spec(s)
        console.print(f"validate: {'[green]ok[/green]' if not problems else problems}")
        try:
            w1, w2 = from_spec(s, allow_code=True), from_spec(s, allow_code=True)
            console.print("[green]round-trip: reconstructable[/green]")
        except Exception as e:
            console.print(f"[yellow]round-trip note: {e}[/yellow]")
        console.print(f"next: [bold]openworld card {spec_path} --open[/bold]  |  "
                      f"[bold]openworld serve specs/ --allow-code[/bold]")


@main.command(help="Optimize a world spec toward a goal, via Claude Code.")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--goal", required=True, help="what to optimize for")
def optimize(spec, goal):
    s = _load_spec(spec)
    name = s.get("name", "world")
    out = Path(spec).with_suffix("").as_posix() + ".v2.json"
    msg = (f"Optimize the OpenWorld spec at {spec} toward this goal: {goal}. "
           f"Use the framework's tuners (openworld.tune Study/Tuner, "
           f"openworld.optimize.sweep, objectives) where useful. Load it with "
           f"from_spec(...,allow_code=True), iterate (propose -> run -> measure -> "
           f"keep best), and write the improved spec to {out}.")
    hint = (f"Paste this to Claude Code (or run your own loop):\n   {msg}\n"
            f"Then: [bold]openworld card {out} --open[/bold]")
    _claude_phase(f"ow-opt-{name}", msg, Path(out), hint)

"""Command line for redrawing figures from a saved data directory.

Run it as a module; nothing is trained::

    uv run --extra reports python -m faultline.viz reports/rooms9_autospec/data
    uv run --extra reports python -m faultline.viz DATA --scenario narrow_door --per-step
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from faultline.scenarios import resolve_scenario, scenario_names
from faultline.viz.render import FIGURE_KINDS, RenderOptions, render_directory
from faultline.viz.style import DEFAULT_FORMATS

DEFAULT_FIGURES_DIRNAME = "figures"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    How:
        Declares the data directory, the output directory, the scenario, seed and
        figure filters, and the overlay and caption switches.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description="Redraw FaultLine figures from saved run data.")
    parser.add_argument("data", type=Path, help="directory holding the saved run JSON files")
    parser.add_argument("--output", type=Path, help="figure directory (default: <data>/../figures)")
    parser.add_argument(
        "--scenario",
        nargs="+",
        help=(
            "only these scenarios: a 9-rooms name, a fixed n-rooms world, or "
            "rooms100_random_<seed>"
        ),
    )
    parser.add_argument("--seed", nargs="+", type=int, help="only these seeds; overlays draw each one")
    parser.add_argument("--figures", nargs="+", choices=FIGURE_KINDS, default=list(FIGURE_KINDS))
    parser.add_argument("--per-step", action="store_true", help="add one overlay panel per refinement step")
    parser.add_argument("--embed-caption", action="store_true", help="also print captions inside the figures")
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), help="image formats")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Load the data directory and write its figures.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    How:
        Builds the render options from the flags and hands the data directory to
        :func:`~faultline.viz.render.render_directory`.

    Returns:
        None: figures are written and a summary is printed.
    """
    args = build_parser().parse_args(argv)
    for name in args.scenario or ():
        try:
            resolve_scenario(name)
        except KeyError:
            known = ", ".join(scenario_names())
            raise SystemExit(
                f"Unknown scenario {name!r}. Known fixed names: {known}; "
                "random worlds use rooms100_random_<seed>."
            ) from None
    output = args.output if args.output is not None else args.data.parent / DEFAULT_FIGURES_DIRNAME
    options = RenderOptions(
        figures=args.figures,
        scenarios=args.scenario,
        seeds=args.seed,
        per_step=args.per_step,
        embed_caption=args.embed_caption,
        formats=tuple(args.formats),
    )
    render_directory(args.data, output, options)

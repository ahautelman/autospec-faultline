"""Command-line entry point for the 9-rooms reporting run.

Runs the checkpointed PPO pipeline over the five 9-rooms scenarios and writes the
raw series plus comparison figures. Run it as a module::

    uv run --extra interop --extra reports python -m faultline.experiments.rooms9_report

Training is expensive (200k steps per edge, five scenarios, five seeds by default),
so every global setting is a flag and ``--plot-only`` redraws the figures from the
persisted data without retraining. Defaults match the paper: a 200k budget for
every scenario, 1000 evaluation rollouts, and a checkpoint every 10k steps.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from faultline.experiments.figures import write_figures
from faultline.experiments.pipeline import default_workers, run_multi_seed, run_single_seed
from faultline.experiments.records import (
    ScenarioReport,
    ScenarioSpread,
    read_json,
    report_from_dict,
    report_to_dict,
    spread_from_dict,
    spread_to_dict,
    write_json,
)
from faultline.experiments.training import (
    DEFAULT_BUDGET,
    DEFAULT_INTERMEDIATE_ROLLOUTS,
    DEFAULT_INTERVAL,
    DEFAULT_ROLLOUTS,
    ReportSettings,
)
from faultline.scenarios import SCENARIOS, Scenario

DEFAULT_OUTPUT = Path("reports/rooms9")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
STEP1_SEED = 0
DATA_DIRNAME = "data"


@dataclass(frozen=True)
class ReportContext:
    """The scenarios, settings, and data directory a report run shares.

    Args:
        scenarios: The scenarios to run, in report order.
        settings: The global training settings.
        data: Directory the JSON records are read from and written to.
        workers: Worker-process count, or ``None`` to auto-size from the CPU count.

    How:
        Groups the values every step needs, so each step function stays within a
        few parameters and the path layout lives in one place.
    """

    scenarios: tuple[Scenario, ...]
    settings: ReportSettings
    data: Path
    workers: int | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    How:
        Declares the output directory, the three global training settings, the
        seed list, the worker-process count, which step to run, and the
        plot-only switch.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description="Run the 9-rooms reporting pipeline.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="figure and data directory")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="steps per edge")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS, help="evaluation rollouts")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="steps between checkpoints")
    parser.add_argument(
        "--intermediate-rollouts",
        type=int,
        default=DEFAULT_INTERMEDIATE_ROLLOUTS,
        help="evaluation rollouts for non-reference checkpoints",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="seeds per scenario")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel worker processes (default: FAULTLINE_WORKERS or cpu_count()-1; 1 is serial)",
    )
    parser.add_argument("--steps", choices=("1", "2", "both"), default="both", help="which step to run")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from existing data")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested steps and write the data and figures.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    How:
        Trains Step 1 (one seed per scenario) and Step 2 (several seeds) unless
        ``--plot-only``, persisting each record under ``<output>/data``. Then loads
        whatever data exists and writes every figure into ``<output>``.

    Returns:
        None: the run writes files and prints a short summary.
    """
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    context = ReportContext(
        scenarios=tuple(SCENARIOS.values()),
        settings=ReportSettings(
            budget=args.budget,
            rollouts=args.rollouts,
            interval=args.interval,
            intermediate_rollouts=args.intermediate_rollouts,
        ),
        data=output / DATA_DIRNAME,
        workers=args.workers,
    )
    resolved_workers = args.workers if args.workers is not None else default_workers()
    print(
        f"9-rooms report: {len(context.scenarios)} scenarios, budget={args.budget}, "
        f"reference/final rollouts={args.rollouts}, intermediate rollouts={args.intermediate_rollouts}, "
        f"interval={args.interval}, seeds={list(args.seeds)}, workers={resolved_workers}",
        flush=True,
    )
    for scenario in context.scenarios:
        print(
            f"  - {scenario.name}: {scenario.description} "
            f"(operator={scenario.expected_operator})",
            flush=True,
        )
    if args.plot_only:
        reports = load_reports(context)
        spreads = load_spreads(context)
    else:
        reports = train_step1(context, args.steps)
        spreads = train_step2(context, args.seeds, args.steps, reports)
    write_figures(reports, spreads, SCENARIOS, output)
    print(
        f"Wrote {len(reports)} reports, {len(spreads)} spreads, and figures to {output}",
        flush=True,
    )


def train_step1(context: ReportContext, steps: str) -> dict[str, ScenarioReport]:
    """Run Step 1 and persist the reports.

    Args:
        context: The scenarios, settings, and data directory.
        steps: Which step was requested; Step 1 is skipped for ``"2"``.

    How:
        Runs the single-seed pipeline once per scenario and writes
        ``step1_<scenario>.json`` for each.

    Returns:
        The reports keyed by scenario name, empty when Step 1 was skipped.
    """
    if steps == "2":
        return {}
    reports = run_single_seed(context.scenarios, STEP1_SEED, context.settings, context.workers)
    for name, report in reports.items():
        write_json(context.data / f"step1_{name}.json", report_to_dict(report))
    return reports


def train_step2(
    context: ReportContext,
    seeds: Sequence[int],
    steps: str,
    known: Mapping[str, ScenarioReport],
) -> dict[str, ScenarioSpread]:
    """Run Step 2 and persist the spreads.

    Args:
        context: The scenarios, settings, and data directory.
        seeds: The seeds to run per scenario.
        steps: Which step was requested; Step 2 is skipped for ``"1"``.
        known: Step 1 reports, reused for their seed so it is not retrained.

    How:
        Runs the multi-seed pipeline, reusing ``known`` where a seed matches, and
        writes ``step2_<scenario>.json`` for each.

    Returns:
        The spreads keyed by scenario name, empty when Step 2 was skipped.
    """
    if steps == "1":
        return {}
    reused = {name: {STEP1_SEED: report} for name, report in known.items()}
    spreads = run_multi_seed(context.scenarios, seeds, context.settings, reused or None, context.workers)
    for name, spread in spreads.items():
        write_json(context.data / f"step2_{name}.json", spread_to_dict(spread))
    return spreads


def load_reports(context: ReportContext) -> dict[str, ScenarioReport]:
    """Load the persisted Step 1 reports that exist.

    Args:
        context: The scenarios and data directory to read from.

    How:
        Reads ``step1_<scenario>.json`` when present and skips the rest, so a
        partially trained run still plots what it has.

    Returns:
        The reports keyed by scenario name.
    """
    reports: dict[str, ScenarioReport] = {}
    for scenario in context.scenarios:
        path = context.data / f"step1_{scenario.name}.json"
        if path.exists():
            reports[scenario.name] = report_from_dict(read_json(path))
    return reports


def load_spreads(context: ReportContext) -> dict[str, ScenarioSpread]:
    """Load the persisted Step 2 spreads that exist.

    Args:
        context: The scenarios and data directory to read from.

    How:
        Reads ``step2_<scenario>.json`` when present and skips the rest.

    Returns:
        The spreads keyed by scenario name.
    """
    spreads: dict[str, ScenarioSpread] = {}
    for scenario in context.scenarios:
        path = context.data / f"step2_{scenario.name}.json"
        if path.exists():
            spreads[scenario.name] = spread_from_dict(read_json(path))
    return spreads


if __name__ == "__main__":
    main()

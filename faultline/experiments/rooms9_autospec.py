"""Command-line entry point for the full Algorithm 1 run on the 9-rooms scenarios.

Where :mod:`faultline.experiments.rooms9_report` applies each scenario's expected
operator to one bottleneck edge, this runs the whole loop: train every edge, scan in
topological order, try seq / add / past / or on each weak edge, and commit the first
candidate that clears the threshold. Run it as a module::

    uv run --extra interop --extra reports python -m faultline.experiments.rooms9_autospec

Every ``(scenario, seed)`` run shares one process pool, one task per edge training, so
five scenarios times five seeds keep every CPU busy. Training uses the report's
checkpointed recipe and evaluation counts; ``--curves`` keeps the learning curves.
Figures are drawn into ``<output>/figures`` and can be redrawn with
``python -m faultline.viz <output>/data``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from faultline.experiments.autospec.executor import make_executor
from faultline.experiments.autospec.records import (
    AutoSpecReport,
    report_from_dict,
    report_to_dict,
)
from faultline.experiments.autospec.runner import Job, RunContext, run_jobs
from faultline.experiments.autospec.summary import summary_table
from faultline.experiments.pipeline import default_workers
from faultline.experiments.records import read_json, write_json
from faultline.experiments.training import (
    DEFAULT_BUDGET,
    DEFAULT_INTERMEDIATE_ROLLOUTS,
    DEFAULT_INTERVAL,
    DEFAULT_ROLLOUTS,
    ReportSettings,
)
from faultline.refinement.loop import AutoSpecConfig
from faultline.scenarios import SCENARIOS
from faultline.viz.render import RenderOptions, render_directory

DEFAULT_OUTPUT = Path("reports/rooms9_autospec")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DATA_DIRNAME = "data"
SUMMARY_FILENAME = "summary.md"
FIGURES_DIRNAME = "figures"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    How:
        Declares the output directory, the AutoSpec hyperparameters (defaulting to
        :class:`AutoSpecConfig`), the checkpointing options, the scenario and seed
        selection, the worker count, and the resume, plot-only, and verbosity
        switches.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    defaults = AutoSpecConfig()
    parser = argparse.ArgumentParser(description="Run the full AutoSpec loop on the 9-rooms scenarios.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="data directory (data/) and figure directory (figures/)")
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="success threshold p")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="steps per edge")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS, help="evaluation rollouts")
    parser.add_argument("--avoid-tail", type=int, default=defaults.avoid_tail_length, help="AvoidRefine tail length")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="seeds per scenario")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--curves", action="store_true", help="record learning curves (needed for the curve figures)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="steps between checkpoints (--curves)")
    parser.add_argument(
        "--intermediate-rollouts",
        type=int,
        default=DEFAULT_INTERMEDIATE_ROLLOUTS,
        help="evaluation rollouts for non-reference checkpoints (--curves)",
    )
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: cpu_count()-1; 1 is serial)")
    parser.add_argument("--resume", action="store_true", help="skip runs whose data file already exists")
    parser.add_argument("--plot-only", action="store_true", help="rebuild the table and figures from existing data")
    parser.add_argument("--verbose", action="store_true", help="show every worker checkpoint line")
    return parser


def build_context(args: argparse.Namespace, workers: int) -> RunContext:
    """Build the shared run context from the parsed arguments.

    Args:
        args: The parsed command line.
        workers: The resolved worker-process count.

    How:
        Builds the :class:`AutoSpecConfig` from the flags and derives the training
        settings from it, so budget and rollouts cannot disagree. Without
        ``--curves`` the checkpoint interval equals the budget, leaving only the
        final full-rollout evaluation.

    Returns:
        The context, owning a new executor the caller must shut down.
    """
    config = AutoSpecConfig(
        threshold=args.threshold,
        budget=args.budget,
        n_rollouts=args.rollouts,
        avoid_tail_length=args.avoid_tail,
    )
    settings = ReportSettings(
        budget=config.budget,
        rollouts=config.n_rollouts,
        interval=args.interval if args.curves else config.budget,
        intermediate_rollouts=args.intermediate_rollouts,
    )
    return RunContext(config, settings, make_executor(workers), quiet=not args.verbose)


def data_path(output: Path, scenario: str, seed: int) -> Path:
    """Return the JSON file holding one run's report.

    Args:
        output: The output directory.
        scenario: Scenario name.
        seed: Run seed.

    How:
        Joins the data subdirectory with a name built from the scenario and seed.

    Returns:
        The file path.
    """
    return output / DATA_DIRNAME / f"autospec_{scenario}_seed{seed}.json"


def select_jobs(args: argparse.Namespace, output: Path) -> list[Job]:
    """List the runs to execute, skipping finished ones under ``--resume``.

    Args:
        args: The parsed command line.
        output: The output directory holding the data files.

    How:
        Pairs each chosen scenario with each seed, scenario-major, and drops a pair
        whose data file exists when ``--resume`` is set.

    Returns:
        The ``(scenario, seed)`` jobs to run.
    """
    pairs = [(SCENARIOS[name], seed) for name in args.scenarios for seed in args.seeds]
    if not args.resume:
        return pairs
    return [(s, seed) for s, seed in pairs if not data_path(output, s.name, seed).exists()]


def load_reports(args: argparse.Namespace, output: Path) -> list[AutoSpecReport]:
    """Load every persisted report for the chosen scenarios and seeds.

    Args:
        args: The parsed command line.
        output: The output directory holding the data files.

    How:
        Reads each ``autospec_<scenario>_seed<k>.json`` that exists, in scenario
        then seed order, and ignores the rest.

    Returns:
        The reports found.
    """
    paths = [data_path(output, name, seed) for name in args.scenarios for seed in args.seeds]
    return [report_from_dict(read_json(path)) for path in paths if path.exists()]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the batch, then write the summary table and optional figures.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    How:
        Unless ``--plot-only``, runs the selected jobs through one shared pool and
        writes each report as it finishes. It then reloads all reports, writes
        ``summary.md``, and draws every figure the data supports into
        ``<output>/figures`` through :mod:`faultline.viz`; learning curves need
        ``--curves`` at run time and are skipped with a note otherwise.

    Returns:
        None: files are written and the summary is printed.
    """
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if not args.plot_only:
        workers = args.workers if args.workers is not None else default_workers()
        jobs = select_jobs(args, output)
        print(f"AutoSpec 9-rooms: {len(jobs)} runs, workers={workers}, curves={args.curves}", flush=True)
        context = build_context(args, workers)
        try:
            run_jobs(
                jobs,
                context,
                lambda report: write_json(
                    data_path(output, report.scenario, report.seed), report_to_dict(report)
                ),
            )
        finally:
            context.executor.shutdown(wait=True, cancel_futures=True)
    reports = load_reports(args, output)
    table = summary_table(reports)
    (output / SUMMARY_FILENAME).write_text(table + "\n", encoding="utf-8")
    print(table, flush=True)
    options = RenderOptions(scenarios=args.scenarios, seeds=args.seeds)
    render_directory(output / DATA_DIRNAME, output / FIGURES_DIRNAME, options)


if __name__ == "__main__":
    main()

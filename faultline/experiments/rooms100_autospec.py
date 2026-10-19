"""Run the full Algorithm 1 loop on n-rooms worlds.

This mirrors :mod:`faultline.experiments.rooms9_autospec` but resolves worlds
through the shared scenario resolver, so the same command handles the fixed
Fig. 3 / Fig. 6 layouts, the frozen generator outputs, and a paper-style random
world named ``rooms100_random_<seed>``. Every ``(world, seed)`` run trains its
edges on one shared process pool.

Run it as a module::

    uv run --extra interop --extra reports python -m faultline.experiments.rooms100_autospec \
        --worlds fig3 fig6 --seeds 0 1 2 3 4 --curves --budget 100000 --workers 4

The paper gives no reference scores for 100 rooms, so reports carry no
``reference_*`` numbers and the summary shows per-edge operator expectations
instead of a match score. Figures are written under
``<output>/figures`` and can be redrawn with
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
from faultline.experiments.rooms9_autospec import (
    DATA_DIRNAME,
    DEFAULT_SEEDS,
    FIGURES_DIRNAME,
    SUMMARY_FILENAME,
    data_path,
)
from faultline.experiments.training import (
    DEFAULT_INTERMEDIATE_ROLLOUTS,
    DEFAULT_INTERVAL,
    DEFAULT_ROLLOUTS,
    ReportSettings,
)
from faultline.refinement.loop import AutoSpecConfig
from faultline.scenarios import resolve_scenario, scenario_names
from faultline.viz.render import RenderOptions, render_directory

DEFAULT_OUTPUT = Path("reports/rooms100_autospec")
DEFAULT_WORLDS = ("fig3", "fig6")
DEFAULT_BUDGET = 100_000

SMOKE_WORLDS = ("fig3",)
SMOKE_SEEDS = (0,)
SMOKE_BUDGET = 1_000
SMOKE_ROLLOUTS = 20
SMOKE_INTERVAL = 250


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    How:
        Declares the world and seed selection, the AutoSpec hyperparameters, the
        checkpointing options, the worker count, and the resume, plot-only,
        verbose, and smoke switches. ``--worlds`` is not restricted to a fixed
        choice list because a random world is named ``rooms100_random_<seed>``.
        ``--smoke`` overrides the selection with one world, one seed, and a small
        budget so the pipeline can be exercised quickly.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    defaults = AutoSpecConfig()
    parser = argparse.ArgumentParser(description="Run the full AutoSpec loop on n-rooms worlds.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="data directory (data/) and figure directory (figures/)",
    )
    parser.add_argument(
        "--worlds",
        nargs="+",
        default=list(DEFAULT_WORLDS),
        help="fixed world names or rooms100_random_<seed>",
    )
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="success threshold p")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="steps per edge")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS, help="evaluation rollouts")
    parser.add_argument("--avoid-tail", type=int, default=defaults.avoid_tail_length, help="AvoidRefine tail length")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="seeds per world")
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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one world, one seed, and a small budget to check the pipeline",
    )
    return parser


def selection(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    """Return the effective worlds, seeds, and budget.

    Args:
        args: The parsed command line.

    How:
        Applies the ``--smoke`` overrides without mutating the parsed namespace.

    Returns:
        ``(worlds, seeds, budget)``.
    """
    if args.smoke:
        return SMOKE_WORLDS, SMOKE_SEEDS, SMOKE_BUDGET
    return tuple(args.worlds), tuple(args.seeds), int(args.budget)


def build_context(args: argparse.Namespace, workers: int, budget: int) -> RunContext:
    """Build the shared run context from the parsed arguments.

    Args:
        args: The parsed command line.
        workers: The resolved worker-process count.
        budget: The effective per-edge budget (the smoke mode lowers it).

    How:
        Builds the :class:`AutoSpecConfig` and derives the training settings from
        it, so budget and rollouts cannot disagree. Without ``--curves`` the
        checkpoint interval equals the budget, leaving only the final
        full-rollout evaluation. ``--smoke`` also lowers the rollout count, so a
        pipeline check does not spend the full paper evaluation budget.

    Returns:
        The context, owning a new executor the caller must shut down.
    """
    rollouts = min(int(args.rollouts), SMOKE_ROLLOUTS) if args.smoke else int(args.rollouts)
    config = AutoSpecConfig(
        threshold=args.threshold,
        budget=budget,
        n_rollouts=rollouts,
        avoid_tail_length=args.avoid_tail,
    )
    interval = args.interval if args.curves else budget
    settings = ReportSettings(
        budget=config.budget,
        rollouts=config.n_rollouts,
        interval=min(interval, budget, SMOKE_INTERVAL) if args.smoke else min(interval, budget),
        intermediate_rollouts=_intermediate_rollouts(args, rollouts),
    )
    return RunContext(config, settings, make_executor(workers), quiet=not args.verbose)


def _intermediate_rollouts(args: argparse.Namespace, rollouts: int) -> int:
    """Return the non-reference checkpoint rollout count.

    Args:
        args: The parsed command line.
        rollouts: The final evaluation rollout count.

    How:
        Smoke mode never asks for more intermediate rollouts than final ones.

    Returns:
        The intermediate rollout count.
    """
    if args.smoke:
        return min(int(args.intermediate_rollouts), rollouts)
    return int(args.intermediate_rollouts)


def select_jobs(
    worlds: Sequence[str], seeds: Sequence[int], output: Path, *, resume: bool
) -> list[Job]:
    """List the runs to execute, skipping finished ones under ``--resume``.

    Args:
        worlds: World names accepted by :func:`resolve_scenario`.
        seeds: Run seeds.
        output: The output directory holding the data files.
        resume: Whether to drop pairs whose data file already exists.

    How:
        Pairs each world with each seed, world-major, and resolves the scenario
        once per job. An unknown name raises through the resolver.

    Returns:
        The ``(scenario, seed)`` jobs to run.
    """
    pairs = [(world, seed) for world in worlds for seed in seeds]
    if resume:
        pairs = [
            (world, seed)
            for world, seed in pairs
            if not data_path(output, world, seed).exists()
        ]
    return [(resolve_scenario(world), seed) for world, seed in pairs]


def load_reports(
    worlds: Sequence[str], seeds: Sequence[int], output: Path
) -> list[AutoSpecReport]:
    """Load every persisted report for the chosen worlds and seeds.

    Args:
        worlds: World names.
        seeds: Run seeds.
        output: The output directory holding the data files.

    How:
        Reads each ``autospec_<world>_seed<k>.json`` that exists, in world then
        seed order, and ignores the rest.

    Returns:
        The reports found.
    """
    paths = [data_path(output, world, seed) for world in worlds for seed in seeds]
    return [report_from_dict(read_json(path)) for path in paths if path.exists()]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the batch, then write the summary table and optional figures.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    How:
        Unless ``--plot-only``, runs the selected jobs through one shared pool and
        writes each report as it finishes. It then reloads all reports, writes
        ``summary.md``, and draws every figure the data supports into
        ``<output>/figures`` through :mod:`faultline.viz`.

    Returns:
        None: files are written and the summary is printed.
    """
    args = build_parser().parse_args(argv)
    worlds, seeds, budget = selection(args)
    for world in worlds:
        try:
            resolve_scenario(world)
        except KeyError:
            known = ", ".join(scenario_names())
            raise SystemExit(
                f"Unknown world {world!r}. Known fixed names: {known}; "
                "random worlds use rooms100_random_<seed>."
            ) from None

    output = Path(args.output)
    if not args.plot_only:
        workers = args.workers if args.workers is not None else default_workers()
        jobs = select_jobs(worlds, seeds, output, resume=args.resume)
        estimated_trainings = len(jobs) * _edge_count(jobs)
        print(
            f"AutoSpec n-rooms: {len(jobs)} runs, ~{estimated_trainings}+ edge trainings, "
            f"workers={workers}, budget={budget}, curves={args.curves}",
            flush=True,
        )
        context = build_context(args, workers, budget)
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

    reports = load_reports(worlds, seeds, output)
    table = summary_table(reports)
    (output / SUMMARY_FILENAME).write_text(table + "\n", encoding="utf-8")
    print(table, flush=True)
    options = RenderOptions(scenarios=worlds, seeds=seeds)
    render_directory(output / DATA_DIRNAME, output / FIGURES_DIRNAME, options)


def _edge_count(jobs: Sequence[Job]) -> int:
    """Return the number of initial graph edges in the first job, or zero.

    Args:
        jobs: The ``(scenario, seed)`` jobs to run.

    How:
        Builds the first job's graph and counts its edges; every world shares
        the paper topology.

    Returns:
        The initial edge count, or zero with no jobs.
    """
    if not jobs:
        return 0
    scenario, _seed = jobs[0]
    return len(scenario.build_graph().edges)


if __name__ == "__main__":
    main()

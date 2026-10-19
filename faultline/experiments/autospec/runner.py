"""Run every ``(scenario, seed)`` Algorithm 1 loop concurrently over one shared pool.

Each loop run executes in a parent thread that mostly waits: every edge training it
needs is a task on the shared executor. All runs therefore feed the same pool, so the
CPUs stay busy across scenarios and seeds instead of finishing one scenario's seeds
before the next scenario starts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace

from faultline.experiments.autospec.curves import graph_curve, path_trajectories
from faultline.experiments.autospec.executor import SerialExecutor
from faultline.experiments.autospec.learner import PooledCheckpointLearner
from faultline.experiments.autospec.records import SCHEMA_VERSION, AutoSpecReport
from faultline.envs import layout_to_dict
from faultline.experiments.training import ReportSettings
from faultline.geometry import predicate_to_dict
from faultline.graph.snapshot import snapshot_graph
from faultline.refinement.loop import AutoSpecConfig
from faultline.scenarios import Scenario
from faultline.scenarios.runner import run_scenario

Job = tuple[Scenario, int]


@dataclass(frozen=True)
class RunContext:
    """What every run in a batch shares.

    Args:
        template: The AutoSpec settings; each run overrides the seed.
        settings: Training settings, consistent with ``template``'s budget and rollouts.
        executor: The shared executor edge trainings are submitted to.
        quiet: Whether workers swallow their per-checkpoint log lines.

    How:
        Groups the batch-wide values so each run function stays within a few
        parameters.
    """

    template: AutoSpecConfig
    settings: ReportSettings
    executor: Executor
    quiet: bool = True


def run_job(scenario: Scenario, seed: int, context: RunContext) -> AutoSpecReport:
    """Run the full loop for one scenario and seed.

    Args:
        scenario: The scenario to run.
        seed: The run seed.
        context: The shared configuration and executor.

    How:
        Prefetches the initial graph's edges so they train in parallel, runs
        :func:`~faultline.scenarios.runner.run_scenario` unchanged with the pooled
        learner, then reads the before and after curves from the stored trainings.

    Returns:
        The run's :class:`AutoSpecReport`, including the layout, graph snapshots and
        sampled trajectories the figures are redrawn from.
    """
    started = time.perf_counter()
    config = replace(context.template, seed=seed)
    learner = PooledCheckpointLearner(
        scenario.name, seed, context.settings, context.executor, context.quiet
    )
    initial_graph = scenario.build_graph()
    learner.prefetch(initial_graph.edges)
    result = run_scenario(scenario, learner, config)
    run = result.run
    return AutoSpecReport(
        scenario=scenario.name,
        seed=seed,
        expected_operator=scenario.expected_operator,
        config={"autospec": asdict(config), "training": asdict(context.settings)},
        steps=run.steps,
        initial_edges=run.initial_edge_success,
        final_edges=run.edge_success,
        best_path_before=result.best_path_before,
        best_path_after=result.best_path_after,
        reference_before=scenario.reference_before,
        reference_after=scenario.reference_after,
        before_curve=graph_curve(initial_graph, learner.trainings),
        after_curve=graph_curve(run.graph, learner.trainings),
        elapsed_seconds=time.perf_counter() - started,
        schema_version=SCHEMA_VERSION,
        layout=layout_to_dict(scenario.layout),
        unsafe_region=_unsafe_region_dict(scenario),
        initial_graph=snapshot_graph(initial_graph),
        final_graph=snapshot_graph(run.graph),
        trajectories_before=path_trajectories(initial_graph, learner.trainings),
        trajectories_after=path_trajectories(run.graph, learner.trainings),
        scenario_metadata={
            **dict(scenario.metadata),
            "expected_operators": {
                edge: list(operators)
                for edge, operators in scenario.expected_operators.items()
            },
        },
    )


def _unsafe_region_dict(scenario: Scenario) -> dict | None:
    """Serialise a scenario's hidden trap.

    Args:
        scenario: The scenario to read.

    How:
        Encodes the trap predicate when the scenario has one.

    Returns:
        The serialised region, or ``None`` for a scenario with no trap.
    """
    if scenario.unsafe_region is None:
        return None
    return predicate_to_dict(scenario.unsafe_region)


def run_jobs(
    jobs: Sequence[Job],
    context: RunContext,
    on_done: Callable[[AutoSpecReport], None],
) -> list[AutoSpecReport]:
    """Run all jobs, handing each finished report to ``on_done``.

    Args:
        jobs: The ``(scenario, seed)`` pairs to run.
        context: The shared configuration and executor.
        on_done: Called in the parent as each report completes, so a crash later
            in the batch does not lose finished runs.

    How:
        With a process pool, starts one thread per job so every run submits its
        edge trainings at once and the pool never waits on a single run's
        sequential steps. With the serial executor, runs the jobs one after
        another in this thread.

    Returns:
        The reports in completion order.
    """
    reports: list[AutoSpecReport] = []
    if not jobs:
        return reports
    if _is_serial(context.executor):
        for scenario, seed in jobs:
            reports.append(_finish(run_job(scenario, seed, context), on_done, len(reports) + 1, len(jobs)))
        return reports
    with ThreadPoolExecutor(max_workers=len(jobs)) as threads:
        futures = [threads.submit(run_job, scenario, seed, context) for scenario, seed in jobs]
        for future in as_completed(futures):
            reports.append(_finish(future.result(), on_done, len(reports) + 1, len(jobs)))
    return reports


def _is_serial(executor: Executor) -> bool:
    """Return whether the executor runs calls in the calling thread.

    Args:
        executor: The executor to inspect.

    How:
        Checks for :class:`SerialExecutor`.

    Returns:
        ``True`` for the serial executor.
    """
    return isinstance(executor, SerialExecutor)


def _finish(
    report: AutoSpecReport, on_done: Callable[[AutoSpecReport], None], completed: int, total: int
) -> AutoSpecReport:
    """Persist a finished report and log its completion.

    Args:
        report: The finished report.
        on_done: The persistence callback.
        completed: Runs finished so far, including this one.
        total: Runs in the batch.

    How:
        Calls ``on_done`` first so the data is on disk, then prints one flushed
        line.

    Returns:
        The same report, so callers can collect it inline.
    """
    on_done(report)
    print(
        f"[done] scenario={report.scenario} seed={report.seed} operators={list(report.operators)} "
        f"best_path {report.best_path_before:.3f}->{report.best_path_after:.3f} ({completed}/{total})",
        flush=True,
    )
    return report

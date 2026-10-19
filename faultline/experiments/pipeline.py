"""The 9-rooms reporting pipeline (paper sections 3, 8, and 11).

Two steps produce the data the figures read back:

- Step 1 trains one seed per scenario: the bottleneck edge, then the edge(s) the
  scenario's AutoSpec operator produces, each checkpointed over ``0..budget``.
- Step 2 repeats the pipeline for several seeds and keeps the before and after
  success at the reference and final checkpoints.

Both steps run their independent ``(scenario, seed)`` jobs concurrently in worker
processes by default, sized to ``os.cpu_count() - 1``. Each worker is pinned to a
single compute thread so the cores run distinct jobs instead of oversubscribed
internal threads. A serial run uses the same job function in one process, so the
process boundary is the only difference between the two.

The operator per scenario is the row of the section 11 table the scenario
isolates, applied through the same candidate builders the Algorithm 1 loop uses.
Pinning it keeps a run to one training per curve, which is what makes the full
``0..budget`` checkpoint series affordable; the loop's learned commit gate is
covered by the pytest suite.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

from faultline.experiments.overlays import build_overlay
from faultline.experiments.progress import disable as disable_progress
from faultline.experiments.records import (
    REFERENCE_STEP,
    Overlay,
    ScenarioReport,
    ScenarioSpread,
    SeedSummary,
    success_at,
)
from faultline.experiments.training import ReportSettings, ScenarioTrainer
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners import PPOEdgeLearner
from faultline.scenarios import SCENARIOS, Scenario

# Environment variable that overrides the auto-sized worker-process pool.
WORKERS_ENV = "FAULTLINE_WORKERS"


def source_edge(graph: AbstractGraph) -> Edge:
    """Return the bottleneck edge leaving the graph's source vertex.

    Args:
        graph: A scenario graph with its source set.

    How:
        Every scenario starts from one source vertex; this returns its single
        outgoing edge, the edge the report trains and refines.

    Returns:
        The source vertex's outgoing edge.

    Raises:
        ValueError: If the source is unset or does not have exactly one edge.
    """
    if graph.source is None:
        raise ValueError("Scenario graph has no source vertex")
    outgoing = graph.successors(graph.source)
    if len(outgoing) != 1:
        raise ValueError(f"Expected one source edge, found {len(outgoing)}")
    return outgoing[0]


def run_reported_scenario(scenario: Scenario, seed: int, settings: ReportSettings) -> ScenarioReport:
    """Train one scenario's before and after curves at a single seed.

    Args:
        scenario: The scenario to run.
        seed: Training seed.
        settings: The global training settings.

    How:
        Trains the bottleneck edge (the before curve), applies the scenario's
        AutoSpec operator to its final rollouts to get the refined route, then
        trains the route (the after curve). Both the unrefined policy and the
        refined route are then run end to end from the original source region and
        judged against the original edge, giving the specification-level
        satisfaction before and after. The overlay captures the operator's
        geometry from the same final rollouts, and it records whether the
        operator refused, so a zero after curve is readable. When nothing
        refines, the after curve repeats the before curve.

    Returns:
        The :class:`ScenarioReport` for the scenario and seed.
    """
    graph = scenario.build_graph()
    edge = scenario.report_edge(graph)
    operator = scenario.expected_operator
    print(f"\n=== scenario {scenario.name} (seed {seed}) ===", flush=True)
    print(f"    {scenario.description}", flush=True)
    print(
        f"    bottleneck edge {edge.source.name}->{edge.target.name}: "
        f"source={edge.source.region} target={edge.target.region} safe={edge.safe}",
        flush=True,
    )
    trainer = ScenarioTrainer(scenario.make_env(), PPOEdgeLearner(seed=seed), seed, settings)
    print("    stage 1/3: train bottleneck edge", flush=True)
    before = trainer.train(edge)
    overlay = build_overlay(operator, graph, edge, before.estimate) if operator else _empty_overlay()
    print(f"    stage 2/3: refine with {operator or 'none'}", flush=True)
    outcome = trainer.refined_series(graph, edge, before.estimate, operator)
    satisfaction_before = trainer.satisfaction(edge, (edge,), (before,))
    satisfaction_after = (
        trainer.satisfaction(edge, outcome.route, outcome.records)
        if outcome.route
        else satisfaction_before
    )
    print(
        f"    stage 3/3: scenario done, bottleneck final={before.estimate.success_probability:.3f} "
        f"refused={outcome.refused} | spec satisfaction before={satisfaction_before:.3f} "
        f"after={satisfaction_after:.3f}",
        flush=True,
    )
    return ScenarioReport(
        scenario=scenario.name,
        operator=operator or "none",
        seed=seed,
        budget=settings.budget,
        before=before.checkpoints,
        after=outcome.series or before.checkpoints,
        overlay=overlay,
        reference_before=scenario.reference_before,
        reference_after=scenario.reference_after,
        operator_refused=outcome.refused,
        satisfaction_before=satisfaction_before,
        satisfaction_after=satisfaction_after,
    )


def default_workers() -> int:
    """Return the worker-process count to use when the caller sets none.

    How:
        Reads :data:`WORKERS_ENV` when it holds a positive integer; otherwise
        returns ``os.cpu_count() - 1`` so one core stays for the OS and the
        orchestration loop. Never returns less than one.

    Returns:
        The number of worker processes.

    Raises:
        ValueError: If :data:`WORKERS_ENV` is set but is not a positive integer.
    """
    override = os.environ.get(WORKERS_ENV)
    if override is None:
        return max(1, (os.cpu_count() or 1) - 1)
    try:
        workers = int(override)
    except ValueError as err:
        raise ValueError(f"{WORKERS_ENV} must be an integer, got {override!r}") from err
    if workers < 1:
        raise ValueError(f"{WORKERS_ENV} must be positive, got {workers}")
    return workers


def configure_worker() -> None:
    """Pin a worker process to one compute thread and silence its progress bars.

    How:
        Sets the OpenMP, MKL, and OpenBLAS thread caps and limits PyTorch's own
        pool to a single thread, so N workers on an N-core machine run N distinct
        jobs instead of oversubscribing the cores with internal threads. Also
        disables tqdm, whose bars from several processes interleave unreadably.

    Returns:
        None: the environment and this process are configured in place.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    disable_progress()
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


def _resolve_workers(workers: int | None) -> int:
    """Return an explicit worker count, or the environment default.

    Args:
        workers: A caller-supplied count, or ``None`` to auto-size.

    How:
        Validates an explicit count and otherwise falls back to
        :func:`default_workers`.

    Returns:
        A positive worker-process count.

    Raises:
        ValueError: If ``workers`` is given but is not positive.
    """
    if workers is None:
        return default_workers()
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    return workers


def _run_job(job: tuple[str, int], settings: ReportSettings) -> ScenarioReport:
    """Run one ``(scenario, seed)`` job in the current process.

    Args:
        job: The scenario name and training seed to run.
        settings: The global training settings.

    How:
        Resolves the scenario through the catalogue, so only its name crosses the
        process boundary (a scenario's graph factory is not picklable), then runs
        the ordinary single-job pipeline.

    Returns:
        The scenario report for the job.
    """
    name, seed = job
    return run_reported_scenario(SCENARIOS[name], seed, settings)


def _log_done(desc: str, job: tuple[str, int], completed: int, total: int) -> None:
    """Print one completion line for a finished job.

    Args:
        desc: Label for the step being run.
        job: The finished job's scenario name and seed.
        completed: Number of jobs finished so far, including this one.
        total: Number of jobs submitted for the step.

    How:
        Writes a single flushed line, so several completions can interleave
        without corrupting each other's output.

    Returns:
        None: the line is printed.
    """
    name, seed = job
    print(f"[done] {desc} scenario={name} seed={seed} ({completed}/{total})", flush=True)


def _run_jobs(
    jobs: Sequence[tuple[str, int]], settings: ReportSettings, workers: int, desc: str
) -> dict[tuple[str, int], ScenarioReport]:
    """Run independent jobs concurrently when asked, sequentially otherwise.

    Args:
        jobs: The jobs to run, each a scenario name and seed.
        settings: The global training settings.
        workers: Worker-process count; ``1`` runs in-process and sequentially.
        desc: Label for the per-completion log line.

    How:
        With more than one worker, submits every job to a
        :class:`~concurrent.futures.ProcessPoolExecutor` sized to ``workers``,
        each worker pinned to one thread by :func:`configure_worker`, and logs
        each as it finishes. With one worker it runs the same job function
        in-process, so serial and concurrent differ only by the process boundary.
        Jobs share no state: each builds its own environment, policy, and
        trajectories, and the caller writes result files only after they return.

    Returns:
        The reports keyed by ``(scenario name, seed)``.
    """
    results: dict[tuple[str, int], ScenarioReport] = {}
    total = len(jobs)
    if workers <= 1:
        for completed, job in enumerate(jobs, start=1):
            results[job] = _run_job(job, settings)
            _log_done(desc, job, completed, total)
        return results
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_worker) as pool:
        futures = {pool.submit(_run_job, job, settings): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            results[job] = future.result()
            _log_done(desc, job, completed, total)
    return results


def run_single_seed(
    scenarios: Sequence[Scenario],
    seed: int,
    settings: ReportSettings,
    workers: int | None = None,
) -> dict[str, ScenarioReport]:
    """Run Step 1: one seed per scenario.

    Args:
        scenarios: The scenarios to run, in report order.
        seed: The training seed.
        settings: The global training settings.
        workers: Worker-process count; ``None`` uses :func:`default_workers`.

    How:
        Dispatches one ``(scenario, seed)`` job per scenario through
        :func:`_run_jobs`, so the independent scenarios run concurrently by
        default, and keys the results by scenario name.

    Returns:
        A mapping from scenario name to its report.
    """
    jobs = [(scenario.name, seed) for scenario in scenarios]
    results = _run_jobs(jobs, settings, _resolve_workers(workers), "step 1")
    return {name: results[(name, seed)] for name, _ in jobs}


def run_multi_seed(
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    settings: ReportSettings,
    known: Mapping[str, Mapping[int, ScenarioReport]] | None = None,
    workers: int | None = None,
) -> dict[str, ScenarioSpread]:
    """Run Step 2: the before and after success per seed.

    Args:
        scenarios: The scenarios to run.
        seeds: The seeds to run per scenario.
        settings: The global training settings.
        known: Reports already computed, keyed by scenario then seed, so Step 2
            can reuse Step 1's seed rather than retraining it.
        workers: Worker-process count; ``None`` uses :func:`default_workers`.

    How:
        Collects every ``(scenario, seed)`` pair missing from ``known``, dispatches
        them together through :func:`_run_jobs` so they run concurrently by
        default, then keeps the before and after success at the reference and
        final checkpoints per seed.

    Returns:
        A mapping from scenario name to its :class:`ScenarioSpread`.
    """
    existing = known or {}
    reports: dict[str, dict[int, ScenarioReport]] = {
        scenario.name: dict(existing.get(scenario.name, {})) for scenario in scenarios
    }
    jobs = [
        (scenario.name, seed)
        for scenario in scenarios
        for seed in seeds
        if seed not in reports[scenario.name]
    ]
    results = _run_jobs(jobs, settings, _resolve_workers(workers), "step 2")
    for (name, seed), report in results.items():
        reports[name][seed] = report
    return {
        scenario.name: _spread(scenario, reports[scenario.name], seeds, settings.budget)
        for scenario in scenarios
    }


def _spread(
    scenario: Scenario,
    reports: Mapping[int, ScenarioReport],
    seeds: Sequence[int],
    budget: int,
) -> ScenarioSpread:
    """Summarise the per-seed reports at the two reported checkpoints.

    Args:
        scenario: The scenario being summarised.
        reports: The per-seed reports to read, keyed by seed.
        seeds: The seeds to summarise, in order.
        budget: The training budget, the final checkpoint to read.

    How:
        For each seed, reads the before and after series at the reference step and
        at the budget, giving the four numbers the spread figure aggregates.

    Returns:
        The scenario's :class:`ScenarioSpread`.
    """
    summaries = []
    for seed in seeds:
        report = reports[seed]
        summaries.append(
            SeedSummary(
                seed=seed,
                before_80k=success_at(report.before, REFERENCE_STEP),
                after_80k=success_at(report.after, REFERENCE_STEP),
                before_final=success_at(report.before, budget),
                after_final=success_at(report.after, budget),
            )
        )
    return ScenarioSpread(scenario.name, budget, tuple(summaries))


def _empty_overlay() -> Overlay:
    """Return an overlay with no geometry, for a scenario without an operator.

    How:
        Builds an ``edge``-kind overlay with no points, boundary, or added edge,
        which the figure layer renders as the bare regions.

    Returns:
        The empty overlay.
    """
    return Overlay("edge", (), (), None, None, None)

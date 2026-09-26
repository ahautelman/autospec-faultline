"""The 9-rooms reporting pipeline (paper sections 3, 8, and 11).

Two steps produce the data the figures read back:

- Step 1 trains one seed per scenario: the bottleneck edge, then the edge(s) the
  scenario's AutoSpec operator produces, each checkpointed over ``0..budget``.
- Step 2 repeats the pipeline for several seeds and keeps the before and after
  success at the reference and final checkpoints.

The operator per scenario is the row of the section 11 table the scenario
isolates, applied through the same candidate builders the Algorithm 1 loop uses.
Pinning it keeps a run to one training per curve, which is what makes the full
``0..budget`` checkpoint series affordable; the loop's learned commit gate is
covered by the pytest suite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from faultline.envs import Rooms9
from faultline.experiments.overlays import build_overlay
from faultline.experiments.records import (
    REFERENCE_STEP,
    Checkpoint,
    Overlay,
    ScenarioReport,
    ScenarioSpread,
    SeedSummary,
    success_at,
)
from faultline.experiments.training import ReportSettings, TrainingRecord, train_with_checkpoints
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners import CannotRefineError, EdgeEstimate, PPOEdgeLearner
from faultline.refinement.candidates import (
    add_candidate,
    or_candidate,
    past_candidate,
    seq_candidate,
)
from faultline.refinement.operators import AVOID_TAIL_LENGTH
from faultline.scenarios import Scenario

_CANDIDATE_BUILDERS = {
    "seq": seq_candidate,
    "add": add_candidate,
    "past": past_candidate,
    "or": or_candidate,
}


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


class ScenarioTrainer:
    """Trains a scenario's edges with checkpoints.

    Args:
        env: The training environment.
        learner: The PPO learner the report drives.
        seed: Training seed for this run.
        settings: The global budget, rollout count, and checkpoint interval.

    How:
        Holds the training context so the pipeline can train any edge, and the
        refined route, without threading the settings through every call.
    """

    def __init__(
        self, env: Rooms9, learner: PPOEdgeLearner, seed: int, settings: ReportSettings
    ) -> None:
        self.env = env
        self.learner = learner
        self.seed = seed
        self.settings = settings

    def train(self, edge: Edge) -> TrainingRecord:
        """Train one edge with checkpoints.

        Args:
            edge: The reach-avoid edge to train.

        How:
            Delegates to :func:`~faultline.experiments.training.train_with_checkpoints`
            with this run's seed and settings.

        Returns:
            The edge's :class:`TrainingRecord`.
        """
        return train_with_checkpoints(self.learner, self.env, edge, self.seed, self.settings)

    def refined_series(
        self, graph: AbstractGraph, edge: Edge, estimate: EdgeEstimate, operator: str | None
    ) -> tuple[Checkpoint, ...]:
        """Train the refined route and return its product series.

        Args:
            graph: The graph the operator runs on.
            edge: The bottleneck edge being refined.
            estimate: The edge's final estimate, supplying the rollouts.
            operator: The operator name, or ``None`` to skip refinement.

        How:
            Builds the operator's candidate through the same builder the loop
            uses. An operator that refuses for lack of data yields an empty
            series, so the caller keeps the before curve. Otherwise every route
            edge is trained with checkpoints and the route product is returned.

        Returns:
            The refined route's success series, or ``()`` when nothing refined.
        """
        if operator is None:
            return ()
        try:
            candidate = _CANDIDATE_BUILDERS[operator](graph, edge, estimate, AVOID_TAIL_LENGTH)
        except CannotRefineError:
            return ()
        records = [self.train(route_edge) for route_edge in candidate.routes[0]]
        return route_product(records)


def run_reported_scenario(scenario: Scenario, seed: int, settings: ReportSettings) -> ScenarioReport:
    """Train one scenario's before and after curves at a single seed.

    Args:
        scenario: The scenario to run.
        seed: Training seed.
        settings: The global training settings.

    How:
        Trains the bottleneck edge (the before curve), applies the scenario's
        AutoSpec operator to its final rollouts to get the refined route, then
        trains the route (the after curve). The overlay captures the operator's
        geometry from the same final rollouts. When nothing refines, the after
        curve repeats the before curve.

    Returns:
        The :class:`ScenarioReport` for the scenario and seed.
    """
    graph = scenario.build_graph()
    trainer = ScenarioTrainer(scenario.make_env(), PPOEdgeLearner(seed=seed), seed, settings)
    edge = source_edge(graph)
    operator = scenario.expected_operator
    before = trainer.train(edge)
    overlay = build_overlay(operator, graph, edge, before.estimate) if operator else _empty_overlay()
    after = trainer.refined_series(graph, edge, before.estimate, operator)
    return ScenarioReport(
        scenario=scenario.name,
        operator=operator or "none",
        seed=seed,
        budget=settings.budget,
        before=before.checkpoints,
        after=after or before.checkpoints,
        overlay=overlay,
        reference_before=scenario.reference_before,
        reference_after=scenario.reference_after,
    )


def run_single_seed(
    scenarios: Sequence[Scenario], seed: int, settings: ReportSettings
) -> dict[str, ScenarioReport]:
    """Run Step 1: one seed per scenario.

    Args:
        scenarios: The scenarios to run, in report order.
        seed: The training seed.
        settings: The global training settings.

    How:
        Runs :func:`run_reported_scenario` for each scenario and keys the result
        by scenario name.

    Returns:
        A mapping from scenario name to its report.
    """
    return {scenario.name: run_reported_scenario(scenario, seed, settings) for scenario in scenarios}


def run_multi_seed(
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    settings: ReportSettings,
    known: Mapping[str, Mapping[int, ScenarioReport]] | None = None,
) -> dict[str, ScenarioSpread]:
    """Run Step 2: the before and after success per seed.

    Args:
        scenarios: The scenarios to run.
        seeds: The seeds to run per scenario.
        settings: The global training settings.
        known: Reports already computed, keyed by scenario then seed, so Step 2
            can reuse Step 1's seed rather than retraining it.

    How:
        For each scenario, runs any seed missing from ``known`` and then keeps the
        before and after success at the reference and final checkpoints per seed.

    Returns:
        A mapping from scenario name to its :class:`ScenarioSpread`.
    """
    existing = known or {}
    spreads: dict[str, ScenarioSpread] = {}
    for scenario in scenarios:
        reports = dict(existing.get(scenario.name, {}))
        for seed in seeds:
            if seed not in reports:
                reports[seed] = run_reported_scenario(scenario, seed, settings)
        spreads[scenario.name] = _spread(scenario, reports, seeds, settings.budget)
    return spreads


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


def route_product(records: Sequence[TrainingRecord]) -> tuple[Checkpoint, ...]:
    """Multiply the route edges' series stepwise.

    Args:
        records: The trained route edges, in route order.

    How:
        Aligns the series on their shared checkpoint steps and multiplies the
        per-edge probabilities at each step, matching the loop's best-route
        product for a two-hop route. A single-edge route is returned unchanged.

    Returns:
        The route's success series.
    """
    if len(records) == 1:
        return records[0].checkpoints
    tables = [{checkpoint.step: checkpoint.probability for checkpoint in record.checkpoints} for record in records]
    common = sorted(set.intersection(*(set(table) for table in tables)))
    return tuple(
        Checkpoint(step, math.prod(table[step] for table in tables)) for step in common
    )


def _empty_overlay() -> Overlay:
    """Return an overlay with no geometry, for a scenario without an operator.

    How:
        Builds an ``edge``-kind overlay with no points, boundary, or added edge,
        which the figure layer renders as the bare regions.

    Returns:
        The empty overlay.
    """
    return Overlay("edge", (), (), None, None, None)

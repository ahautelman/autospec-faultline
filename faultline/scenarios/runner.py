"""Run one 9-rooms scenario through the Algorithm 1 loop (paper sections 3 and 11).

The runner is the thin layer between a :class:`~faultline.scenarios.catalogue.Scenario`
and :func:`~faultline.refinement.loop.run_autospec`. It builds a fresh graph, runs
the loop with a caller-chosen learner, and turns the run record into the per-edge
before and after success and the committed operators the validation suite reports.

Nothing here is learner-specific: the tabular planner and the PPO learner are
interchangeable, because the loop already talks to both through the learner
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.graph.semantics import best_path_product
from faultline.refinement.loop import (
    AutoSpecConfig,
    AutoSpecRun,
    EdgeSuccess,
    RefinementStep,
    run_autospec,
)
from faultline.scenarios.catalogue import Scenario


@dataclass(frozen=True)
class EdgeOutcome:
    """Before and after success for one edge of the pre-refinement graph.

    Args:
        source: Source vertex name.
        target: Target vertex name.
        before: The edge's success estimate before refinement.
        after: The committed candidate's score when the edge was refined, or the
            unchanged estimate when it was not.

    How:
        Matches a pre-refinement edge to its refinement decision by endpoint names,
        which survive the operators' replacement edges.
    """

    source: str
    target: str
    before: float
    after: float


@dataclass(frozen=True)
class ScenarioResult:
    """The outcome of one scenario run.

    Args:
        scenario: The scenario name.
        operators: The operator committed per weak edge examined, in scan order;
            ``None`` marks a weak edge no operator could improve.
        edges: Before and after success per pre-refinement edge, in graph order.
        best_path_before: Whole-graph success before refinement.
        best_path_after: Whole-graph success after refinement.
        run: The underlying run record, for a caller that needs the repaired graph
            or the per-edge final estimates.

    How:
        Pairs the loop's run record with the whole-graph success before refinement,
        which the loop itself does not report.
    """

    scenario: str
    operators: tuple[str | None, ...]
    edges: tuple[EdgeOutcome, ...]
    best_path_before: float
    best_path_after: float
    run: AutoSpecRun


def run_scenario(
    scenario: Scenario, learner: Any, config: AutoSpecConfig | None = None
) -> ScenarioResult:
    """Run one scenario through the Algorithm 1 loop.

    Args:
        scenario: The scenario to run; supplies the layout and the graph factory.
        learner: The spec-guided RL backend that trains and estimates edges.
        config: Run configuration; defaults to :class:`AutoSpecConfig`, so the
            paper's threshold, budget, and rollout count apply.

    How:
        Builds a fresh graph, runs the loop over a fresh environment, then maps the
        run's initial per-edge estimates back onto the pre-refinement graph to
        compute the before best-path product. Edges are matched to refinement
        decisions by endpoint names.

    Returns:
        A :class:`ScenarioResult` with the per-edge before and after success, the
        committed operators, and the whole-graph success before and after.

    Raises:
        ValueError: If the scenario graph's source or target has not been set.
    """
    graph = scenario.build_graph()
    active_config = config if config is not None else AutoSpecConfig()
    run = run_autospec(graph, scenario.make_env(), learner, active_config)
    steps = {(step.source, step.target): step for step in run.steps}
    probabilities = _initial_probabilities(graph, run.initial_edge_success)
    edges = tuple(
        EdgeOutcome(
            outcome.source,
            outcome.target,
            outcome.probability,
            _after_for(outcome, steps),
        )
        for outcome in run.initial_edge_success
    )
    return ScenarioResult(
        scenario=scenario.name,
        operators=tuple(step.operator for step in run.steps),
        edges=edges,
        best_path_before=best_path_product(graph, probabilities),
        best_path_after=run.best_path_success,
        run=run,
    )


def _initial_probabilities(
    graph: AbstractGraph, initial: tuple[EdgeSuccess, ...]
) -> dict[Edge, float]:
    """Map the run's initial per-edge estimates back onto the graph's edges.

    Args:
        graph: The pre-refinement graph the run started from.
        initial: The run's initial per-edge estimates, in graph edge order.

    How:
        The loop copies the graph but preserves edge order, so the estimates line
        up positionally with ``graph.edges``; zipping pairs each edge with its
        estimate for the before best-path product.

    Returns:
        A mapping from each pre-refinement edge to its success estimate.

    Raises:
        ValueError: If the estimate count does not match the graph's edge count,
            which would mean the loop reordered or dropped edges.
    """
    if len(initial) != len(graph.edges):
        raise ValueError(f"Got {len(initial)} initial estimates for {len(graph.edges)} edges")
    paired = zip(graph.edges, initial, strict=True)
    return {edge: estimate.probability for edge, estimate in paired}


def _after_for(outcome: EdgeSuccess, steps: dict[tuple[str, str], RefinementStep]) -> float:
    """Return the after success for one pre-refinement edge.

    Args:
        outcome: The edge's initial estimate.
        steps: Refinement decisions keyed by ``(source, target)`` names.

    How:
        Returns the committed candidate's score when a decision matches the edge's
        endpoint names, otherwise the unchanged initial estimate.

    Returns:
        The edge's success estimate after refinement.
    """
    step = steps.get((outcome.source, outcome.target))
    return step.after if step is not None else outcome.probability

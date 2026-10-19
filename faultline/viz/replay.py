"""Rebuild the refinement of a toy report so its overlay can be drawn.

A toy ``rooms9_report`` run saves only what the operator consumed (sample points and, for
a split, the fitted half-plane), not the refined graph. Replaying the real operator on
stand-in rollouts built from those points reproduces the same graph, and the usual trace
is then taken from it. The figure therefore shows exactly what a full-loop run would, and
no operator geometry is re-derived here.
"""

from __future__ import annotations

import numpy as np

from faultline.experiments.records import Overlay, ScenarioReport
from faultline.graph.snapshot import GraphSnapshot, snapshot_graph
from faultline.learners import CannotRefineError, EdgeEstimate, Rollout
from faultline.refinement.candidates import add_candidate, or_candidate, past_candidate, seq_candidate
from faultline.refinement.loop import RefinementStep
from faultline.refinement.operators import AVOID_TAIL_LENGTH
from faultline.refinement.trace import build_trace
from faultline.refinement.trajectories import StepSamples, thin_points
from faultline.scenarios import Scenario

CANDIDATES = {
    "seq": seq_candidate,
    "add": add_candidate,
    "past": past_candidate,
    "or": or_candidate,
}


def replay_refinement(
    report: ScenarioReport, scenario: Scenario
) -> tuple[RefinementStep, GraphSnapshot] | None:
    """Replay a toy report's operator and describe the edit it made.

    Args:
        report: The toy report, whose overlay holds the operator's inputs.
        scenario: The scenario it ran, supplying the graph and bottleneck edge.

    How:
        Builds rollouts that give the operator the same inputs it saw, applies the
        real candidate builder to the scenario's graph, and diffs the graphs before
        and after into a trace that also carries the saved sample points.

    Returns:
        The step with its trace and the refined graph's snapshot, or ``None`` when
        nothing was applied or the operator cannot reproduce the edit from the saved
        points (an older report, for example).
    """
    build = CANDIDATES.get(report.operator)
    if build is None or report.operator_refused:
        return None
    graph = scenario.build_graph()
    edge = scenario.report_edge(graph)
    estimate = _stand_in_estimate(report.operator, report.overlay)
    try:
        candidate = build(graph, edge, estimate, AVOID_TAIL_LENGTH)
    except CannotRefineError:
        return None
    samples = StepSamples(
        thin_points(report.overlay.points), thin_points(report.overlay.avoid_points), ()
    )
    before, after = snapshot_graph(graph), snapshot_graph(candidate.graph)
    first = report.before[-1].probability if report.before else 0.0
    last = report.after[-1].probability if report.after else 0.0
    step = RefinementStep(
        edge.source.name, edge.target.name, (report.operator,), report.operator, first, last,
        build_trace(before, after, samples),
    )
    return step, after


def _stand_in_estimate(operator: str, overlay: Overlay) -> EdgeEstimate:
    """Build rollouts that give ``operator`` the inputs the overlay recorded.

    Args:
        operator: The operator key.
        overlay: The saved operator inputs.

    How:
        Each operator reads one slice of the rollouts, so each is given exactly that
        slice: ReachRefine the visited states of successes, AvoidRefine the unsafe
        tails (one single-state failure per tail state, so any tail length keeps it),
        AddRefine the midpoints, PastRefine the success and failure starts. OrRefine
        reads no rollouts.

    Returns:
        The stand-in estimate.
    """
    if operator == "seq":
        # An episode with no states would break the operators, so empty sets add none.
        rollouts = [_rollout(overlay.points, True)] if overlay.points else []
        rollouts += [_rollout([point], False) for point in overlay.avoid_points]
    elif operator == "add":
        rollouts = [_rollout([point] * 3, True) for point in overlay.points]
    elif operator == "past":
        rollouts = [_rollout([point], True) for point in overlay.points]
        rollouts += [_rollout([point], False) for point in overlay.counter_points]
    else:
        rollouts = []
    return EdgeEstimate(0.0, tuple(rollouts))


def _rollout(points: list[tuple[float, float]], success: bool) -> Rollout:
    """Build a rollout through the given points, successful or ended unsafe."""
    states = tuple(np.array(point, dtype=float) for point in points)
    return Rollout(states, reached=success, safe=success, final_safe=success)

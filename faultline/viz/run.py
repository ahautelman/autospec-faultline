"""The one record every figure is drawn from.

A figure never reads a report type or a live scenario directly: it gets a ``RunView``,
a plain bundle of the room, the graphs, the refinement steps and the sampled rollouts.
Adapters build it from whatever a run was saved as, so a new kind of run needs one new
adapter and no figure changes. A report written before figure data was recorded is
completed from the scenario catalogue where possible.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from faultline.envs import RoomsLayout, layout_from_dict
from faultline.experiments.autospec.records import AutoSpecReport
from faultline.experiments.records import REFERENCE_STEP, Checkpoint, ScenarioReport, ScenarioSpread
from faultline.geometry import predicate_from_dict
from faultline.graph.abstract_graph import AbstractGraph
from faultline.graph.snapshot import GraphSnapshot, snapshot_graph
from faultline.refinement.loop import RefinementStep
from faultline.refinement.trajectories import TrajectorySample
from faultline.scenarios import SCENARIOS, Scenario, resolve_scenario
from faultline.spectrl.predicates import Predicate
from faultline.viz.replay import replay_refinement


@dataclass(frozen=True)
class RunView:
    """What the figures need to know about one run.

    Args:
        scenario: Scenario name.
        seed: Run seed.
        layout: The room's walls and doors.
        unsafe_region: The hidden trap, or ``None``.
        initial_graph: The graph before any refinement.
        final_graph: The repaired graph, or ``None`` when it was not recorded.
        steps: The refinement steps in scan order.
        trajectories_before: Sampled rollouts of the initial best path.
        trajectories_after: Sampled rollouts of the repaired best path.
        before_curve: Best-path learning curve of the initial graph.
        after_curve: Best-path learning curve of the repaired graph.
        reference_before: The paper's before number, or ``None`` when the run has none.
        reference_after: The paper's after number, or ``None``.
        threshold: The success threshold the loop used, or ``None`` when unknown.
        budget: Training steps per edge, or ``None`` when unknown.

    How:
        Holds plain data only, so a view can be built from a file and drawn with no
        training or catalogue lookup. Everything after ``final_graph`` defaults to
        empty, so a view that only needs the room is cheap to build.
    """

    scenario: str
    seed: int
    layout: RoomsLayout
    unsafe_region: Predicate | None
    initial_graph: GraphSnapshot
    final_graph: GraphSnapshot | None
    steps: tuple[RefinementStep, ...]
    trajectories_before: tuple[TrajectorySample, ...] = ()
    trajectories_after: tuple[TrajectorySample, ...] = ()
    before_curve: tuple[Checkpoint, ...] = ()
    after_curve: tuple[Checkpoint, ...] = ()
    reference_before: float | None = None
    reference_after: float | None = None
    threshold: float | None = None
    budget: int | None = None

    @property
    def committed(self) -> tuple[tuple[int, RefinementStep], ...]:
        """The steps that changed the graph and recorded how.

        How:
            Keeps each step's scan index so a figure can number it as the run did.

        Returns:
            ``(index, step)`` pairs for steps with an operator and a trace.
        """
        return tuple(
            (index, step)
            for index, step in enumerate(self.steps)
            if step.operator is not None and step.trace is not None
        )

    @property
    def world(self) -> float:
        """Side length of the square world."""
        return self.layout.world_size


def run_view_from_report(
    report: AutoSpecReport, catalogue: Mapping[str, Scenario] = SCENARIOS
) -> RunView:
    """Build a view from a full-loop report.

    Args:
        report: The report, of any schema version.
        catalogue: Scenarios to fall back on when the report lacks the room.

    How:
        Reads the stored layout, trap and graph snapshots. A report from before they
        were recorded takes its layout, trap and initial graph from the catalogue and
        has no final graph, so its figures show the room without annotations.

    Returns:
        The view.

    Raises:
        ValueError: If the report has no recorded room and its scenario is not in
            the catalogue.
    """
    if report.layout is None or report.initial_graph is None:
        scenario = _catalogue_scenario(report, catalogue)
        return RunView(
            scenario=report.scenario,
            seed=report.seed,
            layout=scenario.layout,
            unsafe_region=scenario.unsafe_region,
            initial_graph=snapshot_graph(scenario.build_graph()),
            final_graph=None,
            steps=report.steps,
            **_curve_fields(report),
        )
    unsafe = report.unsafe_region
    return RunView(
        scenario=report.scenario,
        seed=report.seed,
        layout=layout_from_dict(report.layout),
        unsafe_region=predicate_from_dict(unsafe) if unsafe is not None else None,
        initial_graph=report.initial_graph,
        final_graph=report.final_graph,
        steps=report.steps,
        trajectories_before=report.trajectories_before,
        trajectories_after=report.trajectories_after,
        **_curve_fields(report),
    )


def run_view_from_scenario_report(report: ScenarioReport, scenario: Scenario) -> RunView:
    """Build a view from the toy single-bottleneck report.

    Args:
        report: The toy report.
        scenario: The scenario it ran, which supplies the room and the graph.

    How:
        Takes the curves and paper references from the report and the room from the
        scenario. The operator is replayed on the saved sample points to rebuild the
        refined graph and its trace, so the overlay can be drawn. If the replay is not
        possible the view keeps one geometry-less step naming the operator, which still
        gives the figures a title and caption.

    Returns:
        The view.
    """
    graph = scenario.build_graph()
    replayed = replay_refinement(report, scenario)
    step, final = replayed if replayed is not None else (_plain_step(report, scenario, graph), None)
    return RunView(
        scenario=report.scenario,
        seed=report.seed,
        layout=scenario.layout,
        unsafe_region=scenario.unsafe_region,
        initial_graph=snapshot_graph(graph),
        final_graph=final,
        steps=(step,) if report.operator else (),
        before_curve=report.before,
        after_curve=report.after,
        reference_before=report.reference_before,
        reference_after=report.reference_after,
        budget=report.budget,
    )


def _plain_step(report: ScenarioReport, scenario: Scenario, graph: AbstractGraph) -> RefinementStep:
    """Describe a toy report's operator without geometry, when it cannot be replayed."""
    edge = scenario.report_edge(graph)
    applied = report.operator if report.operator and not report.operator_refused else None
    return RefinementStep(
        edge.source.name, edge.target.name, (report.operator,), applied,
        report.before[-1].probability if report.before else 0.0,
        report.after[-1].probability if report.after else 0.0,
    )


def run_views_from_spread(spread: ScenarioSpread, scenario: Scenario) -> list[RunView]:
    """Build one view per seed from the toy report's multi-seed summary.

    Args:
        spread: The per-seed summaries of one scenario.
        scenario: The scenario they came from, supplying the room and paper numbers.

    How:
        The summary keeps each seed's success at the paper's reference step and at the
        final budget, so each seed's curve is those exact points. The toy run applies
        the scenario's expected operator, which names the single step. These views suit
        the per-seed and paper-comparison figures, not a learning curve.

    Returns:
        One view per seed, in the spread's order.
    """
    graph = snapshot_graph(scenario.build_graph())
    operator = scenario.expected_operator
    steps = (RefinementStep("", "", (operator,), operator, 0.0, 0.0),) if operator else ()
    return [
        RunView(
            scenario=spread.scenario,
            seed=summary.seed,
            layout=scenario.layout,
            unsafe_region=scenario.unsafe_region,
            initial_graph=graph,
            final_graph=None,
            steps=steps,
            before_curve=_two_points(spread.budget, summary.before_80k, summary.before_final),
            after_curve=_two_points(spread.budget, summary.after_80k, summary.after_final),
            reference_before=scenario.reference_before,
            reference_after=scenario.reference_after,
            budget=spread.budget,
        )
        for summary in spread.summaries
    ]


def _two_points(budget: int, at_reference: float, at_final: float) -> tuple[Checkpoint, ...]:
    """Return the reference-step and final points, or just the final one if they coincide."""
    if budget <= REFERENCE_STEP:
        return (Checkpoint(budget, at_final),)
    return (Checkpoint(REFERENCE_STEP, at_reference), Checkpoint(budget, at_final))


def group_by_scenario(runs: Iterable[RunView]) -> dict[str, list[RunView]]:
    """Group runs by scenario, each group ordered by seed.

    Args:
        runs: Views of any scenarios and seeds.

    How:
        Keeps scenarios in first-seen order and sorts each group's seeds, so a figure
        lists them the same way every time.

    Returns:
        The runs of each scenario, keyed by scenario name.
    """
    groups: dict[str, list[RunView]] = {}
    for run in runs:
        groups.setdefault(run.scenario, []).append(run)
    return {name: sorted(group, key=lambda run: run.seed) for name, group in groups.items()}


def _curve_fields(report: AutoSpecReport) -> dict:
    """Read the curve, reference, threshold and budget fields of a full-loop report."""
    return {
        "before_curve": report.before_curve,
        "after_curve": report.after_curve,
        "reference_before": report.reference_before,
        "reference_after": report.reference_after,
        "threshold": report.config.get("autospec", {}).get("threshold"),
        "budget": report.config.get("training", {}).get("budget"),
    }


def _catalogue_scenario(report: AutoSpecReport, catalogue: Mapping[str, Scenario]) -> Scenario:
    """Look up the scenario of a report that did not record its room.

    How:
        Prefers the caller's catalogue, then falls back to the shared resolver so
        an n-rooms report can be redrawn even when only its name was stored. A
        random world is rebuilt from its seed by the resolver.
    """
    scenario = catalogue.get(report.scenario)
    if scenario is not None:
        return scenario
    try:
        return resolve_scenario(report.scenario)
    except KeyError as missing:
        raise ValueError(
            f"Report for {report.scenario!r} has no recorded room and the scenario is not "
            "in the catalogue, so it cannot be drawn"
        ) from missing

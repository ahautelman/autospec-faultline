"""The persisted result of one full Algorithm 1 run, plus its figure adapters.

One :class:`AutoSpecReport` holds everything for a ``(scenario, seed)`` run in plain
values, so tables and figures are rebuilt from JSON without retraining; the figures read
it through :func:`faultline.viz.run.run_view_from_report`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from faultline.graph.snapshot import GraphSnapshot, snapshot_from_dict
from faultline.experiments.records import (
    REFERENCE_STEP,
    Checkpoint,
    Overlay,
    ScenarioReport,
    ScenarioSpread,
    SeedSummary,
    success_at,
)
from faultline.refinement.loop import EdgeSuccess, RefinementStep
from faultline.refinement.trace import trace_from_dict, trajectory_from_dict
from faultline.refinement.trajectories import TrajectorySample

# Version 1 reports predate traces, snapshots and trajectories; their figures fall
# back to the curves alone. Version 2 added those fields. Version 3 makes the
# paper reference numbers optional and records scenario metadata (the 100-rooms
# worlds have no paper reference and use per-edge operator expectations instead).
# The loader keeps every new field optional, so version 1 and 2 files still load.
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class AutoSpecReport:
    """One scenario and seed run through the full loop.

    Args:
        scenario: Scenario name.
        seed: Run seed.
        expected_operator: The operator the paper's table expects, or ``None``.
        config: The AutoSpec and training settings the run used.
        steps: One record per weak edge the loop examined, in scan order.
        initial_edges: Per-edge success before any refinement.
        final_edges: Per-edge success of the repaired graph.
        best_path_before: Whole-graph success before refinement.
        best_path_after: Whole-graph success after refinement.
        reference_before: The paper's before number, or ``None`` when the paper
            gives none (the 100-rooms worlds).
        reference_after: The paper's after number, or ``None`` when absent.
        before_curve: Best-path learning curve of the initial graph.
        after_curve: Best-path learning curve of the repaired graph.
        elapsed_seconds: Wall-clock time of the run inside the shared pool.
        schema_version: ``1`` for reports without the figure data below, ``2`` once
            it is recorded, ``3`` once references may be absent and scenario
            metadata is carried.
        scenario_metadata: The scenario's recorded metadata, including per-edge
            operator expectations for the 100-rooms worlds. It is displayed but
            never scored.
        layout: The room layout as :func:`~faultline.envs.layout.layout_to_dict`
            output, so the room is redrawn without the scenario catalogue.
        unsafe_region: The hidden trap as a serialised predicate, or ``None``.
        initial_graph: Snapshot of the graph before any refinement.
        final_graph: Snapshot of the repaired graph.
        trajectories_before: Sampled rollouts of the initial graph's best path.
        trajectories_after: Sampled rollouts of the repaired graph's best path.

    How:
        Stores primitives and small frozen records only, so it round-trips through
        JSON. The figure data defaults to empty so version 1 files still load.
    """

    scenario: str
    seed: int
    expected_operator: str | None
    config: dict
    steps: tuple[RefinementStep, ...]
    initial_edges: tuple[EdgeSuccess, ...]
    final_edges: tuple[EdgeSuccess, ...]
    best_path_before: float
    best_path_after: float
    reference_before: float | None
    reference_after: float | None
    before_curve: tuple[Checkpoint, ...]
    after_curve: tuple[Checkpoint, ...]
    elapsed_seconds: float
    schema_version: int = 1
    layout: dict | None = None
    unsafe_region: dict | None = None
    initial_graph: GraphSnapshot | None = None
    final_graph: GraphSnapshot | None = None
    trajectories_before: tuple[TrajectorySample, ...] = ()
    trajectories_after: tuple[TrajectorySample, ...] = ()
    scenario_metadata: dict = field(default_factory=dict)

    @property
    def operators(self) -> tuple[str | None, ...]:
        """The committed operator per weak edge.

        How:
            Reads each step's operator, ``None`` where no operator helped.

        Returns:
            The operators in scan order.
        """
        return tuple(step.operator for step in self.steps)

    @property
    def matched_expected(self) -> bool:
        """Whether the paper's expected operator was committed.

        How:
            Checks the expected operator against the committed ones. A scenario
            with no single expected operator (the 100-rooms worlds) is never
            scored, so it returns ``False`` rather than matching a ``None`` step.

        Returns:
            ``True`` when a non-``None`` expected operator was committed at least
            once.
        """
        return self.expected_operator is not None and self.expected_operator in self.operators


def report_to_dict(report: AutoSpecReport) -> dict:
    """Convert a report to a JSON-serialisable mapping.

    Args:
        report: The report to convert.

    How:
        Uses :func:`dataclasses.asdict`, which already recurses into the nested
        frozen records and turns tuples into lists.

    Returns:
        The mapping.
    """
    return asdict(report)


def report_from_dict(payload: Mapping) -> AutoSpecReport:
    """Rebuild a report from :func:`report_to_dict`'s output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Rebuilds each nested record from its dictionary and restores tuples. Fields
        a version 1 file lacks keep their defaults.

    Returns:
        The report.
    """
    fields = dict(payload)
    fields["steps"] = tuple(_step_from_dict(step) for step in payload["steps"])
    for name in ("initial_edges", "final_edges"):
        fields[name] = tuple(EdgeSuccess(**edge) for edge in payload[name])
    for name in ("before_curve", "after_curve"):
        fields[name] = tuple(Checkpoint(**point) for point in payload[name])
    for name in ("initial_graph", "final_graph"):
        if payload.get(name) is not None:
            fields[name] = snapshot_from_dict(payload[name])
    for name in ("trajectories_before", "trajectories_after"):
        fields[name] = tuple(trajectory_from_dict(t) for t in payload.get(name, ()))
    return AutoSpecReport(**fields)


def _step_from_dict(payload: Mapping) -> RefinementStep:
    """Rebuild one refinement step, with its trace when it has one.

    Args:
        payload: The step's mapping read back from JSON.

    How:
        Restores the operator tuple and decodes the trace if present.

    Returns:
        The step.
    """
    trace = payload.get("trace")
    return RefinementStep(
        **{
            **payload,
            "operators_tried": tuple(payload["operators_tried"]),
            "trace": trace_from_dict(trace) if trace is not None else None,
        }
    )

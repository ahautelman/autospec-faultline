"""Regression tests for fault-local baseline evidence selection."""

from __future__ import annotations

from faultline.experiments.faultline_benchmark import (
    EstimateRecord,
    _baseline_evidence,
)
from faultline.faults import unreachable_start_case
from faultline.refinement.loop import (
    AutoSpecConfig,
    AutoSpecRun,
    EdgeSuccess,
)


def _record(probability: float, successes: int, failures: int) -> EstimateRecord:
    return EstimateRecord(
        source="start",
        target="goal",
        probability=probability,
        reached_rollouts=successes,
        successes=successes,
        failures=failures,
        unsafe_rollouts=0,
        unsafe_reached_rollouts=0,
        visit_states=(),
    )


def test_baseline_evidence_matches_autospec_initial_estimate_not_first_lsts_probe():
    """LSTS may probe an edge several times before returning its final estimate."""

    case = unreachable_start_case().fault

    early = _record(0.0, successes=0, failures=30)
    middle = _record(0.2, successes=6, failures=24)
    final = _record(0.43333333333333335, successes=13, failures=17)

    run = AutoSpecRun(
        graph=case.faulty_graph,
        config=AutoSpecConfig(),
        steps=(),
        initial_edge_success=(
            EdgeSuccess(
                source="start",
                target="goal",
                probability=0.43333333333333335,
            ),
        ),
        edge_success=(),
        best_path_success=0.43333333333333335,
    )

    selected = _baseline_evidence(
        case,
        [early, middle, final],
        run,
    )

    assert selected is final
    assert selected.successes == 13
    assert selected.failures == 17


def test_baseline_evidence_uses_first_matching_probability_from_initial_phase():
    """A later refinement evaluation with the same value must not replace baseline data."""

    case = unreachable_start_case().fault

    initial = _record(0.5, successes=15, failures=15)
    later_same_probability = _record(0.5, successes=14, failures=16)

    run = AutoSpecRun(
        graph=case.faulty_graph,
        config=AutoSpecConfig(),
        steps=(),
        initial_edge_success=(
            EdgeSuccess("start", "goal", 0.5),
        ),
        edge_success=(),
        best_path_success=0.5,
    )

    selected = _baseline_evidence(
        case,
        [initial, later_same_probability],
        run,
    )

    assert selected is initial

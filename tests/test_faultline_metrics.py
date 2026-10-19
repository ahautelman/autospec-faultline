from __future__ import annotations

from faultline.experiments.faultline_metrics import evaluate_recovery
from faultline.faults import (
    goal_trap_case,
    missing_waypoint_case,
    overwide_safety_case,
)


def test_unchanged_overwide_safety_is_not_recovered():
    benchmark = overwide_safety_case()
    case = benchmark.fault

    result = evaluate_recovery(
        case,
        case.faulty_graph,
        (),
    )

    assert result["repaired_iou"] == result["baseline_iou"]
    assert result["normalized_recovery_gain"] == 0.0
    assert result["recovered"] is False


def test_missing_waypoint_recovery_is_topology_based():
    benchmark = missing_waypoint_case()
    case = benchmark.fault

    result = evaluate_recovery(
        case,
        case.correct_graph,
        (),
    )

    assert result["topology_recovered"] is True
    assert result["graph_similarity"] == 1.0
    assert result["recovered"] is True


def test_goal_trap_metrics_distinguish_precision_from_recall():
    benchmark = goal_trap_case()
    case = benchmark.fault

    # The faulty target contains the whole correct target plus an extra trap.
    result = evaluate_recovery(
        case,
        case.faulty_graph,
        (),
    )

    assert result["baseline_recall"] == 1.0
    assert result["baseline_precision"] < 1.0
    assert result["recovered"] is False

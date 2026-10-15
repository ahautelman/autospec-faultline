from __future__ import annotations

from faultline.faults import (
    FaultType,
    all_fault_cases,
    goal_trap_case,
    missing_alternative_case,
    missing_waypoint_case,
    overwide_safety_case,
    summarize_fault,
    unreachable_start_case,
)


def test_all_injected_graphs_are_acyclic_and_independent():

    for benchmark in all_fault_cases():

        case = benchmark.fault

        case.correct_graph.assert_acyclic()
        case.faulty_graph.assert_acyclic()

        assert (
            case.correct_graph
            is not case.faulty_graph
        )


def test_goal_trap_changes_only_goal_region():

    case = goal_trap_case().fault

    summary = summarize_fault(
        case
    )

    assert (
        case.fault_type
        is FaultType.GOAL_TRAP
    )

    assert (
        case.expected_operator
        == "seq"
    )

    assert (
        summary.vertex_region_changes
        == ("goal",)
    )

    assert (
        summary.correct_vertices
        == summary.faulty_vertices
    )

    assert (
        summary.correct_edges
        == summary.faulty_edges
    )

    assert not summary.edge_safe_changes


def test_overwide_safety_changes_only_edge_safety():

    case = overwide_safety_case().fault

    summary = summarize_fault(
        case
    )

    assert (
        case.fault_type
        is FaultType.OVERWIDE_SAFETY
    )

    assert (
        case.expected_operator
        == "seq"
    )

    assert (
        summary.edge_safe_changes
        == ("start->goal",)
    )

    assert not summary.vertex_region_changes


def test_unreachable_start_changes_only_start_region():

    case = unreachable_start_case().fault

    summary = summarize_fault(
        case
    )

    assert (
        case.fault_type
        is FaultType.UNREACHABLE_START
    )

    assert (
        case.expected_operator
        == "past"
    )

    assert (
        summary.vertex_region_changes
        == ("start",)
    )

    assert not summary.edge_safe_changes


def test_missing_waypoint_removes_vertex_and_collapses_two_edges():

    case = missing_waypoint_case().fault

    summary = summarize_fault(
        case
    )

    assert (
        case.fault_type
        is FaultType.MISSING_WAYPOINT
    )

    assert (
        case.expected_operator
        == "add"
    )

    assert (
        summary.removed_vertices
        == ("waypoint",)
    )

    assert set(
        summary.removed_edges
    ) == {
        "start->waypoint",
        "waypoint->goal",
    }

    assert (
        summary.added_edges
        == ("start->goal",)
    )

    assert summary.correct_vertices == 3
    assert summary.faulty_vertices == 2

    assert summary.correct_edges == 2
    assert summary.faulty_edges == 1


def test_missing_alternative_removes_exact_ground_truth_edge():

    case = missing_alternative_case().fault

    summary = summarize_fault(
        case
    )

    assert (
        case.fault_type
        is FaultType.MISSING_ALTERNATIVE
    )

    assert (
        case.expected_operator
        == "or"
    )

    assert (
        summary.removed_edges
        == ("mid1->mid2",)
    )

    assert not summary.added_edges

    assert (
        summary.correct_vertices
        == summary.faulty_vertices
    )

    assert (
        summary.correct_edges
        == summary.faulty_edges + 1
    )
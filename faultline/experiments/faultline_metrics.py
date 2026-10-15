"""Ground-truth recovery metrics for the FaultLine benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from faultline.faults import FaultCase, FaultType
from faultline.graph import AbstractGraph, Edge, Vertex
from faultline.spectrl.predicates import Predicate


def evaluate_recovery(
    case: FaultCase,
    repaired_graph: AbstractGraph,
    baseline_visit_states: tuple[np.ndarray, ...] = (),
    *,
    world_size: float = 24.0,
    grid_resolution: int = 121,
    quality_threshold: float = 0.80,
) -> dict[str, Any]:
    """Compare an AutoSpec repair against the known correct specification."""

    common: dict[str, Any] = {
        "metric_type": None,
        "baseline_iou": None,
        "repaired_iou": None,
        "iou_improvement": None,
        "normalized_recovery_gain": None,
        "baseline_precision": None,
        "baseline_recall": None,
        "repaired_precision": None,
        "repaired_recall": None,
        "baseline_weighted_iou": None,
        "repaired_weighted_iou": None,
        "weighted_iou_improvement": None,
        "subset_fraction": None,
        "normalized_graph_edit_distance": normalized_graph_edit_distance(
            case.correct_graph,
            repaired_graph,
        ),
        "graph_similarity": None,
        "missing_edge_recovered": None,
        "topology_recovered": None,
        "recovered": False,
        "recovery_reason": "metric not applicable",
    }

    if case.fault_type is FaultType.GOAL_TRAP:
        if (
            case.correct_graph.target is None
            or case.faulty_graph.target is None
            or repaired_graph.target is None
        ):
            return common
        return _region_result(
            common,
            correct=case.correct_graph.target.region,
            faulty=case.faulty_graph.target.region,
            repaired=repaired_graph.target.region,
            visit_states=baseline_visit_states,
            world_size=world_size,
            resolution=grid_resolution,
            quality_threshold=quality_threshold,
            metric_type="target_region",
        )

    if case.fault_type is FaultType.UNREACHABLE_START:
        if (
            case.correct_graph.source is None
            or case.faulty_graph.source is None
            or repaired_graph.source is None
        ):
            return common
        return _region_result(
            common,
            correct=case.correct_graph.source.region,
            faulty=case.faulty_graph.source.region,
            repaired=repaired_graph.source.region,
            visit_states=baseline_visit_states,
            world_size=world_size,
            resolution=grid_resolution,
            quality_threshold=quality_threshold,
            metric_type="source_region",
        )

    if case.fault_type is FaultType.OVERWIDE_SAFETY:
        correct_edge = _find_original_fault_edge(case.correct_graph, case.affected_edge)
        faulty_edge = _find_original_fault_edge(case.faulty_graph, case.affected_edge)
        repaired_edge = _repaired_source_edge(repaired_graph, case.affected_edge)
        if correct_edge is None or faulty_edge is None or repaired_edge is None:
            return common
        return _region_result(
            common,
            correct=correct_edge.safe,
            faulty=faulty_edge.safe,
            repaired=repaired_edge.safe,
            visit_states=baseline_visit_states,
            world_size=world_size,
            resolution=grid_resolution,
            quality_threshold=quality_threshold,
            metric_type="safe_region",
        )

    if case.fault_type is FaultType.MISSING_WAYPOINT:
        correct_waypoint = _vertex_named(case.correct_graph, case.affected_vertex)
        repaired_waypoint = _fresh_internal_vertex(case.faulty_graph, repaired_graph)

        waypoint_iou = None
        waypoint_weighted_iou = None
        if correct_waypoint is not None and repaired_waypoint is not None:
            waypoint_iou = predicate_iou(
                correct_waypoint.region,
                repaired_waypoint.region,
                world_size=world_size,
                resolution=grid_resolution,
            )
            waypoint_weighted_iou = weighted_predicate_iou(
                correct_waypoint.region,
                repaired_waypoint.region,
                baseline_visit_states,
            )

        canonical_ged = missing_waypoint_graph_edit_distance(case, repaired_graph)
        graph_similarity = max(0.0, 1.0 - canonical_ged)
        topology_recovered = canonical_ged == 0.0
        recovered = graph_similarity >= quality_threshold

        result = dict(common)
        result.update(
            {
                "metric_type": "waypoint_graph",
                "repaired_iou": waypoint_iou,
                "repaired_weighted_iou": waypoint_weighted_iou,
                "normalized_graph_edit_distance": canonical_ged,
                "graph_similarity": graph_similarity,
                "topology_recovered": topology_recovered,
                "recovered": recovered,
                "recovery_reason": (
                    "graph structure recovered"
                    if recovered
                    else "graph structure remains too far from ground truth"
                ),
            }
        )
        return result

    if case.fault_type is FaultType.MISSING_ALTERNATIVE:
        expected = case.affected_edge
        edge_recovered = bool(
            expected is not None
            and _edge_named(repaired_graph, expected[0], expected[1]) is not None
        )
        ged = normalized_graph_edit_distance(case.correct_graph, repaired_graph)
        graph_similarity = max(0.0, 1.0 - ged)
        topology_recovered = ged == 0.0
        recovered = edge_recovered and graph_similarity >= quality_threshold

        result = dict(common)
        result.update(
            {
                "metric_type": "graph_structure",
                "normalized_graph_edit_distance": ged,
                "graph_similarity": graph_similarity,
                "missing_edge_recovered": edge_recovered,
                "topology_recovered": topology_recovered,
                "recovered": recovered,
                "recovery_reason": (
                    "missing alternative and graph structure recovered"
                    if recovered
                    else (
                        "missing alternative restored but extra graph edits remain"
                        if edge_recovered
                        else "missing alternative was not restored"
                    )
                ),
            }
        )
        return result

    if case.fault_type is FaultType.NONE:
        ged = normalized_graph_edit_distance(case.correct_graph, repaired_graph)
        similarity = max(0.0, 1.0 - ged)
        result = dict(common)
        result.update(
            {
                "metric_type": "no_fault_control",
                "normalized_graph_edit_distance": ged,
                "graph_similarity": similarity,
                "topology_recovered": ged == 0.0,
                "recovered": ged == 0.0,
                "recovery_reason": (
                    "control graph unchanged" if ged == 0.0 else "control graph was modified"
                ),
            }
        )
        return result

    return common


def predicate_iou(
    first: Predicate,
    second: Predicate,
    *,
    world_size: float = 24.0,
    resolution: int = 121,
) -> float:
    counts = _predicate_counts(first, second, world_size=world_size, resolution=resolution)
    return 1.0 if counts["union"] == 0 else counts["intersection"] / counts["union"]


def predicate_precision_recall(
    correct: Predicate,
    candidate: Predicate,
    *,
    world_size: float = 24.0,
    resolution: int = 121,
) -> tuple[float | None, float | None]:
    """Return geometric precision and recall of a candidate region.

    Precision answers: of the states accepted by the candidate, how many belong
    to the correct region?  Recall answers: how much of the correct region did
    the candidate retain?  Together they reveal over-shrinking that IoU alone
    can obscure.
    """

    counts = _predicate_counts(correct, candidate, world_size=world_size, resolution=resolution)
    precision = (
        counts["intersection"] / counts["second"] if counts["second"] > 0 else None
    )
    recall = counts["intersection"] / counts["first"] if counts["first"] > 0 else None
    return precision, recall


def weighted_predicate_iou(
    first: Predicate,
    second: Predicate,
    visit_states: tuple[np.ndarray, ...],
) -> float | None:
    if not visit_states:
        return None

    intersection = 0
    union = 0
    for state in visit_states:
        point = np.asarray(state, dtype=float).reshape(-1)
        a = bool(first.evaluate(point))
        b = bool(second.evaluate(point))
        intersection += int(a and b)
        union += int(a or b)

    if union == 0:
        return None
    return intersection / union


def predicate_subset_fraction(
    candidate: Predicate,
    containing: Predicate,
    *,
    world_size: float = 24.0,
    resolution: int = 121,
) -> float | None:
    xs = np.linspace(0.0, world_size, resolution)
    ys = np.linspace(0.0, world_size, resolution)
    candidate_count = 0
    contained_count = 0

    for x in xs:
        for y in ys:
            point = np.array([x, y], dtype=float)
            if candidate.evaluate(point):
                candidate_count += 1
                contained_count += int(containing.evaluate(point))

    if candidate_count == 0:
        return None
    return contained_count / candidate_count


def normalized_graph_edit_distance(
    correct_graph: AbstractGraph,
    candidate_graph: AbstractGraph,
) -> float:
    correct_vertices = {vertex.name for vertex in correct_graph.vertices}
    candidate_vertices = {vertex.name for vertex in candidate_graph.vertices}
    correct_edges = {
        (edge.source.name, edge.target.name)
        for edge in correct_graph.edges
    }
    candidate_edges = {
        (edge.source.name, edge.target.name)
        for edge in candidate_graph.edges
    }

    edits = len(correct_vertices ^ candidate_vertices) + len(correct_edges ^ candidate_edges)
    denominator = max(1, len(correct_vertices) + len(correct_edges))
    return edits / denominator


def missing_waypoint_graph_edit_distance(
    case: FaultCase,
    repaired_graph: AbstractGraph,
) -> float:
    waypoint_name = case.affected_vertex
    if waypoint_name is None:
        return normalized_graph_edit_distance(case.correct_graph, repaired_graph)

    fresh = _fresh_internal_vertex(case.faulty_graph, repaired_graph)
    if fresh is None:
        return normalized_graph_edit_distance(case.correct_graph, repaired_graph)

    remap = {fresh.name: waypoint_name}
    correct_vertices = {vertex.name for vertex in case.correct_graph.vertices}
    repaired_vertices = {
        remap.get(vertex.name, vertex.name)
        for vertex in repaired_graph.vertices
    }
    correct_edges = {
        (edge.source.name, edge.target.name)
        for edge in case.correct_graph.edges
    }
    repaired_edges = {
        (
            remap.get(edge.source.name, edge.source.name),
            remap.get(edge.target.name, edge.target.name),
        )
        for edge in repaired_graph.edges
    }

    edits = len(correct_vertices ^ repaired_vertices) + len(correct_edges ^ repaired_edges)
    denominator = max(1, len(correct_vertices) + len(correct_edges))
    return edits / denominator


def _region_result(
    base: dict[str, Any],
    *,
    correct: Predicate,
    faulty: Predicate,
    repaired: Predicate,
    visit_states: tuple[np.ndarray, ...],
    world_size: float,
    resolution: int,
    quality_threshold: float,
    metric_type: str,
) -> dict[str, Any]:
    baseline_iou = predicate_iou(
        correct,
        faulty,
        world_size=world_size,
        resolution=resolution,
    )
    repaired_iou = predicate_iou(
        correct,
        repaired,
        world_size=world_size,
        resolution=resolution,
    )
    iou_improvement = repaired_iou - baseline_iou
    remaining_error = 1.0 - baseline_iou
    recovery_gain = (
        (1.0 if repaired_iou >= 1.0 - 1e-12 else 0.0)
        if remaining_error <= 1e-12
        else iou_improvement / remaining_error
    )

    baseline_precision, baseline_recall = predicate_precision_recall(
        correct,
        faulty,
        world_size=world_size,
        resolution=resolution,
    )
    repaired_precision, repaired_recall = predicate_precision_recall(
        correct,
        repaired,
        world_size=world_size,
        resolution=resolution,
    )

    baseline_weighted = weighted_predicate_iou(correct, faulty, visit_states)
    repaired_weighted = weighted_predicate_iou(correct, repaired, visit_states)
    weighted_improvement = (
        repaired_weighted - baseline_weighted
        if baseline_weighted is not None and repaired_weighted is not None
        else None
    )

    subset = predicate_subset_fraction(
        repaired,
        faulty,
        world_size=world_size,
        resolution=resolution,
    )

    recovered = bool(
        repaired_iou > baseline_iou + 1e-9
        and recovery_gain >= quality_threshold
    )

    if recovered:
        reason = "predicate recovered most of the injected geometric error"
    elif repaired_iou <= baseline_iou + 1e-9:
        reason = "repair did not improve geometric agreement with ground truth"
    elif (
        repaired_precision is not None
        and repaired_recall is not None
        and repaired_precision >= 0.95
        and repaired_recall < quality_threshold
    ):
        reason = "repair removed wrong states but over-shrank the correct region"
    else:
        reason = "partial predicate recovery below the configured quality threshold"

    result = dict(base)
    result.update(
        {
            "metric_type": metric_type,
            "baseline_iou": baseline_iou,
            "repaired_iou": repaired_iou,
            "iou_improvement": iou_improvement,
            "normalized_recovery_gain": recovery_gain,
            "baseline_precision": baseline_precision,
            "baseline_recall": baseline_recall,
            "repaired_precision": repaired_precision,
            "repaired_recall": repaired_recall,
            "baseline_weighted_iou": baseline_weighted,
            "repaired_weighted_iou": repaired_weighted,
            "weighted_iou_improvement": weighted_improvement,
            "subset_fraction": subset,
            "recovered": recovered,
            "recovery_reason": reason,
        }
    )
    return result


def _predicate_counts(
    first: Predicate,
    second: Predicate,
    *,
    world_size: float,
    resolution: int,
) -> dict[str, int]:
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    xs = np.linspace(0.0, world_size, resolution)
    ys = np.linspace(0.0, world_size, resolution)
    first_count = 0
    second_count = 0
    intersection = 0
    union = 0

    for x in xs:
        for y in ys:
            point = np.array([x, y], dtype=float)
            a = bool(first.evaluate(point))
            b = bool(second.evaluate(point))
            first_count += int(a)
            second_count += int(b)
            intersection += int(a and b)
            union += int(a or b)

    return {
        "first": first_count,
        "second": second_count,
        "intersection": intersection,
        "union": union,
    }


def _vertex_named(graph: AbstractGraph, name: str | None) -> Vertex | None:
    if name is None:
        return None
    for vertex in graph.vertices:
        if vertex.name == name:
            return vertex
    return None


def _edge_named(
    graph: AbstractGraph,
    source_name: str,
    target_name: str,
) -> Edge | None:
    for edge in graph.edges:
        if edge.source.name == source_name and edge.target.name == target_name:
            return edge
    return None


def _find_original_fault_edge(
    graph: AbstractGraph,
    affected_edge: tuple[str, str] | None,
) -> Edge | None:
    if affected_edge is None:
        return None
    return _edge_named(graph, affected_edge[0], affected_edge[1])


def _repaired_source_edge(
    graph: AbstractGraph,
    affected_edge: tuple[str, str] | None,
) -> Edge | None:
    if affected_edge is not None:
        exact = _edge_named(graph, affected_edge[0], affected_edge[1])
        if exact is not None:
            return exact
    if graph.source is None:
        return None
    outgoing = graph.successors(graph.source)
    return outgoing[0] if len(outgoing) == 1 else None


def _fresh_internal_vertex(
    faulty_graph: AbstractGraph,
    repaired_graph: AbstractGraph,
) -> Vertex | None:
    existing = {vertex.name for vertex in faulty_graph.vertices}
    candidates = [
        vertex
        for vertex in repaired_graph.vertices
        if vertex.name not in existing
    ]
    internal = [
        vertex
        for vertex in candidates
        if vertex is not repaired_graph.source and vertex is not repaired_graph.target
    ]
    if len(internal) == 1:
        return internal[0]
    if len(candidates) == 1:
        return candidates[0]
    return None

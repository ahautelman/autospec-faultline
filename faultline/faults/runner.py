"""Inspection utilities for FaultLine fault injection."""

from __future__ import annotations

from dataclasses import dataclass

from faultline.faults.base import FaultCase
from faultline.graph import AbstractGraph


@dataclass(frozen=True)
class FaultSummary:
    """Compact structural comparison of correct and faulty graphs."""

    name: str
    fault_type: str
    expected_operator: str | None

    correct_vertices: int
    faulty_vertices: int

    correct_edges: int
    faulty_edges: int

    vertex_region_changes: tuple[str, ...]
    edge_safe_changes: tuple[str, ...]

    added_edges: tuple[str, ...]
    removed_edges: tuple[str, ...]

    added_vertices: tuple[str, ...]
    removed_vertices: tuple[str, ...]


def summarize_fault(
    case: FaultCase,
) -> FaultSummary:
    """Compare the correct and faulty specification graphs."""

    correct_vertices = {
        vertex.name: vertex
        for vertex in case.correct_graph.vertices
    }

    faulty_vertices = {
        vertex.name: vertex
        for vertex in case.faulty_graph.vertices
    }

    shared_vertices = sorted(
        correct_vertices.keys()
        & faulty_vertices.keys()
    )

    vertex_changes = tuple(
        name
        for name in shared_vertices
        if (
            correct_vertices[name].region
            != faulty_vertices[name].region
        )
    )

    correct_edges = _edge_map(
        case.correct_graph
    )

    faulty_edges = _edge_map(
        case.faulty_graph
    )

    shared_edges = sorted(
        correct_edges.keys()
        & faulty_edges.keys()
    )

    safe_changes = tuple(
        _edge_label(key)
        for key in shared_edges
        if (
            correct_edges[key].safe
            != faulty_edges[key].safe
        )
    )

    return FaultSummary(
        name=case.name,
        fault_type=case.fault_type.value,
        expected_operator=case.expected_operator,

        correct_vertices=len(
            case.correct_graph.vertices
        ),

        faulty_vertices=len(
            case.faulty_graph.vertices
        ),

        correct_edges=len(
            case.correct_graph.edges
        ),

        faulty_edges=len(
            case.faulty_graph.edges
        ),

        vertex_region_changes=vertex_changes,
        edge_safe_changes=safe_changes,

        added_edges=tuple(
            _edge_label(key)
            for key in sorted(
                faulty_edges.keys()
                - correct_edges.keys()
            )
        ),

        removed_edges=tuple(
            _edge_label(key)
            for key in sorted(
                correct_edges.keys()
                - faulty_edges.keys()
            )
        ),

        added_vertices=tuple(
            sorted(
                faulty_vertices.keys()
                - correct_vertices.keys()
            )
        ),

        removed_vertices=tuple(
            sorted(
                correct_vertices.keys()
                - faulty_vertices.keys()
            )
        ),
    )


def format_summary(
    summary: FaultSummary,
) -> str:
    """Render a FaultSummary."""

    lines = [
        f"[{summary.name}] {summary.fault_type}",

        (
            f"  expected operator: "
            f"{summary.expected_operator}"
        ),

        (
            "  graph size: "
            f"V {summary.correct_vertices}"
            f"->{summary.faulty_vertices}, "
            f"E {summary.correct_edges}"
            f"->{summary.faulty_edges}"
        ),
    ]

    _append(
        lines,
        "changed vertex regions",
        summary.vertex_region_changes,
    )

    _append(
        lines,
        "changed edge safeties",
        summary.edge_safe_changes,
    )

    _append(
        lines,
        "added vertices",
        summary.added_vertices,
    )

    _append(
        lines,
        "removed vertices",
        summary.removed_vertices,
    )

    _append(
        lines,
        "added edges",
        summary.added_edges,
    )

    _append(
        lines,
        "removed edges",
        summary.removed_edges,
    )

    return "\n".join(lines)


def _edge_map(
    graph: AbstractGraph,
):
    result = {}

    for edge in graph.edges:

        key = (
            edge.source.name,
            edge.target.name,
        )

        if key in result:
            raise ValueError(
                "Fault summary assumes at most one edge "
                "for each source/target pair; "
                f"duplicate {key!r} found."
            )

        result[key] = edge

    return result


def _edge_label(
    key: tuple[str, str],
) -> str:
    return f"{key[0]}->{key[1]}"


def _append(
    lines: list[str],
    label: str,
    values: tuple[str, ...],
) -> None:

    if values:
        lines.append(
            f"  {label}: {', '.join(values)}"
        )
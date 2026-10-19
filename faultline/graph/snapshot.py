"""A plain-data, JSON-friendly view of an abstract graph.

Figures and traces are drawn after training from files on disk, so the graph's
regions have to outlive the process. A snapshot stores each vertex and edge by
name with its region as a serialised predicate, which is all a figure needs and
keeps the live :class:`AbstractGraph` out of the persisted records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from faultline.geometry.serialize import Payload, predicate_to_dict
from faultline.graph.abstract_graph import AbstractGraph


@dataclass(frozen=True)
class VertexSnapshot:
    """One vertex: its name and serialised region.

    Args:
        name: The vertex's unique label.
        region: The region predicate as a serialised mapping.
    """

    name: str
    region: Payload


@dataclass(frozen=True)
class EdgeSnapshot:
    """One edge: its endpoint names and serialised safe region.

    Args:
        source: Source vertex name.
        target: Target vertex name.
        safe: The edge's safe-region predicate as a serialised mapping.
    """

    source: str
    target: str
    safe: Payload


@dataclass(frozen=True)
class GraphSnapshot:
    """The vertices, edges and endpoints of a graph at one moment.

    Args:
        vertices: Vertex snapshots in graph insertion order.
        edges: Edge snapshots in graph insertion order.
        source: Name of the graph source vertex, or ``None`` when unset.
        target: Name of the graph target vertex, or ``None`` when unset.

    How:
        Keeps insertion order so two snapshots of one graph line up positionally,
        and refers to vertices by name because identity does not survive JSON.
    """

    vertices: tuple[VertexSnapshot, ...]
    edges: tuple[EdgeSnapshot, ...]
    source: str | None
    target: str | None


def snapshot_graph(graph: AbstractGraph) -> GraphSnapshot:
    """Capture a graph as plain data.

    Args:
        graph: The graph to capture.

    How:
        Serialises every vertex region and edge safe region and records the
        source and target names.

    Returns:
        The graph's snapshot.
    """
    return GraphSnapshot(
        vertices=tuple(
            VertexSnapshot(vertex.name, predicate_to_dict(vertex.region))
            for vertex in graph.vertices
        ),
        edges=tuple(
            EdgeSnapshot(edge.source.name, edge.target.name, predicate_to_dict(edge.safe))
            for edge in graph.edges
        ),
        source=graph.source.name if graph.source is not None else None,
        target=graph.target.name if graph.target is not None else None,
    )


def snapshot_from_dict(payload: Mapping[str, Any]) -> GraphSnapshot:
    """Rebuild a snapshot from ``dataclasses.asdict`` output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Rebuilds each nested record and restores tuples.

    Returns:
        The snapshot.
    """
    return GraphSnapshot(
        vertices=tuple(VertexSnapshot(**vertex) for vertex in payload["vertices"]),
        edges=tuple(EdgeSnapshot(**edge) for edge in payload["edges"]),
        source=payload["source"],
        target=payload["target"],
    )

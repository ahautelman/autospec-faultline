"""What one committed refinement changed in the graph, as plain data.

Operators return whole graphs, so the edit is recovered by comparing the graph before
and after: regions that were tightened in place, vertices and edges that were added,
and edges that were removed. Diffing instead of instrumenting each operator keeps the
operators untouched and describes a new operator's edit with no extra code. A trace
also carries the samples the edit was based on, so a figure can show both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from faultline.geometry.serialize import Payload
from faultline.graph.snapshot import EdgeSnapshot, GraphSnapshot, VertexSnapshot
from faultline.refinement.trajectories import StepSamples, TrajectorySample


@dataclass(frozen=True)
class VertexChange:
    """A vertex whose region was replaced in place.

    Args:
        name: The vertex name.
        before: Region before the edit, serialised.
        after: Region after the edit, serialised.
    """

    name: str
    before: Payload
    after: Payload


@dataclass(frozen=True)
class SafeChange:
    """An edge whose safe region was replaced in place.

    Args:
        source: Source vertex name of the edge.
        target: Target vertex name of the edge.
        before: Safe region before the edit, serialised.
        after: Safe region after the edit, serialised.
    """

    source: str
    target: str
    before: Payload
    after: Payload


@dataclass(frozen=True)
class StepTrace:
    """The graph edit one refinement made, with the samples behind it.

    Args:
        vertex_changes: Vertices whose region changed (ReachRefine, in-place
            PastRefine).
        safe_changes: Edges whose safe region changed (AvoidRefine).
        added_vertices: Vertices the edit created (waypoints, entry regions).
        added_edges: Edges the edit created (routes, rewired edges).
        removed_edges: ``(source, target)`` names of edges the edit dropped.
        samples: Witness points, unsafe tails and sampled paths of the edge.

    How:
        Describes the edit by what changed rather than by operator name, so an
        annotator reads the shape of the edit and ``seq`` needs no special case.
    """

    vertex_changes: tuple[VertexChange, ...]
    safe_changes: tuple[SafeChange, ...]
    added_vertices: tuple[VertexSnapshot, ...]
    added_edges: tuple[EdgeSnapshot, ...]
    removed_edges: tuple[tuple[str, str], ...]
    samples: StepSamples


def build_trace(before: GraphSnapshot, after: GraphSnapshot, samples: StepSamples) -> StepTrace:
    """Describe the edit that turned ``before`` into ``after``.

    Args:
        before: Snapshot of the graph the operator ran on.
        after: Snapshot of the graph it returned.
        samples: The samples the edit was based on.

    How:
        Matches vertices by name. Matches edges by endpoint names, pairing the
        i-th occurrence of a name pair on each side so parallel edges stay
        distinct; an unmatched edge on the old side was removed, on the new side
        added, and a matched pair is checked for a changed safe region.

    Returns:
        The step trace.
    """
    old_vertices = {vertex.name: vertex for vertex in before.vertices}
    vertex_changes = tuple(
        VertexChange(vertex.name, old_vertices[vertex.name].region, vertex.region)
        for vertex in after.vertices
        if vertex.name in old_vertices and old_vertices[vertex.name].region != vertex.region
    )
    added_vertices = tuple(v for v in after.vertices if v.name not in old_vertices)
    matched, added_edges = _match_edges(before.edges, after.edges)
    removed = tuple(
        (edge.source, edge.target) for edge in before.edges if id(edge) not in matched
    )
    safe_changes = tuple(
        SafeChange(old.source, old.target, old.safe, new.safe)
        for old, new in matched.values()
        if old.safe != new.safe
    )
    return StepTrace(vertex_changes, safe_changes, added_vertices, added_edges, removed, samples)


def _match_edges(
    old: tuple[EdgeSnapshot, ...], new: tuple[EdgeSnapshot, ...]
) -> tuple[dict[int, tuple[EdgeSnapshot, EdgeSnapshot]], tuple[EdgeSnapshot, ...]]:
    """Pair old and new edges by endpoint names.

    Args:
        old: Edges before the edit.
        new: Edges after the edit.

    How:
        Queues the old edges under their name pair and pops one for each new edge
        with that pair, in order.

    Returns:
        A map from the ``id`` of each matched old edge to its ``(old, new)`` pair,
        and the new edges with no old counterpart.
    """
    pending: dict[tuple[str, str], list[EdgeSnapshot]] = {}
    for edge in old:
        pending.setdefault((edge.source, edge.target), []).append(edge)
    matched: dict[int, tuple[EdgeSnapshot, EdgeSnapshot]] = {}
    added = []
    for edge in new:
        queue = pending.get((edge.source, edge.target))
        if queue:
            partner = queue.pop(0)
            matched[id(partner)] = (partner, edge)
        else:
            added.append(edge)
    return matched, tuple(added)


def trace_from_dict(payload: Mapping[str, Any]) -> StepTrace:
    """Rebuild a trace from ``dataclasses.asdict`` output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Rebuilds each nested record and restores the tuples JSON turned into lists.

    Returns:
        The trace.
    """
    samples = payload["samples"]
    return StepTrace(
        vertex_changes=tuple(VertexChange(**c) for c in payload["vertex_changes"]),
        safe_changes=tuple(SafeChange(**c) for c in payload["safe_changes"]),
        added_vertices=tuple(VertexSnapshot(**v) for v in payload["added_vertices"]),
        added_edges=tuple(EdgeSnapshot(**e) for e in payload["added_edges"]),
        removed_edges=tuple((s, t) for s, t in payload["removed_edges"]),
        samples=StepSamples(
            witness_points=tuple((x, y) for x, y in samples["witness_points"]),
            unsafe_points=tuple((x, y) for x, y in samples["unsafe_points"]),
            trajectories=tuple(trajectory_from_dict(t) for t in samples["trajectories"]),
        ),
    )


def trajectory_from_dict(payload: Mapping[str, Any]) -> TrajectorySample:
    """Rebuild a trajectory sample from ``dataclasses.asdict`` output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Restores the ``(x, y)`` tuples.

    Returns:
        The sample.
    """
    return TrajectorySample(
        payload["source"],
        payload["target"],
        payload["success"],
        tuple((x, y) for x, y in payload["xy"]),
    )

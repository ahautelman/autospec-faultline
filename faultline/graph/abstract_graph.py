"""Abstract graph data structure from paper section 2.3.

``G = (V, E, beta, s, t)`` is a DAG whose vertices are checkpoint regions and
whose edges are reach-avoid subgoals. ``beta`` labels each vertex with its
target-region predicate and each edge with its safe-region predicate.

Vertices and edges use identity equality. That is deliberate: the refinement
loop attaches per-edge state (trained policy, success estimate) to specific edge
objects and mutates vertex regions in place, so two structurally similar nodes
must stay distinguishable. Structural comparisons in tests read the fields
directly instead of comparing nodes.
"""

from __future__ import annotations

import copy
from collections import deque

from faultline.spectrl.predicates import Predicate, Universe


class Vertex:
    """A checkpoint region in the abstract graph.

    Args:
        name: Human-readable label, unique within a graph.
        region: Predicate whose satisfying states form the checkpoint region.
    """

    def __init__(self, name: str, region: Predicate) -> None:
        self.name = name
        self.region = region

    def __repr__(self) -> str:
        return f"Vertex({self.name!r}, {self.region})"


class Edge:
    """A reach-avoid subgoal from a source vertex to a target vertex.

    Args:
        source: The vertex the subgoal starts in.
        target: The vertex whose region the agent must reach.
        safe: Predicate that must hold throughout the subgoal; defaults to the
            whole state space (no safety constraint).

    How:
        Stores the three components directly. ``safe`` is a predicate instance
        so it can be intersected (tightened) without touching the edge's
        geometry handling.
    """

    def __init__(self, source: Vertex, target: Vertex, safe: Predicate | None = None) -> None:
        self.source = source
        self.target = target
        self.safe = safe if safe is not None else Universe()

    def __repr__(self) -> str:
        return f"Edge({self.source.name} -> {self.target.name}, safe={self.safe})"


class AbstractGraph:
    """A DAG of checkpoint regions and reach-avoid edges.

    How:
        Keeps vertices in a name-keyed registry and edges in insertion order.
        The source and target are set explicitly by the compiler once the whole
        specification is built. Mutating operations (adding/removing edges,
        reassigning a vertex region) are the primitive edits the refinement
        operators compose.
    """

    def __init__(self) -> None:
        self._vertices: dict[str, Vertex] = {}
        self._edges: list[Edge] = []
        self._counter = 0
        self.source: Vertex | None = None
        self.target: Vertex | None = None

    @property
    def vertices(self) -> tuple[Vertex, ...]:
        """The graph's vertices in insertion order.

        How:
            Returns the registry values as an immutable tuple so callers cannot
            reorder or drop vertices by accident.

        Returns:
            The vertices in the order they were added.
        """
        return tuple(self._vertices.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        """The graph's edges in insertion order.

        How:
            Returns the internal edge list as a tuple, preserving the order that
            makes the compiler's "newly added edges" slice stable.

        Returns:
            The edges in the order they were added.
        """
        return tuple(self._edges)

    def add_vertex(self, name: str, region: Predicate) -> Vertex:
        """Create and register a vertex.

        Args:
            name: Unique label for the vertex.
            region: Predicate describing the checkpoint region.

        How:
            Builds a :class:`Vertex` and stores it under ``name``; a duplicate
            name is rejected so vertex identity stays unambiguous.

        Returns:
            The newly created vertex.

        Raises:
            ValueError: If ``name`` is already used in this graph.
        """
        if name in self._vertices:
            raise ValueError(f"Duplicate vertex name {name!r}")
        vertex = Vertex(name, region)
        self._vertices[name] = vertex
        return vertex

    def fresh_vertex(self, prefix: str, region: Predicate) -> Vertex:
        """Create a vertex with an auto-generated unique name.

        Args:
            prefix: Short operator tag used as the name stem (for example ``"ach"``).
            region: Predicate describing the checkpoint region.

        How:
            Increments the graph's counter and appends it to the prefix, giving
            names that are unique within the graph and stable for a given
            compile order.

        Returns:
            The newly created vertex.
        """
        self._counter += 1
        return self.add_vertex(f"{prefix}{self._counter}", region)

    def add_edge(
        self, source: Vertex, target: Vertex, safe: Predicate | None = None
    ) -> Edge:
        """Register a reach-avoid edge between two of this graph's vertices.

        Args:
            source: The edge's source vertex; must belong to this graph.
            target: The edge's target vertex; must belong to this graph.
            safe: Optional safe-region predicate; defaults to the universe.

        How:
            Verifies both endpoints are the very objects in this graph's
            registry (name and identity), then appends the edge.

        Returns:
            The newly created edge.

        Raises:
            ValueError: If an endpoint is not a vertex of this graph.
        """
        for endpoint in (source, target):
            if self._vertices.get(endpoint.name) is not endpoint:
                raise ValueError(f"Vertex {endpoint.name!r} does not belong to this graph")
        edge = Edge(source, target, safe)
        self._edges.append(edge)
        return edge

    def remove_edge(self, edge: Edge) -> None:
        """Remove an edge from the graph.

        Args:
            edge: The exact edge object to remove.

        How:
            Removes by identity from the internal edge list.

        Raises:
            ValueError: If the edge is not present.
        """
        if edge not in self._edges:
            raise ValueError(f"Edge {edge!r} is not in this graph")
        self._edges.remove(edge)

    def successors(self, vertex: Vertex) -> tuple[Edge, ...]:
        """Return the edges leaving ``vertex``.

        Args:
            vertex: The vertex whose outgoing edges are wanted.

        How:
            Filters the edge list by source identity, preserving edge order.

        Returns:
            Outgoing edges in insertion order.
        """
        return tuple(edge for edge in self._edges if edge.source is vertex)

    def predecessors(self, vertex: Vertex) -> tuple[Edge, ...]:
        """Return the edges entering ``vertex``.

        Args:
            vertex: The vertex whose incoming edges are wanted.

        How:
            Filters the edge list by target identity, preserving edge order.

        Returns:
            Incoming edges in insertion order.
        """
        return tuple(edge for edge in self._edges if edge.target is vertex)

    def sources(self) -> tuple[Vertex, ...]:
        """Return vertices with no incoming edges.

        How:
            A vertex is a source when no edge targets it; checked by identity.

        Returns:
            Source vertices in insertion order.
        """
        targets = {id(edge.target) for edge in self._edges}
        return tuple(v for v in self._vertices.values() if id(v) not in targets)

    def sinks(self) -> tuple[Vertex, ...]:
        """Return vertices with no outgoing edges.

        How:
            A vertex is a sink when no edge has it as a source; checked by identity.

        Returns:
            Sink vertices in insertion order.
        """
        sources = {id(edge.source) for edge in self._edges}
        return tuple(v for v in self._vertices.values() if id(v) not in sources)

    def copy(self) -> AbstractGraph:
        """Return a deep copy of the graph.

        How:
            Deep-copies the whole object, including the shared vertex objects
            that edges reference, so edits to the copy never alias the original.

        Returns:
            An independent :class:`AbstractGraph`.
        """
        return copy.deepcopy(self)

    def topological_order(self) -> tuple[Vertex, ...]:
        """Return the vertices in a topological order.

        How:
            Runs Kahn's algorithm, seeding the queue with the zero-indegree
            vertices in name order for determinism and draining it in edge
            order.

        Returns:
            Vertices ordered so every edge points forwards.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        indegree = {name: 0 for name in self._vertices}
        for edge in self._edges:
            indegree[edge.target.name] += 1
        ready = deque(sorted(name for name, degree in indegree.items() if degree == 0))
        order: list[Vertex] = []
        while ready:
            vertex = self._vertices[ready.popleft()]
            order.append(vertex)
            for edge in self.successors(vertex):
                indegree[edge.target.name] -= 1
                if indegree[edge.target.name] == 0:
                    ready.append(edge.target.name)
        if len(order) != len(self._vertices):
            raise ValueError("AbstractGraph contains a cycle")
        return tuple(order)

    def assert_acyclic(self) -> None:
        """Raise if the graph is not a DAG.

        How:
            Delegates to :meth:`topological_order`, which detects cycles.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        self.topological_order()

    def __repr__(self) -> str:
        source = self.source.name if self.source is not None else None
        target = self.target.name if self.target is not None else None
        return (
            f"AbstractGraph(vertices={len(self._vertices)}, edges={len(self._edges)}, "
            f"source={source!r}, target={target!r})"
        )

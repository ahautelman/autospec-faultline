"""Compiler from a SpectRL AST to an abstract graph.

Recursive translation following paper section 2.3: ``G = abstract_graph(phi)``,
with the guarantee that a trajectory satisfies ``phi`` if and only if it
satisfies ``G``. Each ``achieve`` becomes a target vertex and its reach-avoid
edge; sequences chain their subgraphs; choices fan out through a fresh branch
vertex and merge back; ``ensuring`` intersects its safety predicate into every
edge its body introduced.

The compiled graph always has a unique source (labelled with the initial-state
support) and a unique sink, which the refinement operators rely on.
"""

from __future__ import annotations

from faultline.graph.abstract_graph import AbstractGraph, Vertex
from faultline.spectrl.ast import Achieve, Choice, Ensuring, Seq, Spec
from faultline.spectrl.predicates import And, Predicate, Universe


def compile_spec(spec: Spec, source_region: Predicate | None = None) -> AbstractGraph:
    """Compile a SpectRL specification into an abstract graph.

    Args:
        spec: The specification AST to compile.
        source_region: Predicate for the graph's source vertex, normally the
            support of the initial-state distribution; defaults to the whole
            state space.

    How:
        Builds an empty graph, recursively translates ``spec`` into vertices and
        reach-avoid edges, relabels the outermost source with ``source_region``,
        and then checks the DAG and single-source/single-sink invariants.

    Returns:
        The compiled :class:`~faultline.graph.abstract_graph.AbstractGraph`.

    Raises:
        ValueError: If the compiled graph is cyclic or lacks a unique source or
            sink, which would indicate a compiler bug.
        TypeError: If ``spec`` is not a known AST node.
    """
    graph = AbstractGraph()
    start, end = _compile(graph, spec)
    start.region = source_region if source_region is not None else Universe()
    graph.source = start
    graph.target = end
    graph.assert_acyclic()
    if graph.sources() != (start,):
        raise ValueError("Compiled graph does not have a unique source vertex")
    if graph.sinks() != (end,):
        raise ValueError("Compiled graph does not have a unique sink vertex")
    return graph


def _compile(graph: AbstractGraph, spec: Spec) -> tuple[Vertex, Vertex]:
    """Compile one AST node into a fragment of ``graph``.

    Args:
        graph: The graph being built; edited in place.
        spec: The AST node to compile.

    How:
        Dispatches on the node type; every branch returns the fragment's source
        and sink vertices so the caller can chain or merge fragments.

    Returns:
        The ``(source, sink)`` vertex pair for the fragment.

    Raises:
        TypeError: If ``spec`` is not a known AST node.
    """
    if isinstance(spec, Achieve):
        return _compile_achieve(graph, spec)
    if isinstance(spec, Seq):
        return _compile_seq(graph, spec)
    if isinstance(spec, Choice):
        return _compile_choice(graph, spec)
    if isinstance(spec, Ensuring):
        return _compile_ensuring(graph, spec)
    raise TypeError(f"Unsupported specification node: {type(spec).__name__}")


def _compile_achieve(graph: AbstractGraph, spec: Achieve) -> tuple[Vertex, Vertex]:
    """Compile ``achieve b`` into a target vertex and one reach-avoid edge.

    Args:
        graph: The graph being built; edited in place.
        spec: The ``achieve`` node, whose predicate labels the target vertex.

    How:
        Creates a source vertex labelled with the universe (so its region can be
        set by an enclosing composition or by the compiler's top level) and a
        target vertex labelled with the predicate, then connects them with an
        unconstrained reach-avoid edge.

    Returns:
        The ``(source, target)`` vertex pair.
    """
    start = graph.fresh_vertex("ach", Universe())
    end = graph.fresh_vertex("goal", spec.predicate)
    graph.add_edge(start, end)
    return start, end


def _compile_seq(graph: AbstractGraph, spec: Seq) -> tuple[Vertex, Vertex]:
    """Compile a sequence by chaining its steps' fragments in order.

    Args:
        graph: The graph being built; edited in place.
        spec: The ``Seq`` node.

    How:
        Compiles the first step, then for each later step adds a bridge edge
        from the running sink to the new fragment's source and advances the
        sink. The bridge edge carries no safety constraint.

    Returns:
        The ``(source, sink)`` vertex pair spanning all steps.
    """
    start, end = _compile(graph, spec.steps[0])
    for step in spec.steps[1:]:
        next_start, next_end = _compile(graph, step)
        graph.add_edge(end, next_start)
        end = next_end
    return start, end


def _compile_choice(graph: AbstractGraph, spec: Choice) -> tuple[Vertex, Vertex]:
    """Compile a choice into parallel branches between a shared pair of vertices.

    Args:
        graph: The graph being built; edited in place.
        spec: The ``Choice`` node.

    How:
        Creates one fan-out vertex and one merge vertex, then compiles each
        branch and wires it between them, so a satisfying trajectory may take
        any single branch.

    Returns:
        The ``(fan-out, merge)`` vertex pair.
    """
    start = graph.fresh_vertex("choice", Universe())
    end = graph.fresh_vertex("merge", Universe())
    for branch in spec.branches:
        branch_start, branch_end = _compile(graph, branch)
        graph.add_edge(start, branch_start)
        graph.add_edge(branch_end, end)
    return start, end


def _compile_ensuring(graph: AbstractGraph, spec: Ensuring) -> tuple[Vertex, Vertex]:
    """Compile ``body ensuring b`` by tightening the body's edges.

    Args:
        graph: The graph being built; edited in place.
        spec: The ``Ensuring`` node.

    How:
        Notes how many edges exist, compiles the body, then intersects the
        safety predicate into only the edges the body added, leaving any edges
        from surrounding context untouched.

    Returns:
        The body fragment's ``(source, sink)`` vertex pair.
    """
    edge_count_before = len(graph.edges)
    start, end = _compile(graph, spec.body)
    for edge in graph.edges[edge_count_before:]:
        edge.safe = And(edge.safe, spec.predicate)
    return start, end

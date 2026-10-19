"""Sample trajectories that follow a graph's routes, for language-inclusion checks."""

from __future__ import annotations

import numpy as np

from faultline.geometry import region_bounds, sample_within
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import PLANAR_DIMS

DEFAULT_STEPS_PER_LEG = 6


def routes(graph: AbstractGraph) -> list[list]:
    """Return every source-to-target vertex path of a graph.

    Args:
        graph: A graph with source and target set.

    How:
        Depth-first enumeration; graphs here are small DAGs.

    Returns:
        The paths, each a list of vertices.
    """
    found: list[list] = []

    def walk(vertex, path):
        path = path + [vertex]
        if vertex is graph.target:
            found.append(path)
            return
        for edge in graph.successors(vertex):
            walk(edge.target, path)

    walk(graph.source, [])
    return found


def sample_trajectories(
    graph: AbstractGraph,
    rng: np.random.Generator,
    count: int,
    world_size: float,
    steps_per_leg: int = DEFAULT_STEPS_PER_LEG,
    detour_probability: float = 0.5,
) -> list[np.ndarray]:
    """Sample piecewise-linear trajectories that visit a route's regions in order.

    Args:
        graph: The graph whose routes to follow.
        rng: Seeded generator.
        count: Number of trajectories.
        world_size: Side of the square world, for detour points.
        steps_per_leg: States per straight leg.
        detour_probability: Chance of a random detour point between two vertices.

    How:
        Picks a route, draws one point inside each vertex region, optionally adds
        a random detour point between consecutive regions, and joins the points
        with straight legs. Most samples will not satisfy a graph with traps; the
        caller keeps those that do. Satisfying trajectories are exactly what an
        inclusion check needs.

    Returns:
        ``count`` arrays of shape ``(T, 2)``.
    """
    paths = routes(graph)
    samples = []
    for _ in range(count):
        path = paths[int(rng.integers(len(paths)))]
        points = []
        for vertex in path:
            bounds = region_bounds(vertex.region, PLANAR_DIMS)
            lower, upper = bounds if bounds is not None else (np.zeros(2), np.full(2, world_size))
            point = sample_within(vertex.region, lower, upper, rng)
            points.append(point if point is not None else (lower + upper) / 2.0)
            if vertex is not path[-1] and rng.random() < detour_probability:
                points.append(rng.uniform(0.0, world_size, size=2))
        legs = [
            np.linspace(a, b, steps_per_leg, endpoint=False)
            for a, b in zip(points[:-1], points[1:])
        ]
        samples.append(np.vstack(legs + [points[-1][None, :]]) if legs else np.array(points))
    return samples

"""Reference repairs against which AutoSpec's recovery is judged (spec section 8).

Each baseline maps a mutant to a repaired graph without using the refinement result:

* ``unrefined``: the faulty graph itself. Lower anchor: any metric that rewards it is broken.
* ``oracle``: the operator-ceiling repair (region families) or the gold graph (structural
  families). Upper anchor for what the operators can do with perfect witnesses.
* ``trivializer``: shrinks the source and target to tiny boxes inside gold. It games the
  loop's own success estimate (small regions are easy) while destroying the spec, so
  it must score high on the proxy and low on retention and S*. If a metric rates it well,
  the metric rewards the wrong thing.
* ``random_operator``: Algorithm 1 with the operator order shuffled, so the paper's fixed
  order and the family-to-operator taxonomy are not privileged. It is the only baseline
  that runs the loop.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from unittest import mock

import numpy as np

from faultline.faults.graph_ops import find_edge
from faultline.faults.ground_truth import FaultKind, GroundTruthRepair
from faultline.golden.raster import CellGrid
from faultline.graph import AbstractGraph
from faultline.metrics.ceiling import with_ceiling
from faultline.metrics.regions import locus_region
from faultline.refinement import AutoSpecConfig, AutoSpecRun, run_autospec
from faultline.refinement import loop as loop_module
from faultline.spectrl.predicates import Box, Predicate

TRIVIAL_RADIUS = 0.5


def unrefined(repair: GroundTruthRepair) -> AbstractGraph:
    """Return a copy of the faulty graph."""
    return repair.faulty.copy()


def _set_region(graph: AbstractGraph, locus, region: Predicate) -> None:
    if locus.kind == "vertex":
        for vertex in graph.vertices:
            if vertex.name == locus.name:
                vertex.region = region
    elif locus.kind == "edge":
        source, target = locus.name.split("->")
        edge = find_edge(graph, source, target)
        if edge is not None:
            edge.safe = region


def oracle(repair: GroundTruthRepair, grid: CellGrid) -> AbstractGraph:
    """Return the repair an operator reaches with perfect witnesses.

    Args:
        repair: The mutant.
        grid: The shared cell grid.

    How:
        Region families (M1 to M3) take their ceiling region at the locus. Structural
        families (M4, M5) and the null have no region ceiling, so the oracle is the gold
        graph, the best any operator could aim for.

    Returns:
        A new graph.
    """
    ceiling = with_ceiling(repair, grid).ceiling
    if not ceiling:
        return repair.gold.copy()
    graph = repair.faulty.copy()
    for locus, region in ceiling.items():
        _set_region(graph, locus, region)
    return graph


def _centroid_box(region: Predicate, grid: CellGrid, radius: float) -> Predicate:
    cells = np.argwhere(grid.mask(region))
    if len(cells) == 0:
        return region
    centre = np.mean([grid.centre(int(i), int(j)) for i, j in cells], axis=0)
    # snap to the nearest cell of the region so the box is never empty
    nearest = min(cells, key=lambda c: float(np.linalg.norm(grid.centre(int(c[0]), int(c[1])) - centre)))
    point = grid.centre(int(nearest[0]), int(nearest[1]))
    return Box(tuple(point - radius), tuple(point + radius))


def trivializer(repair: GroundTruthRepair, grid: CellGrid, radius: float = TRIVIAL_RADIUS) -> AbstractGraph:
    """Return the faulty graph with source and target shrunk to tiny boxes inside gold.

    Args:
        repair: The mutant.
        grid: The shared cell grid.
        radius: Half-side of the boxes.

    How:
        Each box is centred on the gold cell nearest the gold region's centroid, so
        it is correct but nearly empty. Edges are trained to reach a point of the tiny
        target, which is easy, so the proxy success rises while almost all of gold's
        behaviour is lost.

    Returns:
        A new graph.
    """
    graph = repair.faulty.copy()
    graph.source.region = _centroid_box(repair.gold.source.region, grid, radius)
    graph.target.region = _centroid_box(repair.gold.target.region, grid, radius)
    return graph


@contextmanager
def _shuffled_operators(rng: np.random.Generator) -> Iterator[None]:
    builders = list(loop_module._CANDIDATE_BUILDERS)
    order = [builders[i] for i in rng.permutation(len(builders))]
    with mock.patch.object(loop_module, "_CANDIDATE_BUILDERS", tuple(order)):
        yield


def random_operator(
    graph: AbstractGraph,
    env: Any,
    learner: Any,
    config: AutoSpecConfig,
    rng: np.random.Generator,
    graph_learner: Any | None = None,
) -> AutoSpecRun:
    """Run Algorithm 1 with a random operator order.

    Args:
        graph: The faulty graph.
        env: The environment.
        learner: The edge learner.
        config: The run configuration.
        rng: Seeded generator choosing the order (one permutation per run).
        graph_learner: Optional graph learner.

    How:
        Temporarily replaces the loop's fixed operator tuple with a permutation of it. The
        loop code is untouched, so everything else (commit gate, relearning) is identical
        to the real run. Not safe for concurrent runs in one process; use one process per run.

    Returns:
        The run.
    """
    with _shuffled_operators(rng):
        return run_autospec(graph, env, learner, config, graph_learner)


@dataclass(frozen=True)
class Baselines:
    """The graph-valued baselines of one mutant."""

    unrefined: AbstractGraph
    oracle: AbstractGraph
    trivializer: AbstractGraph


def graph_baselines(repair: GroundTruthRepair, grid: CellGrid) -> Baselines:
    """Build the three baselines that need no learner."""
    return Baselines(unrefined(repair), oracle(repair, grid), trivializer(repair, grid))

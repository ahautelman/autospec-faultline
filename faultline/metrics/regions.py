"""Region-level recovery metrics on the shared cell grid.

Every metric compares three regions at the fault's locus: gold R*, faulty R_f and
repaired R_r. All are rasterised on the protocol's grid, so areas are cell counts and
the discretisation error is one cell of boundary (reported with the resolution).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from faultline.faults.graph_ops import find_edge
from faultline.faults.ground_truth import Locus
from faultline.golden.raster import CellGrid
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import Predicate


@dataclass(frozen=True)
class RegionScores:
    """Recovery of one region.

    Args:
        removal: Share of the fault excess ``R_f \\ R*`` that the repair removed, or
            ``None`` when the fault has no excess on this grid.
        retention: Share of gold ``R*`` the repair kept (collateral = ``1 - retention``).
            This is also the semantic recall.
        precision: Share of the repaired region that lies in gold.
        iou: Intersection over union of repaired and gold.
        excess_cells: Cells of fault excess.
        gold_cells: Cells of gold.
    """

    removal: float | None
    retention: float | None
    precision: float | None
    iou: float | None
    excess_cells: int
    gold_cells: int


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def region_scores(
    gold: np.ndarray, faulty: np.ndarray, repaired: np.ndarray
) -> RegionScores:
    """Score a repaired mask against gold and faulty masks.

    Args:
        gold: Mask of R*.
        faulty: Mask of R_f.
        repaired: Mask of R_r.

    How:
        Removal counts excess cells absent from the repair, retention counts gold cells
        present in it. A repair that deletes the whole region scores removal 1 and
        retention 0, so removal alone can never be gamed.

    Returns:
        The scores.
    """
    excess = faulty & ~gold
    kept = repaired & gold
    union = int((repaired | gold).sum())
    return RegionScores(
        removal=_ratio(int((excess & ~repaired).sum()), int(excess.sum())),
        retention=_ratio(int(kept.sum()), int(gold.sum())),
        precision=_ratio(int(kept.sum()), int(repaired.sum())),
        iou=_ratio(int(kept.sum()), union),
        excess_cells=int(excess.sum()),
        gold_cells=int(gold.sum()),
    )


def locus_region(graph: AbstractGraph, locus: Locus, reference: AbstractGraph | None = None) -> Predicate | None:
    """Return the region a graph holds at a locus, or ``None`` when it has none.

    Args:
        graph: The graph to read.
        locus: A vertex or edge locus.
        reference: The faulty graph. When given, a vertex locus whose named vertex lost
            its role to a fresh entry vertex (PastRefine at an inner vertex) reads the
            fresh vertex that feeds the same successors.

    How:
        Vertices are matched by name, edges by endpoint names; the operators keep names.
        The fallback exists only for the entry vertex PastRefine adds.

    Returns:
        The region (vertex label or edge safe set).
    """
    if locus.kind == "vertex":
        for vertex in graph.vertices:
            if vertex.name == locus.name:
                return vertex.region
        return None
    if locus.kind == "edge":
        source, target = locus.name.split("->")
        edge = find_edge(graph, source, target)
        return None if edge is None else edge.safe
    return None


def score_locus(
    grid: CellGrid, gold: Predicate, faulty: Predicate, repaired: Predicate | None
) -> RegionScores | None:
    """Rasterise three regions and score them; ``None`` when the repair lacks the locus."""
    if repaired is None:
        return None
    return region_scores(grid.mask(gold), grid.mask(faulty), grid.mask(repaired))


def score_loci(
    grid: CellGrid,
    targets: Mapping[Locus, Predicate],
    faulty: AbstractGraph,
    repaired: AbstractGraph,
) -> dict[Locus, RegionScores | None]:
    """Score every locus of a mutant against a repaired graph."""
    out: dict[Locus, RegionScores | None] = {}
    for locus, gold_region in targets.items():
        faulty_region = locus_region(faulty, locus)
        out[locus] = (
            None if faulty_region is None
            else score_locus(grid, gold_region, faulty_region, locus_region(repaired, locus))
        )
    return out

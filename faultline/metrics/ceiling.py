"""Operator ceiling: the best repair an operator's geometry can reach with oracle witnesses.

Operators repair a region from sampled witnesses. With perfect witnesses (every cell of
the right class) the repair is limited only by the operator's shape: a convex hull cannot
carve a non-convex gold region, one half-plane cannot cut two pockets. The ceiling
separates that structural limit from the learner's sampling noise, and the attainment
ratio says how much of the reachable improvement a run achieved.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from faultline.faults.ground_truth import FaultKind, GroundTruthRepair, Locus
from faultline.geometry import convex_hull, half_plane_partition
from faultline.golden.raster import CellGrid
from faultline.metrics.regions import RegionScores, locus_region, region_scores
from faultline.spectrl.predicates import And, Not, Predicate

MAX_POINTS = 4000


def _centres(grid: CellGrid, mask: np.ndarray) -> list[np.ndarray]:
    cells = np.argwhere(mask)
    if len(cells) > MAX_POINTS:
        cells = cells[:: int(np.ceil(len(cells) / MAX_POINTS))]
    return [grid.centre(int(ix), int(iy)) for ix, iy in cells]


def ceiling_region(
    kind: FaultKind, grid: CellGrid, gold: Predicate, faulty: Predicate
) -> Predicate | None:
    """Return the region the family's expected operator reaches with oracle witnesses.

    Args:
        kind: The mutation family.
        grid: The shared cell grid.
        gold: Gold region at the locus.
        faulty: Faulty region at the locus.

    How:
        M1 (ReachRefine): hull of every gold cell, intersected with the faulty region.
        M2 (AvoidRefine): the faulty safe set minus the hull of the unlabelled-trap cells.
        M3 (PastRefine): the faulty region cut by the half-plane separating gold cells
        from pocket cells; a pocket that no half-plane separates leaves the region as is.
        M4 and M5 repair structure, not a region, so they have no region ceiling.

    Returns:
        The ceiling predicate, or ``None`` when the family has no region ceiling.
    """
    g, f = grid.mask(gold), grid.mask(faulty)
    if kind is FaultKind.GOAL_INFLATION:
        return And(convex_hull(_centres(grid, g)), faulty)
    if kind is FaultKind.HAZARD_UNLABELED:
        return And(faulty, Not(convex_hull(_centres(grid, f & ~g))))
    if kind is FaultKind.DEAD_START:
        plane = half_plane_partition(_centres(grid, g), _centres(grid, f & ~g))
        return faulty if plane is None else And(faulty, plane)
    return None


def with_ceiling(repair: GroundTruthRepair, grid: CellGrid) -> GroundTruthRepair:
    """Return the mutant with its ``ceiling`` filled in."""
    out: dict[Locus, Predicate] = {}
    for locus, gold_region in repair.target.items():
        faulty_region = locus_region(repair.faulty, locus)
        if faulty_region is None:
            continue
        region = ceiling_region(repair.kind, grid, gold_region, faulty_region)
        if region is not None:
            out[locus] = region
    return replace(repair, ceiling=out or None)


def attainment(repaired: RegionScores | None, ceiling: RegionScores | None) -> float | None:
    """Return the share of the ceiling's fault removal a repair achieved.

    Args:
        repaired: Scores of the repair.
        ceiling: Scores of the oracle-witness repair.

    How:
        Removal over ceiling removal, clipped to ``[0, 1]``. ``None`` when either is
        missing or the ceiling removes nothing: a fault the operator cannot repair even
        with oracle witnesses is not held against the run.

    Returns:
        The attainment ratio.
    """
    if repaired is None or ceiling is None or not ceiling.removal or repaired.removal is None:
        return None
    return float(min(1.0, max(0.0, repaired.removal / ceiling.removal)))


def ceiling_scores(repair: GroundTruthRepair, grid: CellGrid) -> dict[Locus, RegionScores]:
    """Score each locus's ceiling region against gold and faulty."""
    scores = {}
    for locus, region in (repair.ceiling or {}).items():
        faulty_region = locus_region(repair.faulty, locus)
        scores[locus] = region_scores(
            grid.mask(repair.target[locus]), grid.mask(faulty_region), grid.mask(region)
        )
    return scores

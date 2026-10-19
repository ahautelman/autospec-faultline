"""Region-aware graph edit distance between a gold graph and a repaired graph."""

from __future__ import annotations

from faultline.golden.raster import CellGrid
from faultline.graph import AbstractGraph


def _iou(a, b) -> float:
    union = int((a | b).sum())
    return 1.0 if union == 0 else int((a & b).sum()) / union


def region_ged(grid: CellGrid, gold: AbstractGraph, repaired: AbstractGraph) -> float:
    """Return the normalised, region-aware edit distance in ``[0, 1]``.

    Args:
        grid: The shared cell grid.
        gold: The gold graph.
        repaired: The repaired graph.

    How:
        Vertices match by name and edges by endpoint names (edges between the same pair
        are matched in order). A matched vertex costs ``1 - IoU`` of its labels, a
        matched edge ``1 - IoU`` of its safe sets, and an unmatched element on either
        side costs 1. The total is divided by the number of elements in the union of
        both graphs, so identical graphs score 0 and disjoint ones 1. Unlike a purely
        structural distance, a region repaired most of the way scores near zero.

    Returns:
        The normalised distance.
    """
    cost = 0.0
    names = {v.name: v for v in gold.vertices}
    other = {v.name: v for v in repaired.vertices}
    for name in names.keys() | other.keys():
        if name in names and name in other:
            cost += 1.0 - _iou(grid.mask(names[name].region), grid.mask(other[name].region))
        else:
            cost += 1.0
    total = len(names.keys() | other.keys())

    def grouped(graph):
        out: dict[tuple[str, str], list] = {}
        for e in graph.edges:
            out.setdefault((e.source.name, e.target.name), []).append(e)
        return out

    left, right = grouped(gold), grouped(repaired)
    for key in left.keys() | right.keys():
        a, b = left.get(key, []), right.get(key, [])
        pairs = min(len(a), len(b))
        for x, y in zip(a, b):
            cost += 1.0 - _iou(grid.mask(x.safe), grid.mask(y.safe))
        cost += abs(len(a) - len(b))
        total += max(len(a), len(b))
    return cost / total if total else 0.0

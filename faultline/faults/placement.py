"""Find axis-aligned rectangles of cells inside an allowed mask."""

from __future__ import annotations

import numpy as np

from faultline.golden.raster import CellGrid, Mask, Rectangle
from faultline.spectrl.predicates import Box

AREA_TOLERANCES = (0.1, 0.25, 0.5)
MAX_ASPECT = 4.0


def _shapes(area: int, limit: int, tolerance: float) -> list[tuple[int, int]]:
    """Return ``(width, height)`` shapes with about ``area`` cells and a sane aspect."""
    shapes = set()
    for height in range(1, min(area, limit) + 1):
        width = max(1, round(area / height))
        if width > limit:
            continue
        if abs(width * height - area) > tolerance * area:
            continue
        if max(width, height) / min(width, height) > MAX_ASPECT:
            continue
        shapes.add((width, height))
    return sorted(shapes)


def find_rectangle(
    allowed: Mask, area: int, anchor: Box | None, grid: CellGrid, rng: np.random.Generator
) -> Rectangle | None:
    """Return the allowed rectangle of about ``area`` cells nearest to ``anchor``.

    Args:
        allowed: Cells the rectangle may cover.
        area: Target area in cells.
        anchor: A box the rectangle should sit close to, or ``None`` for no preference.
        grid: The cell grid, for geometry.
        rng: Breaks exact ties, so the choice is deterministic given the seed.

    How:
        Tries every shape within 10% of the target area and an aspect ratio of at
        most 4, widening to 25% and then 50% only if nothing fits. A summed-area table finds every position where all cells are
        allowed. The winner minimises the gap to the anchor; remaining ties are
        broken by the generator.

    Returns:
        ``(ix0, iy0, ix1, iy1)`` inclusive, or ``None`` when nothing fits.
    """
    n = allowed.shape[0]
    table = np.zeros((n + 1, n + 1), dtype=np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(allowed.astype(np.int64), axis=0), axis=1)
    best_gap, candidates = np.inf, []
    for tolerance in AREA_TOLERANCES:
        for width, height in _shapes(max(1, int(area)), n, tolerance):
            sums = (
                table[width:, height:]
                - table[:-width, height:]
                - table[width:, :-height]
                + table[:-width, :-height]
            )
            for ix, iy in np.argwhere(sums == width * height):
                gap = _gap(anchor, grid, ix, iy, width, height)
                if gap < best_gap - 1e-9:
                    best_gap, candidates = gap, []
                if gap <= best_gap + 1e-9:
                    candidates.append((int(ix), int(iy), width, height))
        if candidates:
            break
    if not candidates:
        return None
    ix, iy, width, height = candidates[int(rng.integers(len(candidates)))]
    return ix, iy, ix + width - 1, iy + height - 1


def _gap(anchor: Box | None, grid: CellGrid, ix: int, iy: int, width: int, height: int) -> float:
    """Return the distance between a rectangle and the anchor box (0 if they touch)."""
    if anchor is None:
        return 0.0
    cell = grid.cell
    gap_x = max(0.0, anchor.lower[0] - (ix + width) * cell, ix * cell - anchor.upper[0])
    gap_y = max(0.0, anchor.lower[1] - (iy + height) * cell, iy * cell - anchor.upper[1])
    return float(np.hypot(gap_x, gap_y))

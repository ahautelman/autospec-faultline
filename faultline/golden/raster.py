"""Rasterised cell-grid analysis of a rooms layout, for regions finer than a room.

``RoomGrid`` answers topology questions per room. A trap or a refined region that
sits inside one room is invisible to it, so the gold builder works on this cell
grid instead: the world is cut into square cells, two neighbouring cells are
connected when the straight move between their centres crosses no solid wall, and
regions become boolean masks. Reachability and viability are then breadth-first
searches over masks, the same two queries ``RoomGrid`` answers at room level.

Cell connectivity is the 4-neighbourhood, an approximation of the point-mass
dynamics, which also move diagonally. The error is bounded by the cell size and
is reported with every certificate through the protocol's resolution.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from faultline.envs.rooms.layout import RoomsLayout
from faultline.spectrl.predicates import Box, Predicate

Mask = np.ndarray
Rectangle = tuple[int, int, int, int]


class CellGrid:
    """A square-cell rasterisation of a :class:`RoomsLayout`.

    Args:
        layout: The wall and door model.
        resolution: Cells per axis. Must be a multiple of the rooms per side, so
            every wall lies on a cell boundary.

    How:
        Cell ``(ix, iy)`` covers ``[ix*h, (ix+1)*h) x [iy*h, (iy+1)*h)`` with
        ``h = world_size / resolution`` and is represented by its centre. Moves
        that stay inside a room are always open. Only moves that cross a wall
        line are tested against the layout, which keeps construction cheap.

    Raises:
        ValueError: If ``resolution`` is not a positive multiple of the grid size.
    """

    def __init__(self, layout: RoomsLayout, resolution: int) -> None:
        n = layout.rooms_per_side
        if resolution < n or resolution % n != 0:
            raise ValueError(
                f"resolution {resolution} must be a positive multiple of rooms_per_side {n}"
            )
        self.layout = layout
        self.resolution = int(resolution)
        self.cell = layout.world_size / self.resolution
        self._per_room = self.resolution // n
        self._right = self._open_moves(axis=0)
        self._up = self._open_moves(axis=1)

    def centre(self, ix: int, iy: int) -> np.ndarray:
        """Return the centre of a cell.

        Args:
            ix: Cell column.
            iy: Cell row.

        How:
            Offsets half a cell from the cell's lower corner.

        Returns:
            The centre ``(x, y)``.
        """
        return np.array([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell])

    def _open_moves(self, axis: int) -> Mask:
        """Return which moves in the positive direction of ``axis`` are open.

        Args:
            axis: ``0`` for moves in ``+x``, ``1`` for moves in ``+y``.

        How:
            A move from a cell to its upper neighbour crosses a wall line exactly
            when the neighbour index is a multiple of the room width in cells.
            Those moves ask the layout whether a door covers the crossing; all
            others are open. The last row or column has no neighbour and is
            marked closed.

        Returns:
            A boolean array indexed by the move's source cell.
        """
        n = self.resolution
        moves = np.zeros((n, n), dtype=bool)
        for ix in range(n):
            for iy in range(n):
                index = (ix, iy)[axis] + 1
                if index >= n:
                    continue
                if index % self._per_room != 0:
                    moves[ix, iy] = True
                    continue
                neighbour = (ix + 1, iy) if axis == 0 else (ix, iy + 1)
                moves[ix, iy] = not self.layout.segment_is_blocked(
                    self.centre(ix, iy), self.centre(*neighbour)
                )
        return moves

    def mask(self, region: Predicate) -> Mask:
        """Return the cells whose centre lies in a predicate.

        Args:
            region: The predicate over ``(x, y)``.

        How:
            Evaluates the predicate at every cell centre.

        Returns:
            A boolean array indexed by ``(ix, iy)``.
        """
        n = self.resolution
        result = np.zeros((n, n), dtype=bool)
        for ix in range(n):
            for iy in range(n):
                result[ix, iy] = region.evaluate(self.centre(ix, iy))
        return result

    def neighbours(self, ix: int, iy: int):
        """Yield the cells connected to a cell by an open move.

        Args:
            ix: Cell column.
            iy: Cell row.

        How:
            Combines the stored forward moves with their mirror images, since a
            wall blocks both directions equally.

        Yields:
            Neighbouring ``(ix, iy)`` pairs.
        """
        n = self.resolution
        if ix + 1 < n and self._right[ix, iy]:
            yield ix + 1, iy
        if ix > 0 and self._right[ix - 1, iy]:
            yield ix - 1, iy
        if iy + 1 < n and self._up[ix, iy]:
            yield ix, iy + 1
        if iy > 0 and self._up[ix, iy - 1]:
            yield ix, iy - 1

    def reachable(self, sources: Mask, safe: Mask) -> Mask:
        """Return the cells reachable from ``sources`` through safe cells.

        Args:
            sources: Start cells, always included.
            safe: Cells the route may pass through.

        How:
            Breadth-first search that expands only into safe cells. Sources are
            included even when unsafe, matching ``RoomGrid.reachable``.

        Returns:
            A boolean mask of the reached cells.
        """
        reached = sources.copy()
        queue = deque(map(tuple, np.argwhere(sources)))
        while queue:
            ix, iy = queue.popleft()
            for nx, ny in self.neighbours(ix, iy):
                if safe[nx, ny] and not reached[nx, ny]:
                    reached[nx, ny] = True
                    queue.append((nx, ny))
        return reached

    def viability(self, targets: Mask, safe: Mask) -> Mask:
        """Return the cells with a safe route to ``targets``.

        Args:
            targets: Goal cells, always included.
            safe: Cells the route may pass through.

        How:
            Doors are symmetric, so this is the same search as :meth:`reachable`
            run from the targets.

        Returns:
            A boolean mask of the viable cells.
        """
        return self.reachable(targets, safe)

    def box_of(self, rectangle: Rectangle) -> Box:
        """Return the box covering an inclusive cell rectangle.

        Args:
            rectangle: ``(ix0, iy0, ix1, iy1)``, inclusive on both ends.

        How:
            Converts cell indices to the outer corners of the rectangle.

        Returns:
            The box predicate.
        """
        ix0, iy0, ix1, iy1 = rectangle
        return Box(
            (ix0 * self.cell, iy0 * self.cell),
            ((ix1 + 1) * self.cell, (iy1 + 1) * self.cell),
        )


def largest_rectangle(mask: Mask) -> Rectangle | None:
    """Return the largest axis-aligned rectangle of true cells.

    Args:
        mask: Boolean array indexed by ``(ix, iy)``.

    How:
        The classic histogram method: for each column ``ix``, stack the heights
        of consecutive true cells along ``iy``, then find the largest rectangle
        in each histogram with a monotonic stack. Ties keep the first found, so
        the result is deterministic.

    Returns:
        ``(ix0, iy0, ix1, iy1)`` inclusive, or ``None`` when the mask is empty.
    """
    n_x, n_y = mask.shape
    heights = np.zeros(n_y, dtype=int)
    best_area, best = 0, None
    for ix in range(n_x):
        heights = np.where(mask[ix], heights + 1, 0)
        stack: list[int] = []
        for iy in range(n_y + 1):
            current = heights[iy] if iy < n_y else 0
            while stack and heights[stack[-1]] >= current:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = height * (iy - left)
                if area > best_area:
                    best_area = area
                    best = (ix - height + 1, left, ix, iy - 1)
            stack.append(iy)
    return best

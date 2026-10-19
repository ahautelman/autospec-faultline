"""Rectangular checkpoint regions on a room grid.

The paper's n-rooms regions are 2x2 blocks of contiguous rooms. A block is
identified by the bottom-left room ``(col, row)``; its four rooms are
``(col, row)``, ``(col + 1, row)``, ``(col, row + 1)`` and
``(col + 1, row + 1)``. The block maps to a continuous axis-aligned box for the
specification, with an optional inset from its room walls.
"""

from __future__ import annotations

from dataclasses import dataclass

from faultline.envs.rooms.layout import DEFAULT_ROOM_SIZE, WallKey
from faultline.spectrl.predicates import Box

RoomKey = tuple[int, int]

# A region is a square block of this many rooms per side.
BLOCK_SIDE = 2


@dataclass(frozen=True, order=True)
class RegionBlock:
    """A 2x2 block of rooms, identified by its bottom-left room.

    Args:
        col: Column index of the block's bottom-left room.
        row: Row index of the block's bottom-left room.

    How:
        Holds only the grid indices. :meth:`rooms` and :meth:`to_box` derive the
        room set and the continuous region, so no geometry is stored twice.
    """

    col: int
    row: int

    def rooms(self) -> tuple[RoomKey, ...]:
        """Return the block's four rooms in a fixed order.

        How:
            Bottom-left, bottom-right, top-left, top-right. The order is stable
            so deterministic algorithms can rely on it.

        Returns:
            The four ``(col, row)`` room keys.
        """
        return (
            (self.col, self.row),
            (self.col + 1, self.row),
            (self.col, self.row + 1),
            (self.col + 1, self.row + 1),
        )

    def to_box(self, room_size: float = DEFAULT_ROOM_SIZE, inset: float = 0.0) -> Box:
        """Return the continuous box covering the block.

        Args:
            room_size: Width and height of one room.
            inset: Distance to pull each side inward from the block's walls.
                The paper's 100-rooms regions use the whole block (``0``); the
                9-rooms regions use a one-unit inset.

        How:
            The block spans two rooms from its bottom-left corner, so its box is
            ``[col*size + inset, (col + 2)*size - inset]`` by the same formula
            on the row axis.

        Returns:
            The axis-aligned box predicate over the planar coordinates.
        """
        low = (self.col * room_size + inset, self.row * room_size + inset)
        high = ((self.col + 2) * room_size - inset, (self.row + 2) * room_size - inset)
        return Box(low, high)

    def overlaps(self, other: RegionBlock) -> bool:
        """Return whether two blocks share at least one room.

        Args:
            other: The block to compare with.

        How:
            Two blocks overlap unless one lies entirely left of or below the
            other. Touching blocks (sharing only a wall) do not overlap.

        Returns:
            ``True`` when the room sets intersect.
        """
        return not (
            self.col + 1 < other.col
            or other.col + 1 < self.col
            or self.row + 1 < other.row
            or other.row + 1 < self.row
        )


def valid_blocks(rooms_per_side: int) -> tuple[RegionBlock, ...]:
    """Return every 2x2 block position on an ``n x n`` room grid.

    Args:
        rooms_per_side: Number of rooms along each axis; must be at least 2.

    How:
        Enumerates bottom-left rooms from ``(0, 0)`` through
        ``(n - 2, n - 2)`` in row-major order (row outer, column inner). The
        order is fixed so a sampler's index-to-block mapping is deterministic.

    Returns:
        The ``(n - 1)^2`` blocks.

    Raises:
        ValueError: If ``rooms_per_side`` is below 2.
    """
    if rooms_per_side < BLOCK_SIDE:
        raise ValueError(f"A 2x2 block needs at least 2 rooms per side, got {rooms_per_side}")
    return tuple(
        RegionBlock(col, row)
        for row in range(rooms_per_side - 1)
        for col in range(rooms_per_side - 1)
    )


def block_walls(block: RegionBlock, room_size: float = DEFAULT_ROOM_SIZE) -> dict[WallKey, tuple[float, float]]:
    """Return the four full-edge walls that bound a block from the outside.

    Args:
        block: The block whose boundary walls are wanted.
        room_size: Width and height of one room.

    How:
        Lists the left, right, bottom and top walls of the 2x2 block, each as a
        full-edge interval. A wall shared by two blocks appears in both blocks'
        lists. The helper is useful when a caller needs to seal or inspect a
        region boundary; the generator does not use it.

    Returns:
        A mapping from wall key to the full-edge interval on that wall.
    """
    col, row = block.col, block.row
    low_x, high_x = col * room_size, (col + 2) * room_size
    low_y, high_y = row * room_size, (row + 2) * room_size
    return {
        ("v", col - 1, row): (low_y, high_y),
        ("v", col + 1, row): (low_y, high_y),
        ("h", col, row - 1): (low_x, high_x),
        ("h", col, row + 1): (low_x, high_x),
    }

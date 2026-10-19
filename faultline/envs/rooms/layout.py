"""Generic wall, door, and region model for a square room grid.

The world is an ``n x n`` grid of square rooms. Walls sit on the grid lines. A
wall is solid unless a door opening covers the point where a path crosses it.
Doors are open intervals along a wall's free coordinate.

Wall keys:

- ``("v", col, row)``: the vertical wall between room ``(col, row)`` and room
  ``(col + 1, row)``.
- ``("h", col, row)``: the horizontal wall between room ``(col, row)`` and room
  ``(col, row + 1)``.

The generic model is size-agnostic. A door map passed to :class:`RoomsLayout`
is used verbatim; an omitted door map leaves every internal wall solid. The
9-rooms corridor and the other named 9-rooms door sets live in
:mod:`faultline.envs.rooms9`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

WallKey = tuple[str, int, int]
DoorInterval = tuple[float, float]

_VERTICAL = "v"
_HORIZONTAL = "h"

# Generic defaults. The 9-rooms model uses these values, but nothing in this
# module depends on that grid being the only one.
DEFAULT_ROOM_SIZE = 8.0
DEFAULT_ROOMS_PER_SIDE = 3


def full_edge_interval(key: WallKey, room_size: float) -> DoorInterval:
    """Return the interval that opens one wall across its whole room edge.

    Args:
        key: The wall key.
        room_size: Width and height of one room.

    How:
        A vertical wall ``("v", col, row)`` runs along room row ``row``, so its
        free coordinate is ``y``; a horizontal wall ``("h", col, row)`` runs along
        room column ``col``, so its free coordinate is ``x``.

    Returns:
        The ``(low, high)`` interval of the room edge.
    """
    axis, col, row = key
    index = row if axis == _VERTICAL else col
    return (index * room_size, (index + 1) * room_size)


def internal_wall_keys(rooms_per_side: int) -> tuple[WallKey, ...]:
    """Return every internal wall of a square grid in a fixed order.

    Args:
        rooms_per_side: Number of rooms along each axis.

    How:
        Vertical walls first, row by row and column by column, then horizontal
        walls in the same order. The generator draws one random number per wall
        in this order, so the order is part of the seed-to-layout contract.

    Returns:
        The ``2 * n * (n - 1)`` wall keys.
    """
    vertical = tuple(
        (_VERTICAL, col, row)
        for row in range(rooms_per_side)
        for col in range(rooms_per_side - 1)
    )
    horizontal = tuple(
        (_HORIZONTAL, col, row)
        for row in range(rooms_per_side - 1)
        for col in range(rooms_per_side)
    )
    return vertical + horizontal


def open_doors_except_sealed_rooms(
    sealed_rooms: Iterable[tuple[int, int]],
    *,
    room_size: float = DEFAULT_ROOM_SIZE,
    rooms_per_side: int = DEFAULT_ROOMS_PER_SIDE,
) -> dict[WallKey, DoorInterval]:
    """Return a door map that opens every internal wall except sealed-room walls.

    Args:
        sealed_rooms: Grid coordinates ``(col, row)`` of the rooms to keep sealed.
        room_size: Width and height of one room.
        rooms_per_side: Number of rooms along each axis.

    How:
        Walks every internal vertical and horizontal wall. A wall is opened
        across its full room edge unless either adjacent room is sealed; a wall
        next to a sealed room is left out of the map, so it stays solid and the
        sealed rooms remain enclosed. The opening spans the whole edge rather
        than a centred gap, which merges the open rooms into one area.

    Returns:
        A mapping from wall key to the full-edge open interval.
    """
    sealed = set(sealed_rooms)
    doors: dict[WallKey, DoorInterval] = {}
    for line_index in range(1, rooms_per_side):
        for room in range(rooms_per_side):
            low = room * room_size
            high = low + room_size
            left, right = (line_index - 1, room), (line_index, room)
            if left not in sealed and right not in sealed:
                doors[("v", line_index - 1, room)] = (low, high)
            below, above = (room, line_index - 1), (room, line_index)
            if below not in sealed and above not in sealed:
                doors[("h", room, line_index - 1)] = (low, high)
    return doors


def _room_index(value: float, room_size: float, rooms_per_side: int) -> int:
    """Return the room index that contains one coordinate.

    Args:
        value: A coordinate along one axis.
        room_size: Width of one room.
        rooms_per_side: Number of rooms along the axis.

    How:
        Divides by the room size, floors to an integer, and clamps into range so
        a point exactly on the outer edge still maps to a valid room.

    Returns:
        The room index in ``[0, rooms_per_side - 1]``.
    """
    index = int(value // room_size)
    return max(0, min(rooms_per_side - 1, index))


class RoomsLayout:
    """The walls and doors of a square room grid.

    Args:
        doors: Wall key to open interval mapping. Defaults to an empty map, so
            every internal wall is solid unless the caller supplies doors.
        room_size: Width and height of one room.
        rooms_per_side: Number of rooms along each axis.

    How:
        Stores the door map and the grid size. Wall and door queries derive from
        these two values, so a caller can build a different grid by passing a
        different door map.
    """

    def __init__(
        self,
        doors: dict[WallKey, DoorInterval] | None = None,
        room_size: float = DEFAULT_ROOM_SIZE,
        rooms_per_side: int = DEFAULT_ROOMS_PER_SIDE,
    ) -> None:
        self.room_size = float(room_size)
        self.rooms_per_side = int(rooms_per_side)
        self.world_size = self.room_size * self.rooms_per_side
        self.doors = {} if doors is None else dict(doors)

    def room_of(self, point: np.ndarray) -> tuple[int, int]:
        """Return the ``(col, row)`` room that contains a point.

        Args:
            point: A position ``(x, y)``.

        How:
            Maps each coordinate to a room index with :func:`_room_index`.

        Returns:
            The column and row of the containing room.
        """
        col = _room_index(float(point[0]), self.room_size, self.rooms_per_side)
        row = _room_index(float(point[1]), self.room_size, self.rooms_per_side)
        return col, row

    def clamp(self, point: np.ndarray) -> np.ndarray:
        """Return a copy of ``point`` clipped to the world bounds.

        Args:
            point: A position ``(x, y)``.

        How:
            Clips each coordinate into ``[0, world_size]`` and returns a fresh
            float array.

        Returns:
            The clipped position.
        """
        clipped = np.asarray(point, dtype=float)[:2].copy()
        np.clip(clipped, 0.0, self.world_size, out=clipped)
        return clipped

    def is_open(self, key: WallKey, coordinate: float) -> bool:
        """Return whether a wall is open at one coordinate.

        Args:
            key: The wall key.
            coordinate: The position along the wall's free axis.

        How:
            Looks up the wall's door interval. A missing wall entry means no
            door, so the wall is solid.

        Returns:
            ``True`` when a door covers ``coordinate``.
        """
        interval = self.doors.get(key)
        if interval is None:
            return False
        low, high = interval
        return low <= coordinate <= high

    def _crossing_is_open(self, axis: str, line_index: int, coordinate: float) -> bool:
        """Return whether the wall at one grid line is open at a crossing.

        Args:
            axis: ``"v"`` for a vertical wall, ``"h"`` for a horizontal wall.
            line_index: Index of the grid line, in ``[1, rooms_per_side - 1]``.
            coordinate: Position along the wall's free axis.

        How:
            Converts the grid line and crossing coordinate into a wall key. For
            a vertical wall the room row comes from the coordinate. For a
            horizontal wall the room column comes from the coordinate.

        Returns:
            ``True`` when the crossed wall has a door at the crossing point.
        """
        room = _room_index(coordinate, self.room_size, self.rooms_per_side)
        if axis == _VERTICAL:
            key: WallKey = (_VERTICAL, line_index - 1, room)
        else:
            key = (_HORIZONTAL, room, line_index - 1)
        return self.is_open(key, coordinate)

    def segment_is_blocked(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Return whether a straight move crosses a solid wall.

        Args:
            start: Start position ``(x, y)``.
            end: End position ``(x, y)``.

        How:
            For each internal grid line, checks whether the segment crosses it.
            A crossing is strict, so a move along a wall is not blocked by that
            wall. At each crossing it tests the wall key for a door. One solid
            crossing blocks the move.

        Returns:
            ``True`` when the move hits a wall without a door.
        """
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        for line_index in range(1, self.rooms_per_side):
            line = line_index * self.room_size
            if (x0 - line) * (x1 - line) < 0:
                y_at = y0 + (y1 - y0) * (line - x0) / (x1 - x0)
                if not self._crossing_is_open(_VERTICAL, line_index, y_at):
                    return True
            if (y0 - line) * (y1 - line) < 0:
                x_at = x0 + (x1 - x0) * (line - y0) / (y1 - y0)
                if not self._crossing_is_open(_HORIZONTAL, line_index, x_at):
                    return True
        return False


def layout_to_dict(layout: RoomsLayout) -> dict:
    """Convert a layout to a JSON-serialisable mapping.

    Args:
        layout: The layout to convert.

    How:
        Flattens each door entry to ``[axis, col, row, low, high]``, because JSON
        object keys cannot be tuples, and sorts them so the output is stable.

    Returns:
        A mapping with the room size, the grid size and the door list.
    """
    doors = [[*key, low, high] for key, (low, high) in sorted(layout.doors.items())]
    return {"room_size": layout.room_size, "rooms_per_side": layout.rooms_per_side, "doors": doors}


def layout_from_dict(payload: dict) -> RoomsLayout:
    """Rebuild a layout from :func:`layout_to_dict`'s output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Restores each door's tuple key and interval.

    Returns:
        The layout.
    """
    doors = {(axis, col, row): (low, high) for axis, col, row, low, high in payload["doors"]}
    return RoomsLayout(doors, payload["room_size"], payload["rooms_per_side"])

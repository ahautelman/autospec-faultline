"""Wall, door, and region model for the 9-rooms environment.

The world is a 3x3 grid of square rooms. Walls sit on the grid lines. A wall is
solid unless a door opening covers the point where a path crosses it. Doors are
open intervals along a wall's free coordinate.

Wall keys:

- ``("v", col, row)``: the vertical wall at ``x = (col + 1) * ROOM_SIZE``,
  between room ``(col, row)`` and room ``(col + 1, row)``.
- ``("h", col, row)``: the horizontal wall at ``y = (row + 1) * ROOM_SIZE``,
  between room ``(col, row)`` and room ``(col, row + 1)``.

The default layout opens one corridor from START to GOAL. It passes through MID
and the two side rooms along the way. This gives a single intended route until a
scenario adds or removes doors.
"""

from __future__ import annotations

import numpy as np

from faultline.spectrl.predicates import Box

ROOMS_PER_SIDE = 3
ROOM_SIZE = 8.0
WORLD_SIZE = ROOMS_PER_SIDE * ROOM_SIZE
DOOR_WIDTH = 2.0

# Each door is centred in the wall it cuts, so its offset from the wall's start
# is half a room. The width then spans DOOR_WIDTH / 2 either side of the centre.
_DOOR_CENTRE = ROOM_SIZE / 2.0

WallKey = tuple[str, int, int]
DoorInterval = tuple[float, float]

_VERTICAL = "v"
_HORIZONTAL = "h"


def default_doors() -> dict[WallKey, DoorInterval]:
    """Return the door set of the default 9-rooms corridor.

    How:
        Opens four walls that chain START ``(0, 0)`` to MID ``(1, 1)`` to GOAL
        ``(2, 2)``, passing through rooms ``(1, 0)`` and ``(2, 1)``. Each door
        is centred in its wall.

    Returns:
        A mapping from wall key to the open interval on that wall.
    """
    half = DOOR_WIDTH / 2.0
    return {
        ("v", 0, 0): (_DOOR_CENTRE - half, _DOOR_CENTRE + half),
        ("h", 1, 0): (ROOM_SIZE + _DOOR_CENTRE - half, ROOM_SIZE + _DOOR_CENTRE + half),
        ("v", 1, 1): (ROOM_SIZE + _DOOR_CENTRE - half, ROOM_SIZE + _DOOR_CENTRE + half),
        ("h", 2, 1): (
            2.0 * ROOM_SIZE + _DOOR_CENTRE - half,
            2.0 * ROOM_SIZE + _DOOR_CENTRE + half,
        ),
    }


def default_regions() -> dict[str, Box]:
    """Return the named checkpoint regions of the 9-rooms grid.

    How:
        Places START in the bottom-left room, MID in the centre room, and GOAL
        in the top-right room. Each region is a 6x6 box inset by one unit from
        its room walls.

    Returns:
        A mapping from region name to its box predicate.
    """
    return {
        "start": Box((1.0, 1.0), (7.0, 7.0)),
        "mid": Box((9.0, 9.0), (15.0, 15.0)),
        "goal": Box((17.0, 17.0), (23.0, 23.0)),
    }


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
        doors: Wall key to open interval mapping. Defaults to the 9-rooms
            corridor from :func:`default_doors`.
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
        room_size: float = ROOM_SIZE,
        rooms_per_side: int = ROOMS_PER_SIDE,
    ) -> None:
        self.room_size = float(room_size)
        self.rooms_per_side = int(rooms_per_side)
        self.world_size = self.room_size * self.rooms_per_side
        self.doors = dict(default_doors()) if doors is None else dict(doors)

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

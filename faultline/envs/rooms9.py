"""The 9-rooms continuous environment from paper section 8.

This module keeps the 9-rooms constants, the five hand-built door maps from the
paper's validation scenarios, and :class:`Rooms9`, a thin subclass of the
generic :class:`~faultline.envs.rooms.env.RoomsEnv` that supplies the 9-rooms
defaults. The generic model lives in :mod:`faultline.envs.rooms`.
"""

from __future__ import annotations

from faultline.envs.rooms.env import (
    ACTION_DIM,
    DEFAULT_MAX_STEPS,
    MAX_HEADING,
    MAX_SPEED,
    MIN_SPEED,
    STATE_DIM,
    RoomsEnv,
)
from faultline.envs.rooms.layout import (
    DoorInterval,
    RoomsLayout,
    WallKey,
    open_doors_except_sealed_rooms,
)
from faultline.spectrl.predicates import Box

ROOMS_PER_SIDE = 3
ROOM_SIZE = 8.0
WORLD_SIZE = ROOMS_PER_SIDE * ROOM_SIZE
DOOR_WIDTH = 2.0

# Each door is centred in the wall it cuts, so its offset from the wall's start
# is half a room. The width then spans DOOR_WIDTH / 2 either side of the centre.
_DOOR_CENTRE = ROOM_SIZE / 2.0


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


def long_corridor_doors() -> dict[WallKey, DoorInterval]:
    """Return the door set of the paper's winding 9-rooms corridor.

    How:
        Opens the six internal walls the paper's long-corridor figure leaves
        unblocked. The route from START in room ``(0, 0)`` climbs the left column
        to room ``(0, 2)``, crosses into room ``(1, 2)``, and drops south into GOAL
        in the centre room ``(1, 1)``. The far-right column is walled off from the
        centre except at the bottom row, so that winding path is the single
        intended route. Every opening spans its whole room edge, so the joined
        rooms merge into one area rather than a centred gap.

    Returns:
        A mapping from wall key to the full-edge open interval.
    """
    size = ROOM_SIZE
    return {
        ("h", 0, 0): (0.0 * size, 1.0 * size),  # (0, 0) - (0, 1)
        ("h", 0, 1): (0.0 * size, 1.0 * size),  # (0, 1) - (0, 2)
        ("h", 1, 0): (1.0 * size, 2.0 * size),  # (1, 0) - (1, 1)
        ("h", 1, 1): (1.0 * size, 2.0 * size),  # (1, 1) - (1, 2)
        ("v", 0, 2): (2.0 * size, 3.0 * size),  # (0, 2) - (1, 2)
        ("v", 1, 0): (0.0 * size, 1.0 * size),  # (1, 0) - (2, 0)
    }


def long_mixed_star_doors() -> dict[WallKey, DoorInterval]:
    """Return the door set of the paper's long mixed star room.

    How:
        Opens the six internal walls the paper's long mixed star figure leaves
        unblocked. The free area is a long zig-zag from START in room ``(0, 0)``:
        it climbs into ``(0, 1)``, steps east into ``(1, 1)``, east again into
        ``(2, 1)``, and finally climbs into GOAL in ``(2, 2)``. The bottom row
        ``(0, 0) - (1, 0) - (2, 0)`` is a second, dead-end arm, and the two top-row
        rooms ``(0, 2)`` and ``(1, 2)`` stay walled off. Every opening spans its
        whole room edge, so the joined rooms merge into one area rather than a
        centred gap.

    Returns:
        A mapping from wall key to the full-edge open interval.
    """
    size = ROOM_SIZE
    return {
        ("v", 0, 0): (0.0 * size, 1.0 * size),  # (0, 0) - (1, 0)
        ("v", 1, 0): (0.0 * size, 1.0 * size),  # (1, 0) - (2, 0)
        ("h", 0, 0): (0.0 * size, 1.0 * size),  # (0, 0) - (0, 1)
        ("v", 0, 1): (1.0 * size, 2.0 * size),  # (0, 1) - (1, 1)
        ("v", 1, 1): (1.0 * size, 2.0 * size),  # (1, 1) - (2, 1)
        ("h", 2, 1): (2.0 * size, 3.0 * size),  # (2, 1) - (2, 2)
    }


def blocked_two_path_doors() -> dict[WallKey, DoorInterval]:
    """Return the door set of the paper's blocked two-path room.

    How:
        Reproduces the paper's two-path figure. START sits in room ``(0, 0)``, the
        left column climbs into MID1, the merged pair ``(0, 2) + (1, 2)``; the
        centre column drops from MID1 through room ``(1, 1)`` into MID2, the merged
        pair ``(1, 0) + (2, 0)``; and the right column climbs from MID2 into GOAL in
        room ``(2, 2)``. Two of the four interior vertical walls -- START to the
        bottom-middle room and MID1 to GOAL -- stay solid, so the direct edge that
        would shorten each nominal route is blocked and only the winding path
        realises both. Every wall the winding route crosses is fully open, so each
        adjacent pair of rooms it joins merges into one area.

    Returns:
        A mapping from wall key to the open interval on that wall.
    """
    size = ROOM_SIZE
    return {
        # Full-edge merges: MID1 spans rooms (0, 2) and (1, 2); MID2 spans rooms
        # (1, 0) and (2, 0).
        ("v", 0, 2): (2.0 * size, 3.0 * size),  # (0, 2) - (1, 2)
        ("v", 1, 0): (0.0 * size, 1.0 * size),  # (1, 0) - (2, 0)
        # The centre column is open from MID1 down through (1, 1) into MID2.
        ("h", 1, 1): (1.0 * size, 2.0 * size),  # (1, 1) - (1, 2)
        ("h", 1, 0): (1.0 * size, 2.0 * size),  # (1, 0) - (1, 1)
        # The left and right columns are fully open, so each column's stacked
        # rooms merge the same way the MID1 and MID2 pairs do.
        ("h", 0, 0): (0.0 * size, 1.0 * size),  # (0, 0) - (0, 1)
        ("h", 0, 1): (0.0 * size, 1.0 * size),  # (0, 1) - (0, 2)
        ("h", 2, 0): (2.0 * size, 3.0 * size),  # (2, 0) - (2, 1)
        ("h", 2, 1): (2.0 * size, 3.0 * size),  # (2, 1) - (2, 2)
    }


class Rooms9(RoomsEnv):
    """A 3x3 room grid with point-mass motion and reach-avoid subgoals.

    Args:
        layout: Wall and door model. Defaults to the single-corridor 9-rooms
            layout.
        max_steps: Episode step limit, after which the episode truncates.
        unsafe_region: Optional trap region. Entering it is an absorbing unsafe
            outcome: the episode ends as a violation, so the agent can never
            leave. Defaults to no trap.

    How:
        Supplies the 9-rooms default layout to :class:`RoomsEnv`; all dynamics
        are inherited unchanged.
    """

    def __init__(
        self,
        layout: RoomsLayout | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        unsafe_region=None,
    ) -> None:
        if layout is None:
            layout = RoomsLayout(default_doors(), ROOM_SIZE, ROOMS_PER_SIDE)
        super().__init__(layout=layout, max_steps=max_steps, unsafe_region=unsafe_region)


__all__ = [
    "ACTION_DIM",
    "DEFAULT_MAX_STEPS",
    "DOOR_WIDTH",
    "MAX_HEADING",
    "MAX_SPEED",
    "MIN_SPEED",
    "ROOMS_PER_SIDE",
    "ROOM_SIZE",
    "STATE_DIM",
    "WORLD_SIZE",
    "Rooms9",
    "blocked_two_path_doors",
    "default_doors",
    "default_regions",
    "long_corridor_doors",
    "long_mixed_star_doors",
    "open_doors_except_sealed_rooms",
]

"""Episode step limit derived from the specification's room paths.

A fixed 200-step horizon on a winding 100-rooms edge could make the horizon, not
the specification, the bottleneck, so the n-rooms worlds derive their limit from
the room graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil

from faultline.envs.rooms.grid import RoomGrid, RoomKey

# The repository's point-mass speed and the minimum horizon the 9-rooms
# experiments use. The n-rooms generator keeps the same speed.
DEFAULT_SPEED = 2.0
MIN_MAX_STEPS = 200
DEFAULT_SLACK = 4.0


def default_max_steps(
    grid: RoomGrid,
    edge_rooms: Iterable[tuple[Iterable[RoomKey], Iterable[RoomKey]]],
    *,
    speed: float = DEFAULT_SPEED,
    slack: float = DEFAULT_SLACK,
    floor: int = MIN_MAX_STEPS,
) -> int:
    """Derive an episode step limit from the specification's room paths.

    Args:
        grid: The room graph.
        edge_rooms: One ``(source_rooms, target_rooms)`` pair per spec edge.
        speed: Agent speed in world units per step.
        slack: Multiplicative safety margin over the longest pure travel time.
        floor: Smallest returned horizon.

    How:
        Finds the shortest room path for every edge and keeps the longest one.
        That length, in door crossings, is multiplied by ``room_size / speed``
        and ``slack`` and rounded up. A fixed 200-step horizon on a winding
        100-rooms edge could otherwise make the horizon, not the specification,
        the bottleneck.

        Edges with no room path are skipped rather than rejected: the verifier
        allows a dead branch, so a solvable world may still contain one. If no
        edge is reachable at all the world is unsolvable and the floor is
        returned, which keeps the generator's rejection path cheap.

    Returns:
        The horizon in steps, never below ``floor``.

    Raises:
        ValueError: If ``speed`` is not positive.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    longest = 0
    for source_rooms, target_rooms in edge_rooms:
        path = grid.shortest_path(source_rooms, target_rooms)
        if path is not None:
            longest = max(longest, path.length)
    travel = longest * grid.room_size / speed * slack
    return max(int(floor), int(ceil(travel)))


__all__ = ["DEFAULT_SLACK", "DEFAULT_SPEED", "MIN_MAX_STEPS", "default_max_steps"]

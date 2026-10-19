"""Back-compat shim for the old 9-rooms layout module.

The generic model lives in :mod:`faultline.envs.rooms.layout`; the 9-rooms
constants and hand-built door maps live in :mod:`faultline.envs.rooms9`. This
module re-exports both so existing imports keep working unchanged.
"""

from faultline.envs.rooms.layout import (
    DoorInterval,
    RoomsLayout,
    WallKey,
    layout_from_dict,
    layout_to_dict,
    open_doors_except_sealed_rooms,
)
from faultline.envs.rooms9 import (
    DOOR_WIDTH,
    ROOMS_PER_SIDE,
    ROOM_SIZE,
    WORLD_SIZE,
    blocked_two_path_doors,
    default_doors,
    default_regions,
    long_corridor_doors,
    long_mixed_star_doors,
)

__all__ = [
    "DOOR_WIDTH",
    "DoorInterval",
    "ROOMS_PER_SIDE",
    "ROOM_SIZE",
    "RoomsLayout",
    "WORLD_SIZE",
    "WallKey",
    "blocked_two_path_doors",
    "default_doors",
    "default_regions",
    "layout_from_dict",
    "layout_to_dict",
    "long_corridor_doors",
    "long_mixed_star_doors",
    "open_doors_except_sealed_rooms",
]

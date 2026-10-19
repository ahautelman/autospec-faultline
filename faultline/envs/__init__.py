"""Room-grid environments: wall and door model, and gymnasium environments.

Re-exports the public surface so callers can import from ``faultline.envs``
directly. The 9-rooms scenario remains available as :class:`Rooms9`; the
size-agnostic class is :class:`RoomsEnv`.
"""

from faultline.envs.layout import (
    RoomsLayout,
    blocked_two_path_doors,
    default_doors,
    default_regions,
    layout_from_dict,
    layout_to_dict,
    long_corridor_doors,
    long_mixed_star_doors,
    open_doors_except_sealed_rooms,
)
from faultline.envs.rooms.env import RoomsEnv
from faultline.envs.rooms9 import Rooms9

__all__ = [
    "Rooms9",
    "RoomsEnv",
    "RoomsLayout",
    "blocked_two_path_doors",
    "default_doors",
    "default_regions",
    "layout_from_dict",
    "layout_to_dict",
    "long_corridor_doors",
    "long_mixed_star_doors",
    "open_doors_except_sealed_rooms",
]

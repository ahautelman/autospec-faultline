"""The 9-rooms environment: wall and door model, and the gymnasium environment.

Re-exports the public surface so callers can import from ``faultline.envs``
directly.
"""

from faultline.envs.layout import (
    RoomsLayout,
    blocked_two_path_doors,
    default_doors,
    default_regions,
    long_corridor_doors,
    long_mixed_star_doors,
    open_doors_except_sealed_rooms,
)
from faultline.envs.rooms9 import Rooms9

__all__ = [
    "Rooms9",
    "RoomsLayout",
    "blocked_two_path_doors",
    "default_doors",
    "default_regions",
    "long_corridor_doors",
    "long_mixed_star_doors",
    "open_doors_except_sealed_rooms",
]

"""Generic n-rooms model: layout, room-level analysis, world, and environment.

The 9-rooms constants and hand-built door sets stay in
:mod:`faultline.envs.rooms9`; this package is size-agnostic.
"""

from faultline.envs.rooms.env import RoomsEnv
from faultline.envs.rooms.generator import (
    DEFAULT_DOOR_PROBABILITY,
    DEFAULT_MAX_ATTEMPTS,
    LayoutGenerationError,
    corner_blocks,
    generate_world,
)
from faultline.envs.rooms.grid import RoomGrid, RoomPath
from faultline.envs.rooms.horizon import default_max_steps
from faultline.envs.rooms.layout import (
    DoorInterval,
    RoomsLayout,
    WallKey,
    layout_from_dict,
    layout_to_dict,
    open_doors_except_sealed_rooms,
)
from faultline.envs.rooms.regions import RegionBlock, valid_blocks
from faultline.envs.rooms.rejection import RejectionStats, measure_rejection
from faultline.envs.rooms.stochastic import (
    DIRL_NOT_IMPLEMENTED_MESSAGE,
    DiRLVerifier,
    StochasticReport,
    StochasticVerifier,
)
from faultline.envs.rooms.verify import VerificationReport, verify_heuristic
from faultline.envs.rooms.world import (
    PAPER_REGION_ORDER,
    PAPER_TOPOLOGY,
    RoomsWorld,
    SpecTopology,
    paper_world_name,
    world_from_dict,
    world_to_dict,
)

__all__ = [
    "DEFAULT_DOOR_PROBABILITY",
    "DEFAULT_MAX_ATTEMPTS",
    "DIRL_NOT_IMPLEMENTED_MESSAGE",
    "PAPER_REGION_ORDER",
    "PAPER_TOPOLOGY",
    "DiRLVerifier",
    "DoorInterval",
    "LayoutGenerationError",
    "RejectionStats",
    "RegionBlock",
    "RoomGrid",
    "RoomPath",
    "RoomsEnv",
    "RoomsLayout",
    "RoomsWorld",
    "SpecTopology",
    "StochasticReport",
    "StochasticVerifier",
    "VerificationReport",
    "WallKey",
    "corner_blocks",
    "default_max_steps",
    "generate_world",
    "layout_from_dict",
    "layout_to_dict",
    "measure_rejection",
    "open_doors_except_sealed_rooms",
    "paper_world_name",
    "valid_blocks",
    "verify_heuristic",
    "world_from_dict",
    "world_to_dict",
]

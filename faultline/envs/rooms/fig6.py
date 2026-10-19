"""The transcribed Fig. 6a instance of the paper's procedural 100-rooms world."""

from __future__ import annotations

from faultline.envs.rooms.layout import (
    DEFAULT_ROOM_SIZE,
    RoomsLayout,
    WallKey,
    full_edge_interval,
    internal_wall_keys,
)
from faultline.envs.rooms.regions import RegionBlock
from faultline.envs.rooms.world import PAPER_TOPOLOGY, RoomsWorld


# Transcribed from the paper's single published procedural instance. Unlike
# Fig. 3, the door map here is *not* designed: every internal edge that the
# figure does not draw as a red wall is an open door. The wall list below is
# the record of a one-off pixel detection on a 400-dpi render; the detection
# code is deliberately not shipped.
#
# The figure departs from the generator text in two ways, both recorded in the
# world metadata:
# - the Goal block is on the left edge, not in a corner;
# - door density is 123/180 (~68%), not the stated P(door) = 0.5.
#
# Wall keys follow RoomsLayout: ("v", col, row) separates (col, row) from
# (col + 1, row); ("h", col, row) separates (col, row) from (col, row + 1).

FIG6_REGIONS: dict[str, RegionBlock] = {
    "Start": RegionBlock(8, 0),
    "M1": RegionBlock(5, 2),
    "M2": RegionBlock(4, 0),
    "M3": RegionBlock(8, 4),
    "M4": RegionBlock(2, 3),
    "M5": RegionBlock(3, 8),
    "Goal": RegionBlock(0, 6),
}

FIG6_WALLS: tuple[WallKey, ...] = (
    # Vertical walls, 27.
    ("v", 0, 5),
    ("v", 0, 7),
    ("v", 0, 8),
    ("v", 0, 9),
    ("v", 1, 0),
    ("v", 1, 3),
    ("v", 1, 7),
    ("v", 1, 8),
    ("v", 2, 2),
    ("v", 2, 4),
    ("v", 2, 7),
    ("v", 3, 0),
    ("v", 3, 1),
    ("v", 3, 8),
    ("v", 3, 9),
    ("v", 4, 0),
    ("v", 4, 1),
    ("v", 4, 4),
    ("v", 4, 8),
    ("v", 5, 4),
    ("v", 5, 5),
    ("v", 5, 8),
    # One ambiguous edge: a clear vertical stroke at weak contrast, kept as a
    # wall by the transcription.
    ("v", 6, 3),
    ("v", 6, 7),
    ("v", 7, 5),
    ("v", 7, 6),
    ("v", 7, 9),
    # Horizontal walls, 30.
    ("h", 0, 1),
    ("h", 0, 3),
    ("h", 0, 6),
    ("h", 0, 7),
    ("h", 1, 3),
    ("h", 1, 4),
    ("h", 1, 5),
    ("h", 2, 0),
    ("h", 2, 4),
    ("h", 2, 8),
    ("h", 3, 8),
    ("h", 4, 0),
    ("h", 4, 2),
    ("h", 4, 3),
    ("h", 4, 4),
    ("h", 4, 5),
    ("h", 4, 7),
    ("h", 4, 8),
    ("h", 5, 1),
    ("h", 5, 2),
    ("h", 6, 4),
    ("h", 6, 6),
    ("h", 7, 3),
    ("h", 7, 5),
    ("h", 7, 7),
    ("h", 8, 5),
    ("h", 8, 7),
    ("h", 9, 1),
    ("h", 9, 5),
    ("h", 9, 8),
)


FIG6_DOORS: tuple[WallKey, ...] = tuple(
    key for key in internal_wall_keys(10) if key not in frozenset(FIG6_WALLS)
)


def fig6_world(room_size: float = DEFAULT_ROOM_SIZE) -> RoomsWorld:
    """Build the transcribed Fig. 6a instance.

    Args:
        room_size: Width and height of one room.

    How:
        Opens every internal edge that is not in :data:`FIG6_WALLS` across its
        whole room edge. The region blocks and topology are the paper's. The
        metadata records the source, the one ambiguous wall, and the two ways
        the instance departs from the generator text.

    Returns:
        The Fig. 6 world.
    """
    doors = {key: full_edge_interval(key, room_size) for key in FIG6_DOORS}
    return RoomsWorld(
        name="fig6",
        layout=RoomsLayout(doors, room_size, 10),
        regions=FIG6_REGIONS,
        topology=PAPER_TOPOLOGY,
        meta={
            "source": "paper Fig. 6a",
            "transcription": "pixel-detected from the PDF, one ambiguous edge v(6,3)",
            "departures": [
                "Goal block (0, 6) is on the left edge, not in a corner",
                "door density is 123/180 (~68%), not the stated P(door)=0.5",
            ],
        },
    )

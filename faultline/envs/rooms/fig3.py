"""The hand-designed Fig. 3 replica of the paper's fixed 100-rooms world.

The paper gives the seven regions but not the door map, so the map here is
designed and documented as such.
"""

from __future__ import annotations

from faultline.envs.rooms.layout import (
    DEFAULT_ROOM_SIZE,
    RoomsLayout,
    WallKey,
    full_edge_interval,
)
from faultline.envs.rooms.regions import RegionBlock
from faultline.envs.rooms.world import PAPER_TOPOLOGY, RoomsWorld


# The paper gives the regions but not the door map. The map below is designed,
# not transcribed, and is honest about that: the paper's ≈0% -> ≈60% numbers are
# an order-of-magnitude reference, not something this door map claims to match.
#
# Design table (what each invented door group is for):
#
# | DAG edge       | Design | Expected operator |
# | -------------- | ------ | ----------------- |
# | Start -> M1    | open side corridor from Start into M1; room (4, 1) of M1 is sealed | seq (M1 is 3/4 reachable) |
# | Start -> M2    | open corridor up the left side | none (live) |
# | M1 -> M3       | no direct doors; only a long detour through Start and M2; M3 is half sealed | seq |
# | M2 -> M3       | open corridor into M3's top pair (row 4); M3's bottom pair is sealed | seq (M3 is 1/2 reachable) |
# | M3 -> M4       | single bridge from M3's top pair into M4 | none (see below) |
# | M3 -> M5       | M5 is walled off entirely (dead branch, no samples to refine from) | none |
# | M4 -> Goal     | open corridor from M4 into Goal | none (live) |
# | M5 -> Goal     | unreachable because M5 itself is sealed | none |
#
# Region-level symptoms:
# - M1 is 3/4 reachable because room (4, 1) is sealed inside the block, so the
#   target of Start -> M1 is partly unreachable (ReachRefine).
# - M3 is split into a bottom and a top pair; only the top pair (row 4)
#   connects to M2 and M4, and the bottom pair is sealed. Half of the targets
#   sampled in M3 are unreachable, so ReachRefine on the edges into M3 is the
#   expected repair. It also removes the sealed half from the source of
#   M3 -> M4, which is why PastRefine is not expected there: a tabular smoke run
#   restores M3 -> M4 from 0.3 to 1.0 through the SeqRefine on M2 -> M3 alone.
# - A dead branch (M5) leaves OrRefine nothing to do: M5's only parent is M3, so
#   there is no sibling parent to reroute through, and zero successes means no
#   operator has a witness.
# - The M2 route is the best route; the M1 route exists but is much longer.
#
# The expectations are design intent, derived from the layout diagnostics and
# checked against them in the tests. They are reported next to the committed
# operators and never scored against.

FIG3_REGIONS: dict[str, RegionBlock] = {
    "Start": RegionBlock(0, 0),
    "M1": RegionBlock(3, 0),
    "M2": RegionBlock(0, 3),
    "M3": RegionBlock(3, 3),
    "M4": RegionBlock(6, 4),
    "M5": RegionBlock(3, 6),
    "Goal": RegionBlock(6, 7),
}

FIG3_EXPECTED_OPERATORS: dict[str, list[str]] = {
    "Start->M1": ["seq"],
    "M1->M3": ["seq"],
    "M2->M3": ["seq"],
}

FIG3_DOORS: tuple[WallKey, ...] = (
    # Start block: all four internal walls open.
    ("v", 0, 0),
    ("v", 0, 1),
    ("h", 0, 0),
    ("h", 1, 0),
    # Start -> M1 side corridor.
    ("v", 1, 1),
    ("v", 2, 1),
    # M1 internal: (4, 1) stays sealed by omitting its top and left walls.
    ("h", 3, 0),
    ("v", 3, 0),
    # Start -> M2 corridor up the left side.
    ("h", 0, 1),
    ("h", 0, 2),
    # M2 block: all internal walls open.
    ("v", 0, 3),
    ("v", 0, 4),
    ("h", 0, 3),
    ("h", 1, 3),
    # M2 -> M3 corridor into the top pair.
    ("v", 1, 4),
    ("v", 2, 4),
    # M3's top pair and its sealed bottom pair, with the horizontal divide closed.
    ("v", 3, 4),
    ("v", 3, 3),
    # M3 top pair -> M4 bridge.
    ("v", 4, 4),
    ("v", 5, 4),
    # M4 block: all internal walls open.
    ("v", 6, 4),
    ("v", 6, 5),
    ("h", 6, 4),
    ("h", 7, 4),
    # M4 -> Goal corridor.
    ("h", 6, 5),
    ("h", 6, 6),
    # Goal block: all internal walls open.
    ("v", 6, 7),
    ("v", 6, 8),
    ("h", 6, 7),
    ("h", 7, 7),
)


def fig3_world(room_size: float = DEFAULT_ROOM_SIZE) -> RoomsWorld:
    """Build the hand-designed Fig. 3 replica.

    Args:
        room_size: Width and height of one room.

    How:
        Uses the paper's region table for the seven blocks and the designed door
        map above. Every listed wall opens its whole room edge. The world is
        frozen to ``fig3.json`` and loaded through the registry at run time.

    Returns:
        The Fig. 3 world.
    """
    doors = {key: full_edge_interval(key, room_size) for key in FIG3_DOORS}
    return RoomsWorld(
        name="fig3",
        layout=RoomsLayout(doors, room_size, 10),
        regions=FIG3_REGIONS,
        topology=PAPER_TOPOLOGY,
        meta={
            "design": "hand-designed Fig. 3 door map; paper gives no doors",
            # Per-edge expectations from the design table above; edges with no
            # expected operator are left out (a missing key is "no claim").
            "expected_operators": FIG3_EXPECTED_OPERATORS,
        },
    )

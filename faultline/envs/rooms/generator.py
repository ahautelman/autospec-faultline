"""Procedural n-rooms world generator from paper A.3.3-A.3.4.

The paper samples:

- Start and Goal in opposite corners (eq. 2),
- each remaining 2x2 region uniformly among the valid non-overlapping blocks
  (eq. 3),
- every orthogonally adjacent room pair as a door with probability 0.5,
  independently (eq. 4),

and keeps only layouts with a solvable structure. This module reproduces that
and adds the two choices recorded in the plan: the four ordered corner
orientations are sampled uniformly (the paper writes two, but Fig. 6a shows a
swapped orientation), and a failed verification redraws the placement and the
doors together.

Determinism is the hard requirement: one ``numpy.random.Generator(seed)`` is
consumed in a fixed order, so a seed fully determines the returned world. The
number of attempts is stored in ``meta``; :mod:`faultline.envs.rooms.rejection`
aggregates it into a rejection rate.

Hazards are plumbed through :class:`RoomsWorld` but never generated here: the
paper's n-rooms specifications have no avoid region, so the default is empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from faultline.envs.rooms.layout import (
    DEFAULT_ROOM_SIZE,
    DoorInterval,
    RoomsLayout,
    WallKey,
    full_edge_interval,
    internal_wall_keys,
)
from faultline.envs.rooms.regions import BLOCK_SIDE, RegionBlock, valid_blocks
from faultline.envs.rooms.stochastic import StochasticVerifier
from faultline.envs.rooms.verify import verify_heuristic
from faultline.envs.rooms.world import (
    PAPER_REGION_ORDER,
    PAPER_TOPOLOGY,
    RoomsWorld,
    paper_world_name,
)

DEFAULT_DOOR_PROBABILITY = 0.5
DEFAULT_MAX_ATTEMPTS = 1000

# The four ordered Start/Goal corner pairs in a fixed sampling order. Labels
# use the figure's top/bottom orientation, so Fig. 6a's Start bottom-right and
# Goal top-left is ``BR-TL``.
_CORNER_ORIENTATIONS: tuple[tuple[str, str, str], ...] = (
    ("TL-BR", "TL", "BR"),
    ("BR-TL", "BR", "TL"),
    ("TR-BL", "TR", "BL"),
    ("BL-TR", "BL", "TR"),
)


class LayoutGenerationError(RuntimeError):
    """Raised when no verified layout is found within ``max_attempts``."""


class _PlacementError(Exception):
    """Internal signal that an attempt could not place every region."""


@dataclass(frozen=True)
class _DrawSpec:
    """The fixed parameters of one generation run, shared by every attempt."""

    seed: int
    rooms_per_side: int
    room_size: float
    region_names: tuple[str, ...]
    door_probability: float


def corner_blocks(rooms_per_side: int) -> dict[str, RegionBlock]:
    """Return the four corner 2x2 blocks.

    Args:
        rooms_per_side: Number of rooms along each axis; must be at least 2.

    How:
        ``TL`` and ``TR`` are the top corners, ``BL`` and ``BR`` the bottom ones,
        all inset by one room so the 2x2 block fits.

    Returns:
        A mapping from corner label to its block.

    Raises:
        ValueError: If the grid is smaller than 2 rooms per side.
    """
    if rooms_per_side < BLOCK_SIDE:
        raise ValueError(f"A 2x2 corner needs at least 2 rooms per side, got {rooms_per_side}")
    far = rooms_per_side - BLOCK_SIDE
    return {
        "TL": RegionBlock(0, far),
        "TR": RegionBlock(far, far),
        "BL": RegionBlock(0, 0),
        "BR": RegionBlock(far, 0),
    }


def _validate_regions(regions: Sequence[str]) -> tuple[str, ...]:
    """Validate the region-name order the generator needs.

    Args:
        regions: Region names in the order the caller wants them placed.

    How:
        Requires ``Start`` first and ``Goal`` last, both present exactly once,
        and no duplicates. The intermediate names are used in order, so the
        distribution depends on the order the caller passes.

    Returns:
        The names as a tuple.

    Raises:
        ValueError: If the order is malformed.
    """
    names = tuple(regions)
    if not names or names[0] != "Start" or names[-1] != "Goal":
        raise ValueError("regions must start with 'Start' and end with 'Goal'")
    if len(set(names)) != len(names):
        raise ValueError("regions must be unique")
    return names


def _sample_placement(
    rng: np.random.Generator,
    rooms_per_side: int,
    region_names: Sequence[str],
) -> tuple[str, dict[str, RegionBlock]]:
    """Sample the corner orientation and every intermediate block.

    Args:
        rng: The generator's single random source.
        rooms_per_side: Number of rooms along each axis.
        region_names: Region names in placement order, ``Start`` first and
            ``Goal`` last.

    How:
        Draws one of the four ordered corner pairs, then places the intermediate
        regions in the given order. Each intermediate is drawn uniformly from
        the valid blocks that overlap nothing placed so far. Touching blocks are
        allowed. A draw from an empty candidate set fails the attempt.

    Returns:
        The orientation label and the region blocks.

    Raises:
        _PlacementError: If some intermediate region has no free block.
    """
    corners = corner_blocks(rooms_per_side)
    orientation = _CORNER_ORIENTATIONS[int(rng.integers(len(_CORNER_ORIENTATIONS)))]
    label, start_corner, goal_corner = orientation
    placed: dict[str, RegionBlock] = {
        "Start": corners[start_corner],
        "Goal": corners[goal_corner],
    }
    blocks = valid_blocks(rooms_per_side)
    for name in region_names[1:-1]:
        candidates = [
            block
            for block in blocks
            if all(not block.overlaps(other) for other in placed.values())
        ]
        if not candidates:
            raise _PlacementError(f"No free block for region {name!r}")
        placed[name] = candidates[int(rng.integers(len(candidates)))]
    return label, placed


def _sample_doors(
    rng: np.random.Generator,
    rooms_per_side: int,
    room_size: float,
    door_probability: float,
) -> dict[WallKey, DoorInterval]:
    """Sample each internal wall independently as a full-edge door or solid.

    Args:
        rng: The generator's single random source.
        rooms_per_side: Number of rooms along each axis.
        room_size: Width and height of one room.
        door_probability: Independent probability that a wall opens.

    How:
        Visits the walls in :func:`internal_wall_keys` order. Each wall consumes
        exactly one draw, so the sequence is fixed and a seed is reproducible. An
        open wall covers its whole room edge.

    Returns:
        The full-edge door map.
    """
    return {
        key: full_edge_interval(key, room_size)
        for key in internal_wall_keys(rooms_per_side)
        if rng.random() < door_probability
    }


def _sample_world(rng: np.random.Generator, spec: _DrawSpec) -> tuple[RoomsWorld, str]:
    """Draw one unverified world and return it with its orientation label.

    Args:
        rng: The generator's single random source.
        spec: The fixed parameters of the run.

    How:
        Samples the placement and then the doors from the same source, in that
        order, so a seed fixes the world.

    Returns:
        The unverified world and its corner-orientation label.
    """
    label, regions = _sample_placement(rng, spec.rooms_per_side, spec.region_names)
    doors = _sample_doors(rng, spec.rooms_per_side, spec.room_size, spec.door_probability)
    world = RoomsWorld(
        name=paper_world_name(spec.seed),
        layout=RoomsLayout(doors, spec.room_size, spec.rooms_per_side),
        regions=regions,
        topology=PAPER_TOPOLOGY,
    )
    return world, label


def _attempt(
    rng: np.random.Generator,
    spec: _DrawSpec,
    attempt: int,
    stochastic: StochasticVerifier | None,
) -> tuple[RoomsWorld | None, dict]:
    """Draw and check one candidate world.

    Args:
        rng: The generator's single random source.
        spec: The fixed parameters of the run.
        attempt: One-based number of this attempt, stored on acceptance.
        stochastic: Optional learned-policy verifier.

    How:
        Samples placement and doors, then runs the heuristic verifier and, when
        given, the stochastic one. A rejection returns the failing report so the
        caller can quote the last reason; an acceptance returns the world with
        its ``meta`` (attempt count, orientation, door data, reports) filled in.

    Returns:
        ``(world, {})`` on acceptance, ``(None, failure_report)`` on rejection.
    """
    try:
        world, orientation = _sample_world(rng, spec)
    except _PlacementError as failure:
        return None, {"placement_failure": str(failure)}
    report = verify_heuristic(world)
    if not report.solvable:
        return None, report.to_dict()
    meta = {
        "seed": spec.seed,
        "orientation": orientation,
        "attempts": attempt,
        "door_probability": spec.door_probability,
        "door_count": len(world.layout.doors),
        "rooms_per_side": spec.rooms_per_side,
        "verification": report.to_dict(),
    }
    if stochastic is not None:
        stochastic_report = stochastic.verify(world, seed=spec.seed)
        if not stochastic_report.passed:
            return None, stochastic_report.to_dict()
        meta["stochastic"] = stochastic_report.to_dict()
    return replace(world, meta=meta), {}


def generate_world(  # noqa: PLR0913 - keyword-only knobs of one public entry point
    seed: int,
    *,
    door_probability: float = DEFAULT_DOOR_PROBABILITY,
    rooms_per_side: int = 10,
    room_size: float = DEFAULT_ROOM_SIZE,
    regions: Sequence[str] = PAPER_REGION_ORDER,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    stochastic: StochasticVerifier | None = None,
) -> RoomsWorld:
    """Generate a verified n-rooms world from one seed.

    Args:
        seed: Seed for the single RNG that determines the whole world.
        door_probability: Independent probability that an internal wall opens.
        rooms_per_side: Number of rooms along each axis.
        room_size: Width and height of one room.
        regions: Region names in placement order; must start with ``Start`` and
            end with ``Goal``, matching ``PAPER_TOPOLOGY``.
        max_attempts: Maximum placement + door draws before failing.
        stochastic: Optional learned-policy verifier. When given, a world must
            pass both it and the heuristic verifier.

    How:
        Each attempt draws Start/Goal, the intermediates, then the doors from the
        same RNG. A failed heuristic or stochastic check redraws everything, so
        the placement distribution stays unconditional on door luck. The
        returned world records the attempt count, orientation, door probability
        and full verification report in ``meta``.

    Returns:
        The verified world.

    Raises:
        ValueError: If a parameter is out of range.
        LayoutGenerationError: If no verified world is found.
    """
    names = _validate_regions(regions)
    if names != PAPER_TOPOLOGY.vertices:
        raise ValueError(
            f"PAPER_TOPOLOGY expects vertices {PAPER_TOPOLOGY.vertices}, got {names}"
        )
    if not 0.0 <= door_probability <= 1.0:
        raise ValueError(f"door_probability must be in [0, 1], got {door_probability}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    spec = _DrawSpec(seed, rooms_per_side, room_size, names, door_probability)
    rng = np.random.default_rng(seed)
    last_report: dict = {}
    for attempt in range(1, max_attempts + 1):
        world, last_report = _attempt(rng, spec, attempt, stochastic)
        if world is not None:
            return world
    raise LayoutGenerationError(
        f"No verified layout for seed {seed} after {max_attempts} attempts; "
        f"last verification: {last_report}"
    )


__all__ = [
    "DEFAULT_DOOR_PROBABILITY",
    "DEFAULT_MAX_ATTEMPTS",
    "LayoutGenerationError",
    "corner_blocks",
    "generate_world",
]

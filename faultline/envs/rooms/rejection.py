"""Rejection-rate measurement for the n-rooms generator.

The generator re-draws a world until it verifies, so the share of draws that
survive is a property of the paper's sampling (door probability 0.5 sits near the
bond-percolation threshold). This module measures it with the real rejection
loop, so a run report can quote a number instead of a guess.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from faultline.envs.rooms.generator import (
    DEFAULT_DOOR_PROBABILITY,
    DEFAULT_MAX_ATTEMPTS,
    LayoutGenerationError,
    generate_world,
)
from faultline.envs.rooms.layout import DEFAULT_ROOM_SIZE


@dataclass(frozen=True)
class RejectionStats:
    """Acceptance and rejection measurements for the generator.

    Args:
        seeds: Number of seeds attempted.
        accepted: Seeds that produced a verified world within ``max_attempts``.
        failed: Seeds that exhausted ``max_attempts``.
        attempts: ``attempts -> count`` over accepted seeds.
        dead_branch_share: Share of accepted worlds with at least one dead edge.
        low_region_share: Share of accepted worlds with any region reachability
            below 1.0.

    How:
        ``acceptance_rate`` is accepted worlds per attempt, so its reciprocal is
        the mean number of draws per accepted world. The share fields describe
        what kinds of diagnostics solvable worlds carry.
    """

    seeds: int
    accepted: int
    failed: int
    attempts: Mapping[int, int]
    dead_branch_share: float
    low_region_share: float

    @property
    def acceptance_rate(self) -> float:
        """Return accepted worlds divided by total attempts.

        How:
            Divides accepted worlds by the total number of draws they took.

        Returns:
            The share of draws that were accepted, or zero with no draws.
        """
        total = sum(attempt * count for attempt, count in self.attempts.items())
        return self.accepted / total if total else 0.0

    @property
    def mean_attempts(self) -> float | None:
        """Return the mean attempts per accepted world, or ``None`` if none.

        How:
            Averages the attempt counts over accepted worlds.

        Returns:
            The mean draws per accepted world, or ``None`` if none were
            accepted.
        """
        if not self.accepted:
            return None
        total = sum(attempt * count for attempt, count in self.attempts.items())
        return total / self.accepted

    def to_dict(self) -> dict:
        """Convert the statistics to a JSON-serialisable mapping.

        How:
            Turns the attempt histogram keys into strings, which JSON requires,
            and adds the derived rates.

        Returns:
            The statistics as a JSON-serialisable mapping.
        """
        return {
            "seeds": self.seeds,
            "accepted": self.accepted,
            "failed": self.failed,
            "attempts": {str(attempt): count for attempt, count in sorted(self.attempts.items())},
            "acceptance_rate": self.acceptance_rate,
            "mean_attempts": self.mean_attempts,
            "dead_branch_share": self.dead_branch_share,
            "low_region_share": self.low_region_share,
        }


def measure_rejection(
    seeds: Iterable[int],
    *,
    door_probability: float = DEFAULT_DOOR_PROBABILITY,
    rooms_per_side: int = 10,
    room_size: float = DEFAULT_ROOM_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> RejectionStats:
    """Generate worlds for many seeds and aggregate the rejection statistics.

    Args:
        seeds: Seeds to attempt.
        door_probability: Door probability passed to the generator.
        rooms_per_side: Number of rooms along each axis.
        room_size: Width and height of one room.
        max_attempts: Per-seed attempt cap.

    How:
        Uses the real rejection loop for each seed. Records the attempts of each
        accepted world, counts seeds that exhaust the cap, and inspects each
        accepted world's stored verification report for a dead branch or a
        partly unreachable region. These are the numbers the run report should
        quote instead of a guessed rejection rate.

    Returns:
        The aggregated statistics.
    """
    attempts: dict[int, int] = {}
    accepted = 0
    failed = 0
    dead = 0
    low = 0
    seed_list = tuple(seeds)
    for seed in seed_list:
        try:
            world = generate_world(
                seed,
                door_probability=door_probability,
                rooms_per_side=rooms_per_side,
                room_size=room_size,
                max_attempts=max_attempts,
            )
        except LayoutGenerationError:
            failed += 1
            continue
        accepted += 1
        count = int(world.meta["attempts"])
        attempts[count] = attempts.get(count, 0) + 1
        verification = world.meta["verification"]
        if verification.get("dead_edges"):
            dead += 1
        if any(value < 1.0 for value in verification.get("region_reachability", {}).values()):
            low += 1
    return RejectionStats(
        seeds=len(seed_list),
        accepted=accepted,
        failed=failed,
        attempts=attempts,
        dead_branch_share=dead / accepted if accepted else 0.0,
        low_region_share=low / accepted if accepted else 0.0,
    )


__all__ = ["RejectionStats", "measure_rejection"]

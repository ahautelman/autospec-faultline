"""Behavioural tests for procedural n-rooms generation.

Determinism is the central contract: the same seed must produce byte-identical
world data. The remaining tests pin the paper's placement rules, the door
probability extremes, and the rejection loop's guarantee that every returned
world passes verification.
"""

from __future__ import annotations

from itertools import combinations

import pytest

from faultline.envs.rooms.generator import (
    LayoutGenerationError,
    corner_blocks,
    generate_world,
)
from faultline.envs.rooms.rejection import measure_rejection
from faultline.envs.rooms.stochastic import (
    StochasticReport,
)
from faultline.envs.rooms.verify import (
    VerificationReport,
    verify_heuristic,
)
from faultline.envs.rooms.world import PAPER_REGION_ORDER, world_to_dict

CORNER_LABEL_TO_PAIR = {
    "TL-BR": ("TL", "BR"),
    "BR-TL": ("BR", "TL"),
    "TR-BL": ("TR", "BL"),
    "BL-TR": ("BL", "TR"),
}


def test_same_seed_produces_identical_world_data():
    """Two calls with one seed serialise to the same mapping."""
    first = generate_world(11)
    second = generate_world(11)

    assert world_to_dict(first) == world_to_dict(second)
    assert first.meta["attempts"] == second.meta["attempts"]
    assert first.meta["orientation"] == second.meta["orientation"]


def test_start_and_goal_occupy_opposite_corners():
    """Placement follows one of the four ordered opposite-corner pairs."""
    corners = corner_blocks(10)

    for seed in range(10):
        world = generate_world(seed)
        label = world.meta["orientation"]
        start_corner, goal_corner = CORNER_LABEL_TO_PAIR[label]

        assert world.regions["Start"] == corners[start_corner]
        assert world.regions["Goal"] == corners[goal_corner]
        assert world.regions["Start"] != world.regions["Goal"]


def test_regions_are_inside_the_grid_and_do_not_overlap():
    """Every named region is a valid, pairwise disjoint 2x2 block."""
    for seed in range(10):
        world = generate_world(seed)
        blocks = list(world.regions.values())

        for block in blocks:
            assert 0 <= block.col <= 8
            assert 0 <= block.row <= 8
        for first, second in combinations(blocks, 2):
            assert first.overlaps(second) is False


def test_placement_order_is_the_paper_order():
    """The generated world carries every paper region."""
    world = generate_world(3)

    assert tuple(world.regions) == PAPER_REGION_ORDER
    assert tuple(world.topology.vertices) == PAPER_REGION_ORDER


def test_door_counts_stay_near_the_binomial_mean():
    """At p=0.5 the door count sits around half of the 180 walls."""
    counts = [len(generate_world(seed).layout.doors) for seed in range(20)]

    assert all(40 <= count <= 140 for count in counts)
    assert 70 <= sum(counts) / len(counts) <= 110
    # The room grid accepts every door as full-edge, so the analysis path works.
    generate_world(0).grid()


def test_door_probability_one_opens_every_wall():
    """p=1 gives all 180 full-edge doors and passes on the first attempt."""
    world = generate_world(0, door_probability=1.0)

    assert world.meta["attempts"] == 1
    assert len(world.layout.doors) == 180
    assert verify_heuristic(world).solvable is True


def test_door_probability_zero_exhausts_the_attempts():
    """With no doors no layout can be solvable, so the cap is reached."""
    with pytest.raises(LayoutGenerationError, match="after 3 attempts"):
        generate_world(0, door_probability=0.0, max_attempts=3)


def test_generator_never_returns_an_unverified_world():
    """Every returned world passes the verifier and records its attempts."""
    for seed in range(10):
        world = generate_world(seed)

        assert world.meta["attempts"] >= 1
        assert world.meta["verification"]["solvable"] is True
        assert verify_heuristic(world).solvable is True
        assert world.meta["door_count"] == len(world.layout.doors)


def test_measure_rejection_aggregates_accepted_worlds():
    """The measurement function reports attempts, shares and acceptance rate."""
    stats = measure_rejection(range(10))

    assert stats.seeds == 10
    assert stats.accepted == 10
    assert stats.failed == 0
    assert sum(stats.attempts.values()) == stats.accepted
    assert 0.0 < stats.acceptance_rate <= 1.0
    assert stats.mean_attempts is not None and stats.mean_attempts >= 1.0
    assert 0.0 <= stats.dead_branch_share <= 1.0
    assert 0.0 <= stats.low_region_share <= 1.0
    assert "acceptance_rate" in stats.to_dict()


class _StubVerifier:
    """A deterministic stochastic verifier for the two-stage protocol test."""

    def __init__(self, passed: bool) -> None:
        self.passed = passed

    def verify(self, world, *, seed: int) -> StochasticReport:
        return StochasticReport(
            learner="stub",
            seed=seed,
            budget=0,
            threshold=1.0,
            per_edge={"Start->Goal": 1.0 if self.passed else 0.0},
            best_path=1.0 if self.passed else 0.0,
        )


def test_stochastic_stage_can_accept_and_record_a_world():
    """A passing stochastic verifier is recorded in the world metadata."""
    world = generate_world(0, stochastic=_StubVerifier(True))

    assert world.meta["stochastic"]["passed"] is True
    assert world.meta["verification"]["solvable"] is True


def test_stochastic_stage_rejects_a_failing_world():
    """A failing stochastic verifier triggers the same redraw loop."""
    with pytest.raises(LayoutGenerationError):
        generate_world(0, max_attempts=3, stochastic=_StubVerifier(False))


def test_malformed_generator_arguments_fail_fast():
    """Bad probabilities, attempt caps, and region orders are rejected."""
    with pytest.raises(ValueError, match="door_probability"):
        generate_world(0, door_probability=1.5)
    with pytest.raises(ValueError, match="max_attempts"):
        generate_world(0, max_attempts=0)
    with pytest.raises(ValueError, match="Start"):
        generate_world(0, regions=("M1", "Goal"))


def test_verification_report_round_trips_inside_metadata():
    """The stored report decodes to the same verdict and diagnostics."""
    stored = generate_world(5).meta["verification"]

    report = VerificationReport.from_dict(stored)

    assert report.solvable is True
    assert report.to_dict() == stored


@pytest.mark.slow
def test_rejection_rate_measurement_over_one_thousand_seeds():
    """Measure the unconditional acceptance rate without freezing a constant."""
    stats = measure_rejection(range(1000))

    assert stats.seeds == 1000
    assert stats.accepted + stats.failed == stats.seeds
    assert 0.0 < stats.acceptance_rate < 1.0
    assert sum(stats.attempts.values()) == stats.accepted
    print(stats.to_dict())

"""The validity gate (spec 6.3, gate 1): a mutant must only coarsen the gold spec.

Two checks, both mechanical. Region containment: at every shared vertex and edge the
gold region is a subset of the faulty one, on the cell grid. Language inclusion: no
sampled trajectory satisfies the gold graph but not the faulty one. Failure is a bug
in a mutation, so :func:`assert_valid` raises instead of returning a quiet flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faultline.faults.graph_ops import find_edge
from faultline.faults.ground_truth import GroundTruthRepair
from faultline.faults.sampling import sample_trajectories
from faultline.golden.raster import CellGrid
from faultline.golden.spec import GoldSpec
from faultline.graph import models_trajectory

DEFAULT_SAMPLES = 300


class ValidityError(AssertionError):
    """Raised when a mutant is stricter than gold somewhere."""


@dataclass(frozen=True, eq=False)
class ValidityReport:
    """Outcome of the validity gate.

    Args:
        region_violations: Loci where a gold region is not inside the faulty one.
        counterexample: A trajectory gold accepts and the faulty graph rejects, or ``None``.
        n_sampled: Trajectories sampled.
        n_gold_accepted: Of those, how many gold accepts. Zero makes the language
            check vacuous.
    """

    region_violations: tuple[str, ...]
    counterexample: np.ndarray | None
    n_sampled: int
    n_gold_accepted: int

    @property
    def region_ok(self) -> bool:
        """Return whether every shared region only grew."""
        return not self.region_violations

    @property
    def language_ok(self) -> bool:
        """Return whether no counterexample was found and the check was not vacuous."""
        return self.counterexample is None and self.n_gold_accepted > 0

    @property
    def ok(self) -> bool:
        """Return whether both checks passed."""
        return self.region_ok and self.language_ok


def check_validity(
    repair: GroundTruthRepair,
    gold: GoldSpec,
    rng: np.random.Generator,
    n_samples: int = DEFAULT_SAMPLES,
) -> ValidityReport:
    """Run both validity checks on a mutant.

    Args:
        repair: The mutant.
        gold: The gold spec the mutant came from, for the grid and world size.
        rng: Seeded generator for trajectory sampling.
        n_samples: Number of trajectories to sample from the gold graph.

    How:
        Region containment compares masks of every shared vertex label and edge
        safe set that differs between the two graphs, so unchanged parts cost
        nothing. Language inclusion samples gold routes and checks each satisfying
        trajectory against the faulty graph.

    Returns:
        The report.
    """
    repair.faulty.assert_acyclic()
    grid = CellGrid(gold.layout, gold.resolution)
    violations = []
    for vertex in repair.gold.vertices:
        match = next((v for v in repair.faulty.vertices if v.name == vertex.name), None)
        if match is not None and match.region != vertex.region:
            if not (grid.mask(vertex.region) <= grid.mask(match.region)).all():
                violations.append(f"vertex {vertex.name}")
    for edge in repair.gold.edges:
        match = find_edge(repair.faulty, edge.source.name, edge.target.name)
        if match is not None and match.safe != edge.safe:
            if not (grid.mask(edge.safe) <= grid.mask(match.safe)).all():
                violations.append(f"edge {edge.source.name}->{edge.target.name}")

    samples = sample_trajectories(repair.gold, rng, n_samples, gold.layout.world_size)
    accepted, counterexample = 0, None
    for states in samples:
        trajectory = list(states)
        if models_trajectory(repair.gold, trajectory):
            accepted += 1
            if counterexample is None and not models_trajectory(repair.faulty, trajectory):
                counterexample = states
    return ValidityReport(tuple(violations), counterexample, len(samples), accepted)


def assert_valid(
    repair: GroundTruthRepair,
    gold: GoldSpec,
    rng: np.random.Generator,
    n_samples: int = DEFAULT_SAMPLES,
) -> ValidityReport:
    """Return the report, raising if the mutant is invalid.

    Args:
        repair: The mutant.
        gold: The gold spec.
        rng: Seeded generator.
        n_samples: Number of trajectories to sample.

    How:
        Runs :func:`check_validity` and raises on any violation, a found
        counterexample, or a vacuous language check.

    Returns:
        The passing report.

    Raises:
        ValidityError: If the mutant is stricter than gold or could not be checked.
    """
    report = check_validity(repair, gold, rng, n_samples)
    if not report.ok:
        raise ValidityError(
            f"{repair.kind.value}: regions={list(report.region_violations)}, "
            f"counterexample={'yes' if report.counterexample is not None else 'no'}, "
            f"gold-accepted samples={report.n_gold_accepted}/{report.n_sampled}"
        )
    return report

"""One call that scores a repaired graph against a mutant with every region-level metric."""

from __future__ import annotations

import numpy as np

from faultline.faults.ground_truth import GroundTruthRepair
from faultline.golden.raster import CellGrid
from faultline.graph import AbstractGraph
from faultline.metrics.audit import SoundnessAudit, soundness_audit
from faultline.metrics.ceiling import attainment, ceiling_scores, with_ceiling
from faultline.metrics.ged import region_ged
from faultline.metrics.regions import RegionScores, score_loci


def f1(removal: float | None, retention: float | None) -> float | None:
    """Return the harmonic mean of fault removal and collateral retention.

    How:
        Removal alone rewards deleting everything and retention alone rewards doing
        nothing; the harmonic mean is zero unless both are positive.

    Returns:
        The score, or ``None`` when either input is undefined.
    """
    if removal is None or retention is None:
        return None
    return 0.0 if removal + retention == 0 else 2 * removal * retention / (removal + retention)


def evaluate_repair(
    repair: GroundTruthRepair,
    repaired: AbstractGraph,
    grid: CellGrid,
    rng: np.random.Generator,
    *,
    audit_samples: int = 400,
) -> dict:
    """Score one repaired graph.

    Args:
        repair: The mutant, with or without ``ceiling`` filled in.
        repaired: The graph to score (an AutoSpec result or a baseline).
        grid: The shared cell grid.
        rng: Seeded generator for the trajectory audit.
        audit_samples: Trajectories per class in the audit.

    How:
        Computes, per locus, region scores, the oracle ceiling, attainment and F1 of
        removal and retention; then the region-aware GED to gold and the trajectory
        audit. Everything is deterministic given ``rng``.

    Returns:
        A JSON-friendly dict with ``loci``, ``ged`` and ``audit``.
    """
    repair = repair if repair.ceiling is not None else with_ceiling(repair, grid)
    scores = score_loci(grid, repair.target, repair.faulty, repaired)
    ceilings = ceiling_scores(repair, grid)
    loci = {}
    for locus, score in scores.items():
        ceiling = ceilings.get(locus)
        loci[f"{locus.kind}:{locus.name}"] = {
            "scores": None if score is None else _as_dict(score),
            "f1": None if score is None else f1(score.removal, score.retention),
            "attainment": attainment(score, ceiling),
        }
    audit: SoundnessAudit = soundness_audit(
        repair.gold, repair.faulty, repaired, rng, grid.layout.world_size, audit_samples
    )
    return {"loci": loci, "ged": region_ged(grid, repair.gold, repaired), "audit": audit.__dict__}


def _as_dict(score: RegionScores) -> dict:
    return dict(score.__dict__)

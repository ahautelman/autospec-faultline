"""Soundness audit and satisfaction-gap helpers for a repaired graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from faultline import stats
from faultline.faults.sampling import sample_trajectories
from faultline.graph import AbstractGraph, models_trajectory

AUDIT_SAMPLES = 400


@dataclass(frozen=True)
class SoundnessAudit:
    """What a repaired graph accepts and rejects, compared with gold and faulty.

    Args:
        gold_kept: Share of gold-satisfying trajectories the repaired graph still accepts
            (language-level retention; 1 minus the share of correct behaviour lost).
        fault_rejected: Share of trajectories the faulty graph accepts and gold rejects
            that the repaired graph now rejects (language-level fault removal).
        n_gold: Gold-satisfying trajectories sampled.
        n_fault: Fault-only trajectories sampled.
        gold_kept_interval: Two-sided 95% Clopper-Pearson interval of ``gold_kept``.
    """

    gold_kept: float | None
    fault_rejected: float | None
    n_gold: int
    n_fault: int
    gold_kept_interval: tuple[float, float] | None


def soundness_audit(
    gold: AbstractGraph,
    faulty: AbstractGraph,
    repaired: AbstractGraph,
    rng: np.random.Generator,
    world_size: float,
    count: int = AUDIT_SAMPLES,
) -> SoundnessAudit:
    """Sample trajectories and count how the repaired graph classifies them.

    Args:
        gold: The gold graph.
        faulty: The faulty graph.
        repaired: The repaired graph.
        rng: Seeded generator.
        world_size: Side of the world.
        count: Trajectories sampled from each of gold and faulty.

    How:
        Trajectories drawn from gold's routes that gold accepts measure collateral
        damage. Trajectories drawn from the faulty graph's routes that the faulty graph
        accepts and gold rejects are the fault's behaviour; the share the repair now
        rejects measures removal in the trajectory language, which is independent of
        how the regions were drawn. Both are sampling estimates, not proofs.

    Returns:
        The audit; a fraction is ``None`` when no trajectory of its class was found.
    """
    gold_set = [t for t in sample_trajectories(gold, rng, count, world_size) if models_trajectory(gold, list(t))]
    fault_set = [
        t for t in sample_trajectories(faulty, rng, count, world_size)
        if models_trajectory(faulty, list(t)) and not models_trajectory(gold, list(t))
    ]
    kept = sum(models_trajectory(repaired, list(t)) for t in gold_set)
    rejected = sum(not models_trajectory(repaired, list(t)) for t in fault_set)
    return SoundnessAudit(
        gold_kept=kept / len(gold_set) if gold_set else None,
        fault_rejected=rejected / len(fault_set) if fault_set else None,
        n_gold=len(gold_set),
        n_fault=len(fault_set),
        gold_kept_interval=stats.two_sided(kept, len(gold_set)) if gold_set else None,
    )


def satisfaction_gap(proxy: float, trajectories: Sequence[np.ndarray], gold: AbstractGraph) -> dict[str, float]:
    """Compare the loop's proxy success with the gold-spec satisfaction S*.

    Args:
        proxy: The repaired graph's best-path success under its own spec.
        trajectories: End-to-end trajectories of the repaired routes, started from gold's
            source region (a start the route does not cover is an empty trajectory).
        gold: The gold graph.

    How:
        S* is the share of trajectories gold accepts, with a Clopper-Pearson interval.
        The gap ``proxy - S*`` is the amount by which refinement flatters itself:
        a repair that shrinks the goal or source scores high on the proxy and low on S*.

    Returns:
        ``s_star``, ``s_star_low``, ``s_star_high`` and ``gap``.

    Raises:
        ValueError: If there are no trajectories.
    """
    if not trajectories:
        raise ValueError("satisfaction_gap needs at least one trajectory")
    ok = sum(len(t) > 0 and models_trajectory(gold, list(t)) for t in trajectories)
    low, high = stats.two_sided(ok, len(trajectories))
    s_star = ok / len(trajectories)
    return {"s_star": s_star, "s_star_low": low, "s_star_high": high, "gap": proxy - s_star}

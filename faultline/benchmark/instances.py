"""Instance generation: gold x family x severity x seed, with every rejection counted."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from faultline.faults.gates import Funnel, Stage
from faultline.faults.ground_truth import FaultKind, GroundTruthRepair, MutationNotApplicable, MutationParams
from faultline.faults.mutations import mutate
from faultline.faults.validity import check_validity
from faultline.golden.spec import GoldSpec

FAMILIES = (
    FaultKind.GOAL_INFLATION, FaultKind.HAZARD_UNLABELED, FaultKind.DEAD_START,
    FaultKind.WAYPOINT_CONTRACTION, FaultKind.ROUTE_PRUNING,
)
LEVELS = ("low", "mid", "high")


@dataclass(frozen=True)
class Instance:
    """One generated, validated mutant of a gold spec."""

    id: str
    gold: GoldSpec
    kind: FaultKind
    level: str
    seed: int
    repair: GroundTruthRepair


def instance_id(gold_id: str, kind: FaultKind, level: str, seed: int) -> str:
    """Return the stable identifier of an instance."""
    return f"{gold_id}|{kind.value}|{level}|{seed}"


def instance_rng(identifier: str) -> np.random.Generator:
    """Return a generator derived from the identifier, so generation is order independent."""
    digest = hashlib.sha256(identifier.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def generate(
    golds: Sequence[GoldSpec],
    *,
    families: Sequence[FaultKind] = FAMILIES,
    levels: Sequence[str] = LEVELS,
    seeds: Sequence[int],
    protocol,
    funnel: Funnel,
    audit_samples: int = 200,
) -> list[Instance]:
    """Generate every instance, record every rejection, and add the null instances.

    Args:
        golds: Certified (or Tier-1-valid) gold specs.
        families: Mutation families.
        levels: Severity levels.
        seeds: Instance seeds; each seed gives a different placement.
        protocol: The loaded protocol (severity grid, null fraction).
        funnel: Receives one record per attempt.
        audit_samples: Trajectories for the validity gate's language check.

    How:
        Each attempt draws its generator from the instance id alone. A mutation with no
        site is filed ``not_applicable``; one failing the validity gate is filed
        ``invalid``. Both are expected outcomes and are reported, never retried with a
        different seed. After the mutants, null instances are added so they make up at
        least ``null_fraction`` of the total, spread over the golds and seeds.

    Returns:
        The valid instances (mutants then nulls).
    """
    out: list[Instance] = []
    for gold in golds:
        for kind in families:
            # M4 and M5 have no severity level; run them once per seed.
            for level in (levels if kind not in (FaultKind.ROUTE_PRUNING,) else ("mid",)):
                for seed in seeds:
                    ident = instance_id(gold.id, kind, level, seed)
                    try:
                        repair = mutate(kind, gold, MutationParams(level=level), instance_rng(ident), protocol)
                    except MutationNotApplicable as exc:
                        funnel.record(kind, Stage.NOT_APPLICABLE, str(exc), ident)
                        continue
                    report = check_validity(repair, gold, instance_rng(ident + "|validity"), audit_samples)
                    if not report.ok:
                        funnel.record(kind, Stage.INVALID, _why(report), ident)
                        continue
                    funnel.record(kind, Stage.VALID)
                    out.append(Instance(ident, gold, kind, level, seed, repair))
    wanted = int(np.ceil(protocol["statistics"]["null_fraction"] * len(out) / (1 - protocol["statistics"]["null_fraction"]))) if out else 0
    nulls = [(g, s) for s in seeds for g in golds]
    for gold, seed in nulls[:wanted]:
        ident = instance_id(gold.id, FaultKind.NULL, "fixed", seed)
        funnel.record(FaultKind.NULL, Stage.VALID)
        out.append(Instance(ident, gold, FaultKind.NULL, "fixed", seed, mutate(FaultKind.NULL, gold, MutationParams(), instance_rng(ident))))
    return out


def _why(report) -> str:
    if report.region_violations:
        return "region not a coarsening: " + ", ".join(report.region_violations)
    if report.counterexample is not None:
        return "a gold-accepted trajectory is rejected by the faulty graph"
    return "language check vacuous (no gold-accepted trajectory sampled)"

"""Pre-run gates and the funnel (spec section 7): is a valid mutant worth running?

A mutant that passed the validity gate is a legitimate fault, but it may not be a *test*:
the faulty baseline may do as well as gold (the fault is harmless), or the loop may lack
the witnesses its operator needs (the fault is invisible to it). Both are decided here,
from baseline runs only, before any refinement run, and every rejection is counted.

* Effectiveness: on at least ``seed_fraction`` of the seeds, the faulty graph's unrefined
  best-path success trails gold's by ``min_gap`` and some faulty edge is below the
  refinement threshold, so the loop has a reason to act.
* Observability: the baseline rollouts at the fault's locus hold at least
  ``min_witnesses`` of the samples the expected operator consumes. An unobservable mutant
  is kept as its own stratum; it is not dropped.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from faultline.faults.ground_truth import FaultKind, GroundTruthRepair
from faultline.golden.spec import edge_id
from faultline.graph import AbstractGraph
from faultline.graph.semantics import best_path_product
from faultline.learners.base import EdgeEstimate

# Edge id -> baseline estimate, or ``None`` when the graph learner skipped the edge.
Baseline = Mapping[str, EdgeEstimate | None]


def learn_baseline(
    graph: AbstractGraph, env: Any, learner: Any, graph_learner: Any | None, *, budget: int,
    n_rollouts: int, seed: int,
) -> dict[str, EdgeEstimate | None]:
    """Train every edge once, with no refinement.

    Args:
        graph: The graph to learn over.
        env: The environment.
        learner: The edge learner.
        graph_learner: DiRL or LSTS, or ``None`` to train every edge.
        budget: Training steps per edge.
        n_rollouts: Rollouts per estimate.
        seed: Run seed.

    How:
        Mirrors the loop's first phase (graph learner or edge-by-edge training and
        estimating) and stops there, so the gate measures what Algorithm 1 sees at its
        first scan.

    Returns:
        Estimates keyed by edge id; skipped edges map to ``None``.
    """
    if graph_learner is not None:
        results = graph_learner.learn_graph(
            graph=graph, env=env, edge_learner=learner, budget=budget, n_rollouts=n_rollouts, seed=seed
        )
        learned = {edge_id(r.edge.source.name, r.edge.target.name): r.estimate for r in results}
        return {
            edge_id(e.source.name, e.target.name): learned.get(edge_id(e.source.name, e.target.name))
            for e in graph.edges
        }
    out: dict[str, EdgeEstimate | None] = {}
    for edge in graph.edges:
        policy = learner.train_edge(env, edge, budget, seed=seed)
        out[edge_id(edge.source.name, edge.target.name)] = learner.estimate_reach(
            env, edge, policy, n_rollouts, seed
        )
    return out


def best_path(graph: AbstractGraph, baseline: Baseline) -> float:
    """Return the best start-to-target path product of a baseline (unlearned edges count 0)."""
    probabilities = {}
    for edge in graph.edges:
        estimate = baseline.get(edge_id(edge.source.name, edge.target.name))
        probabilities[edge] = 0.0 if estimate is None else estimate.success_probability
    return best_path_product(graph, probabilities)


@dataclass(frozen=True)
class EffectivenessReport:
    """Outcome of the effectiveness gate.

    Args:
        effective: Whether enough seeds showed a gap and a below-threshold edge.
        gaps: Per seed, gold best-path success minus faulty best-path success.
        triggers: Per seed, whether some faulty edge was below the threshold.
        fraction: Share of seeds that passed both conditions.
    """

    effective: bool
    gaps: Mapping[int, float]
    triggers: Mapping[int, bool]
    fraction: float


def check_effectiveness(
    repair: GroundTruthRepair,
    gold_baselines: Mapping[int, Baseline],
    faulty_baselines: Mapping[int, Baseline],
    *,
    threshold: float,
    min_gap: float,
    seed_fraction: float,
) -> EffectivenessReport:
    """Decide whether the faulty baseline is measurably worse than gold's.

    Args:
        repair: The mutant.
        gold_baselines: Gold baseline per seed.
        faulty_baselines: Faulty baseline per seed; the same seeds as gold.
        threshold: The refinement threshold p.
        min_gap: Required best-path success deficit.
        seed_fraction: Share of seeds that must pass.

    How:
        Compares the unrefined best-path success of the two graphs per seed. The gap
        uses the faulty spec's own success estimate, which is the signal the loop reads.
        A seed also needs an edge under the threshold, otherwise the loop would accept
        the faulty graph without trying any operator.

    Returns:
        The report.

    Raises:
        ValueError: If the seed sets differ.
    """
    if set(gold_baselines) != set(faulty_baselines) or not gold_baselines:
        raise ValueError("Gold and faulty baselines must cover the same non-empty seed set")
    gaps: dict[int, float] = {}
    triggers: dict[int, bool] = {}
    for seed, faulty in faulty_baselines.items():
        gaps[seed] = best_path(repair.gold, gold_baselines[seed]) - best_path(repair.faulty, faulty)
        triggers[seed] = any(
            est is None or est.success_probability < threshold for est in faulty.values()
        )
    passing = sum(1 for s in gaps if gaps[s] >= min_gap and triggers[s])
    fraction = passing / len(gaps)
    return EffectivenessReport(fraction >= seed_fraction, gaps, triggers, fraction)


@dataclass(frozen=True)
class ObservabilityReport:
    """Outcome of the observability gate.

    Args:
        observable: Whether the locus edges held enough witnesses on every seed.
        witnesses: Per seed, the largest witness count over the locus edges, or
            ``None`` for a family whose operator needs no samples.
        kind_of_witness: What was counted.
    """

    observable: bool
    witnesses: Mapping[int, int | None]
    kind_of_witness: str


def locus_edges(repair: GroundTruthRepair) -> tuple[str, ...]:
    """Return the faulty edges at the fault's locus (incident edges, or the named edge)."""
    locus = repair.locus
    if locus.kind == "edge":
        return (locus.name,)
    if locus.kind == "vertex":
        return tuple(
            edge_id(e.source.name, e.target.name)
            for e in repair.faulty.edges
            if locus.name in (e.source.name, e.target.name)
        )
    return ()


def witness_count(kind: FaultKind, estimate: EdgeEstimate) -> int | None:
    """Count the samples the family's expected operator consumes.

    How:
        ReachRefine and AddRefine read successful rollouts, AvoidRefine reads rollouts
        that ended unsafe, PastRefine needs both successes and failures (the smaller
        count limits it), OrRefine reads no samples.

    Returns:
        The count, or ``None`` when the operator needs no samples.
    """
    successes = sum(1 for r in estimate.rollouts if r.success)
    if kind in (FaultKind.GOAL_INFLATION, FaultKind.WAYPOINT_CONTRACTION):
        return successes
    if kind is FaultKind.HAZARD_UNLABELED:
        return sum(1 for r in estimate.rollouts if not r.ended_safe)
    if kind is FaultKind.DEAD_START:
        return min(successes, len(estimate.rollouts) - successes)
    return None


_WITNESS_NAMES = {
    FaultKind.GOAL_INFLATION: "successful rollouts",
    FaultKind.WAYPOINT_CONTRACTION: "successful rollouts",
    FaultKind.HAZARD_UNLABELED: "rollouts that ended unsafe",
    FaultKind.DEAD_START: "min(successes, failures)",
}


def check_observability(
    repair: GroundTruthRepair, faulty_baselines: Mapping[int, Baseline], *, min_witnesses: int
) -> ObservabilityReport:
    """Decide whether the baseline rollouts carry enough witnesses for the operator.

    Args:
        repair: The mutant.
        faulty_baselines: Faulty baseline per seed.
        min_witnesses: Required witness count (protocol ``observability_min_witnesses``).

    How:
        Takes, per seed, the best witness count over the locus edges and requires it to
        reach the minimum on every seed. Route pruning and the null need no samples and
        are observable by construction.

    Returns:
        The report.
    """
    if repair.kind in (FaultKind.ROUTE_PRUNING, FaultKind.NULL):
        return ObservabilityReport(True, {s: None for s in faulty_baselines}, "none needed")
    edges = locus_edges(repair)
    counts: dict[int, int | None] = {}
    for seed, baseline in faulty_baselines.items():
        found = [
            witness_count(repair.kind, baseline[e]) for e in edges if baseline.get(e) is not None
        ]
        counts[seed] = max((c for c in found if c is not None), default=0)
    ok = all((c or 0) >= min_witnesses for c in counts.values())
    return ObservabilityReport(ok, counts, _WITNESS_NAMES[repair.kind])


class Stage(str, Enum):
    """Where a generated instance leaves the funnel."""

    NOT_APPLICABLE = "not_applicable"
    VALID = "valid"
    INVALID = "invalid"
    INEFFECTIVE = "ineffective"
    UNOBSERVABLE = "unobservable"
    ACCEPTED = "accepted"


@dataclass
class Funnel:
    """Counts per family of every instance, with the reasons for each rejection.

    How:
        ``record`` files one attempt: at generation as ``not_applicable``, ``invalid`` or ``valid``, then a valid one again under its gate outcome. ``accepted`` instances are
        valid, effective and observable; ``unobservable`` instances are valid and
        effective and are reported as their own stratum, not discarded.
    """

    counts: dict[str, Counter] = field(default_factory=dict)
    reasons: list[dict[str, str]] = field(default_factory=list)

    def record(self, kind: FaultKind, stage: Stage, reason: str = "", instance: str = "") -> None:
        """File one attempt."""
        self.counts.setdefault(kind.value, Counter())[stage.value] += 1
        if stage not in (Stage.ACCEPTED, Stage.VALID):
            self.reasons.append(
                {"kind": kind.value, "stage": stage.value, "instance": instance, "reason": reason}
            )

    def generated(self, kind: FaultKind) -> int:
        """Return the number of generation attempts for a family.

        Gate stages (ineffective, unobservable, accepted) refine ``valid`` and are not added.
        """
        return self._attempts(self.counts.get(kind.value, Counter()))

    @staticmethod
    def _attempts(counter: Counter) -> int:
        return sum(counter[s.value] for s in (Stage.NOT_APPLICABLE, Stage.INVALID, Stage.VALID))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "counts": {k: dict(v) | {"generated": self._attempts(v)} for k, v in self.counts.items()},
            "rejections": list(self.reasons),
        }


@dataclass(frozen=True)
class GateResult:
    """Terminal stage of one mutant with the reports that led there."""

    stage: Stage
    effectiveness: EffectivenessReport | None = None
    observability: ObservabilityReport | None = None
    reason: str = ""


def screen(
    repair: GroundTruthRepair,
    gold_baselines: Mapping[int, Baseline],
    faulty_baselines: Mapping[int, Baseline],
    *,
    threshold: float,
    min_gap: float,
    seed_fraction: float,
    min_witnesses: int,
) -> GateResult:
    """Run both gates on a valid mutant and name its terminal stage.

    Args:
        repair: A mutant that passed the validity gate.
        gold_baselines: Gold baseline per seed.
        faulty_baselines: Faulty baseline per seed.
        threshold: Refinement threshold p.
        min_gap: Effectiveness gap.
        seed_fraction: Effectiveness seed share.
        min_witnesses: Observability minimum.

    How:
        Effectiveness first (an ineffective mutant is not a test whatever it shows),
        then observability.

    Returns:
        The result; the null is accepted without gating.
    """
    if repair.kind is FaultKind.NULL:
        return GateResult(Stage.ACCEPTED)
    eff = check_effectiveness(
        repair, gold_baselines, faulty_baselines,
        threshold=threshold, min_gap=min_gap, seed_fraction=seed_fraction,
    )
    if not eff.effective:
        return GateResult(
            Stage.INEFFECTIVE, eff, None,
            f"only {eff.fraction:.0%} of seeds show a gap >= {min_gap} with an edge below {threshold}",
        )
    obs = check_observability(repair, faulty_baselines, min_witnesses=min_witnesses)
    if not obs.observable:
        return GateResult(
            Stage.UNOBSERVABLE, eff, obs, f"fewer than {min_witnesses} {obs.kind_of_witness}"
        )
    return GateResult(Stage.ACCEPTED, eff, obs)


def screen_with_learning(
    repair: GroundTruthRepair,
    gold_baselines: Mapping[int, Baseline],
    seeds: Sequence[int],
    learn: Callable[[AbstractGraph, int], Baseline],
    **gate_args: Any,
) -> GateResult:
    """Learn the faulty baseline on every seed, then screen.

    Args:
        repair: A valid mutant.
        gold_baselines: Gold baselines, cached per gold and seed by the caller.
        seeds: Gate seeds.
        learn: ``(graph, seed) -> baseline``; wraps :func:`learn_baseline`.
        **gate_args: Arguments of :func:`screen`.

    Returns:
        The result.
    """
    faulty = {int(s): learn(repair.faulty, int(s)) for s in seeds}
    return screen(repair, gold_baselines, faulty, **gate_args)

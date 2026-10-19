"""Learning-based certification of a gold spec (spec section 5.2, tiers 2 and 3).

Tier 1 (feasibility) is exact and lives in :func:`faultline.golden.builder.build_gold`.
This module runs the learner: Tier 2 checks that every gold edge is learnable with a
Clopper-Pearson lower bound above ``p_gold_min`` on every certification seed, Tier 3
checks that Algorithm 1 accepts no refinement on gold, on held-out seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from faultline import stats
from faultline.golden.spec import Certificate, GoldSpec, edge_id
from faultline.refinement import AutoSpecConfig, run_autospec


def certify_learning(
    gold: GoldSpec,
    make_env: Callable[[], Any],
    make_learner: Callable[[], Any],
    make_graph_learner: Callable[[], Any | None],
    *,
    config: AutoSpecConfig,
    cert_seeds: Sequence[int],
    holdout_seeds: Sequence[int],
    p_gold_min: float,
    confidence: float = 0.95,
    protocol_hash: str = "",
    run: Callable[..., Any] = run_autospec,
) -> GoldSpec:
    """Run Tier 2 and Tier 3 and attach the result to the gold spec.

    Args:
        gold: A gold spec that passed Tier 1.
        make_env: Builds a fresh environment for the gold layout and horizon.
        make_learner: Builds a fresh edge learner.
        make_graph_learner: Builds a fresh graph learner, or returns ``None``.
        config: AutoSpec configuration; its seed is replaced per run.
        cert_seeds: Seeds for Tier 2.
        holdout_seeds: Seeds for Tier 3, disjoint from ``cert_seeds``.
        p_gold_min: Required lower bound for every edge and seed.
        confidence: One-sided confidence of the lower bound.
        protocol_hash: Hash of the protocol file.
        run: The loop entry point; injectable so tests can script it.

    How:
        Tier 2 reads each edge's first estimate (before any refinement) from a run
        on every certification seed and converts it to a success count over
        ``config.n_rollouts``. An edge the graph learner left unlearned fails the
        tier, because a gold spec must have no dead edge. Tier 3 runs the full loop
        on each held-out seed and requires zero committed refinements. Tier 1 is
        not repeated: a failed Tier 1 returns the spec unchanged.

    Returns:
        The gold spec with a certificate carrying all three tiers.

    Raises:
        ValueError: If the seed sets overlap.
    """
    if set(cert_seeds) & set(holdout_seeds):
        raise ValueError("Certification and hold-out seeds must be disjoint")
    if not gold.certificate.tier1_ok:
        return gold

    problems = list(gold.certificate.problems)
    lowers: dict[str, list[float]] = {}
    for seed in cert_seeds:
        result = run(
            gold.graph, make_env(), make_learner(), replace(config, seed=int(seed)), make_graph_learner()
        )
        for entry in result.initial_edge_success:
            key = edge_id(entry.source, entry.target)
            if entry.probability is None:
                lowers.setdefault(key, []).append(0.0)
                problems.append(f"edge {key} was not learned on seed {seed}")
                continue
            successes = round(entry.probability * config.n_rollouts)
            lowers.setdefault(key, []).append(
                stats.one_sided_lower(successes, config.n_rollouts, confidence)
            )
    tier2 = all(value >= p_gold_min for values in lowers.values() for value in values)
    for key, values in lowers.items():
        if min(values) < p_gold_min:
            problems.append(f"edge {key} lower bound {min(values):.3f} < {p_gold_min}")

    tier3 = True
    for seed in holdout_seeds:
        result = run(
            gold.graph, make_env(), make_learner(), replace(config, seed=int(seed)), make_graph_learner()
        )
        accepted = [step for step in result.steps if step.operator is not None]
        if accepted:
            tier3 = False
            names = ", ".join(f"{s.source}->{s.target}:{s.operator}" for s in accepted)
            problems.append(f"gold was refined on hold-out seed {seed}: {names}")

    certificate = Certificate(
        tier1_ok=True,
        tier2_ok=tier2,
        tier3_ok=tier3,
        cert_seeds=tuple(int(s) for s in cert_seeds),
        edge_cp_lower={key: tuple(values) for key, values in lowers.items()},
        p_gold=p_gold_min,
        protocol_hash=protocol_hash,
        problems=tuple(problems),
    )
    return replace(gold, certificate=certificate)

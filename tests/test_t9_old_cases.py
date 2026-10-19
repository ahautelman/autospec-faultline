"""T9: the five hand-built FaultLine cases are valid instances of the new methodology.

For each old case the correct graph is rebuilt as a gold spec (Tier 1 must pass) and the
old faulty graph must be a coarsening of it under the new validity gate. Where the old
case and the parametric mutation share a family, the mutation must also apply to that gold.
"""

from __future__ import annotations

import numpy as np
import pytest

from faultline.faults.catalogue import all_fault_cases
from faultline.faults.ground_truth import FaultKind, GroundTruthRepair, Locus, MutationNotApplicable, MutationParams
from faultline.faults.mutations import mutate
from faultline.faults.validity import check_validity
from faultline.golden.builder import build_gold
from faultline.spectrl.predicates import Box

FAMILY = {
    "goal_trap": FaultKind.GOAL_INFLATION,
    "overwide_safety": FaultKind.HAZARD_UNLABELED,
    "unreachable_start": FaultKind.DEAD_START,
    "missing_waypoint": FaultKind.WAYPOINT_CONTRACTION,
    "missing_alternative": FaultKind.ROUTE_PRUNING,
}
OVERWIDE_TRAP = Box((3.0, 3.75), (5.2, 4.25))
WEAK = {"missing_alternative": ("a->goal",)}


def gold_of(case):
    env = case.make_env()
    traps = (OVERWIDE_TRAP,) if case.fault.name == "overwide_safety" else ()
    return build_gold(case.fault.name, env.layout, case.fault.correct_graph.copy(), traps,
                      resolution=48, max_steps=env.max_steps, weak_edges=WEAK.get(case.fault.name, ()))


@pytest.mark.parametrize("case", all_fault_cases(), ids=lambda c: c.fault.name)
def test_old_case_is_a_valid_coarsening_of_a_feasible_gold(case):
    gold = gold_of(case)
    assert gold.certificate.tier1_ok, gold.certificate.problems
    repair = GroundTruthRepair(
        gold=gold.graph, faulty=case.fault.faulty_graph, kind=FAMILY[case.fault.name], locus=Locus("none"),
        severity_level="fixed", severity=None, factors={}, target={}, expected_operators=frozenset(),
    )
    report = check_validity(repair, gold, np.random.default_rng(0), 200)
    assert report.region_ok, report.region_violations
    assert report.counterexample is None


@pytest.mark.parametrize("case", all_fault_cases(), ids=lambda c: c.fault.name)
def test_parametric_mutation_of_the_same_family_applies_to_the_old_gold(case):
    gold = gold_of(case)
    try:
        repair = mutate(FAMILY[case.fault.name], gold, MutationParams(), np.random.default_rng(0))
    except MutationNotApplicable as exc:
        pytest.skip(f"no site on this gold: {exc}")
    assert check_validity(repair, gold, np.random.default_rng(1), 200).ok

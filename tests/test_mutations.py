"""The five mutation families on small certified golds: placement, flags, validity, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from faultline.envs.rooms.layout import RoomsLayout, full_edge_interval
from faultline.faults.graph_ops import edge_named, vertex_named
from faultline.faults.ground_truth import FaultKind, MutationNotApplicable, MutationParams
from faultline.faults.mutations import mutate, null_mutation
from faultline.faults.validity import ValidityError, assert_valid, check_validity
from faultline.golden.builder import build_gold, chain_graph
from faultline.golden.raster import CellGrid
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import Box, Not

RES = 24
LEVELS = ("low", "mid", "high")


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def gold_pocket():
    """Goal inside a solid-walled room, so every other room is a sealed pocket site."""
    return build_gold("pocket", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (), Box((4, 1), (8, 7))), resolution=RES, max_steps=20)


def gold_trap():
    trap = Box((3, 2), (6, 6))
    return build_gold("trap", RoomsLayout(doors={}),
                      chain_graph(Box((1, 3), (2, 5)), (), Box((7, 3), (8, 5))), (trap,),
                      resolution=RES, max_steps=20)


def gold_big_start():
    return build_gold("start", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (5, 5)), (), Box((6, 1), (8, 7))), resolution=RES, max_steps=20)


def gold_centre_start():
    return build_gold("centre", RoomsLayout(doors={}),
                      chain_graph(Box((9, 9), (15, 11)), (), Box((9, 13), (15, 15))), resolution=RES, max_steps=20)


def gold_corridor():
    doors = {key: full_edge_interval(key, 8.0) for key in [("v", 0, 0), ("v", 1, 0)]}
    graph = AbstractGraph()
    start = graph.add_vertex("start", Box((1, 1), (3, 3)))
    u = graph.add_vertex("u", Box((9, 1), (15, 7)))
    goal = graph.add_vertex("goal", Box((18, 1), (22, 3)))
    graph.add_edge(start, u)
    graph.add_edge(u, goal, Not(Box((10, 0), (11, 8))))  # spec-only mouth hazard on the outgoing edge
    graph.source, graph.target = start, goal
    return build_gold("corridor", RoomsLayout(doors=doors), graph, resolution=RES, max_steps=14)


def gold_chain():
    return build_gold("chain", RoomsLayout(doors={}),
                      chain_graph(Box((0.5, 3), (1.5, 5)),
                                  (Box((2.5, 3), (3.5, 5)), Box((4.5, 3), (5.5, 5))),
                                  Box((6.5, 3), (7.5, 5))), resolution=RES, max_steps=10)


def gold_diamond():
    graph = AbstractGraph()
    start = graph.add_vertex("start", Box((0.5, 3.0), (1.5, 4.0)))
    a = graph.add_vertex("a", Box((0.5, 0.5), (1.5, 1.5)))
    goal = graph.add_vertex("goal", Box((7.0, 0.5), (7.9, 1.5)))
    b = graph.add_vertex("b", Box((4.0, 0.5), (5.0, 1.5)))
    graph.add_edge(start, a)
    graph.add_edge(a, goal)
    graph.add_edge(a, b)
    graph.add_edge(b, goal)
    graph.source, graph.target = start, goal
    return build_gold("diamond", RoomsLayout(doors={}), graph, resolution=RES, max_steps=2,
                      weak_edges=("a->goal",))


def test_fixtures_are_certified_feasible():
    for gold in (gold_pocket(), gold_trap(), gold_big_start(), gold_centre_start(), gold_corridor(),
                 gold_chain(), gold_diamond()):
        assert gold.certificate.tier1_ok, (gold.id, gold.certificate.problems)


# ---- goal inflation -------------------------------------------------------------------

def test_goal_inflation_adds_an_unreachable_pocket_of_the_requested_mass():
    gold = gold_pocket()
    repair = mutate(FaultKind.GOAL_INFLATION, gold, MutationParams(level="mid"), rng())
    grid = CellGrid(gold.layout, gold.resolution)
    pocket = Box(repair.factors["pocket"][:2], repair.factors["pocket"][2:])
    reached = grid.reachable(grid.mask(gold.graph.source.region), np.ones((RES, RES), dtype=bool))
    assert not (grid.mask(pocket) & reached).any()
    assert repair.factors["mass"] == pytest.approx(0.25, abs=0.06)
    assert repair.factors["hull_recoverable"] is True
    assert repair.expected_operators == {"seq"}


def test_goal_inflation_mass_grows_with_the_level():
    masses = [mutate(FaultKind.GOAL_INFLATION, gold_pocket(), MutationParams(level=level), rng()).factors["mass"]
              for level in LEVELS]
    assert masses[0] < masses[1] < masses[2]


def test_goal_inflation_needs_an_incoming_edge():
    with pytest.raises(MutationNotApplicable):
        mutate(FaultKind.GOAL_INFLATION, gold_pocket(), MutationParams(vertex="start"), rng())


# ---- hazard unlabeled -----------------------------------------------------------------

def test_hazard_unlabeling_labels_only_the_far_part_of_the_trap():
    gold = gold_trap()
    repair = mutate(FaultKind.HAZARD_UNLABELED, gold, MutationParams(level="mid"), rng())
    gold_edge, faulty_edge = gold.graph.edges[0], repair.faulty.edges[0]
    approach, core = np.array([3.5, 4.0]), np.array([5.5, 4.0])
    assert not gold_edge.safe.evaluate(approach) and faulty_edge.safe.evaluate(approach)
    assert not gold_edge.safe.evaluate(core) and not faulty_edge.safe.evaluate(core)
    assert repair.factors["unlabeled_fraction"] == pytest.approx(2 / 3)
    assert repair.factors["axis"] == "x"
    assert repair.factors["trap_path_distance"] == 0.0


def test_hazard_unlabeling_grows_with_the_level():
    cells = [mutate(FaultKind.HAZARD_UNLABELED, gold_trap(), MutationParams(level=level), rng()).factors["unlabeled_cells"]
             for level in LEVELS]
    assert cells[0] <= cells[1] <= cells[2] and cells[0] < cells[2]


def test_hazard_unlabeling_without_a_trap_is_not_applicable():
    with pytest.raises(MutationNotApplicable):
        mutate(FaultKind.HAZARD_UNLABELED, gold_pocket(), MutationParams(), rng())


# ---- dead start -----------------------------------------------------------------------

def test_dead_start_adds_cells_with_no_route_to_the_goal():
    gold = gold_big_start()
    repair = mutate(FaultKind.DEAD_START, gold, MutationParams(level="mid"), rng())
    grid = CellGrid(gold.layout, gold.resolution)
    pocket = Box(repair.factors["pockets"][0][:2], repair.factors["pockets"][0][2:])
    assert not (grid.mask(pocket) & gold.viability["start->goal"]).any()
    assert repair.factors["halfplane_separable"] is True and repair.factors["has_upstream"] is False
    assert repair.expected_operators == {"past"}


def test_dead_start_at_an_inner_vertex_is_reachable_from_upstream_but_not_viable():
    gold = gold_corridor()
    repair = mutate(FaultKind.DEAD_START, gold, MutationParams(vertex="u", level="mid"), rng())
    grid = CellGrid(gold.layout, gold.resolution)
    pocket = grid.mask(Box(repair.factors["pockets"][0][:2], repair.factors["pockets"][0][2:]))
    safe_in = grid.mask(gold.graph.edges[0].safe)
    reached = grid.reachable(grid.mask(gold.graph.source.region) & safe_in, safe_in)
    assert (pocket <= reached).all()
    assert not (pocket & gold.viability["u->goal"]).any()
    assert repair.factors["has_upstream"] is True


def test_two_pockets_on_opposite_sides_defeat_a_single_half_plane():
    gold = gold_centre_start()
    repair = mutate(FaultKind.DEAD_START, gold, MutationParams(level="mid", pockets=2), rng())
    one, two = (np.array(p).reshape(2, 2) for p in repair.factors["pockets"])
    region = gold.graph.source.region
    centre = (region.lower + region.upper) / 2.0
    first, second = one.mean(axis=0), two.mean(axis=0)
    # Independent geometry: the segment between the pocket centres crosses the region.
    midpoint_t = np.clip(np.dot(centre - first, second - first) / np.dot(second - first, second - first), 0, 1)
    nearest = first + midpoint_t * (second - first)
    assert region.evaluate(nearest)
    assert repair.factors["halfplane_separable"] is False
    single = mutate(FaultKind.DEAD_START, gold, MutationParams(level="mid"), rng())
    assert single.factors["halfplane_separable"] is True


# ---- waypoint contraction -------------------------------------------------------------

def test_waypoint_contraction_drops_the_waypoint_nearest_the_target_position():
    gold = gold_chain()
    mid = mutate(FaultKind.WAYPOINT_CONTRACTION, gold, MutationParams(level="mid"), rng())
    high = mutate(FaultKind.WAYPOINT_CONTRACTION, gold, MutationParams(level="high"), rng())
    assert mid.factors["dropped"] == ("w1",) and high.factors["dropped"] == ("w2",)
    assert [v.name for v in high.faulty.vertices] == ["start", "w1", "goal"]
    assert high.factors["witness_aligned"] == pytest.approx(abs(high.factors["position"] - 0.5))


def test_dropping_two_waypoints_joins_start_to_goal():
    repair = mutate(FaultKind.WAYPOINT_CONTRACTION, gold_chain(), MutationParams(n_dropped=2), rng())
    assert repair.factors["dropped"] == ("w1", "w2")
    assert [(e.source.name, e.target.name) for e in repair.faulty.edges] == [("start", "goal")]


def test_waypoint_contraction_needs_a_droppable_waypoint():
    with pytest.raises(MutationNotApplicable):
        mutate(FaultKind.WAYPOINT_CONTRACTION, gold_pocket(), MutationParams(), rng())


# ---- route pruning --------------------------------------------------------------------

def test_route_pruning_removes_an_edge_that_cuts_every_alternative():
    gold = gold_diamond()
    repair = mutate(FaultKind.ROUTE_PRUNING, gold, MutationParams(), rng())
    assert repair.factors["removed"] in {"a->b", "b->goal"}
    assert repair.factors["weak_edge"] == "a->goal"
    assert len(repair.faulty.edges) == len(gold.graph.edges) - 1
    assert repair.expected_operators == {"or"} and repair.severity is None


def test_route_pruning_needs_a_weak_edge():
    with pytest.raises(MutationNotApplicable):
        mutate(FaultKind.ROUTE_PRUNING, gold_chain(), MutationParams(), rng())


# ---- validity and determinism (T5) ---------------------------------------------------

CASES = [
    (FaultKind.GOAL_INFLATION, gold_pocket, {}),
    (FaultKind.HAZARD_UNLABELED, gold_trap, {}),
    (FaultKind.DEAD_START, gold_big_start, {}),
    (FaultKind.DEAD_START, gold_corridor, {"vertex": "u"}),
    (FaultKind.DEAD_START, gold_centre_start, {"pockets": 2}),
    (FaultKind.WAYPOINT_CONTRACTION, gold_chain, {}),
    (FaultKind.WAYPOINT_CONTRACTION, gold_chain, {"n_dropped": 2}),
    (FaultKind.ROUTE_PRUNING, gold_diamond, {}),
]


@pytest.mark.parametrize("kind, make_gold, extra", CASES)
def test_every_mutation_only_coarsens_gold_across_levels_and_seeds(kind, make_gold, extra):
    gold = make_gold()
    for level in LEVELS:
        for seed in range(4):
            repair = mutate(kind, gold, MutationParams(level=level, **extra), rng(seed))
            report = assert_valid(repair, gold, rng(100 + seed), n_samples=120)
            assert report.n_gold_accepted > 0


@pytest.mark.parametrize("kind, make_gold, extra", CASES)
def test_mutations_are_deterministic_given_the_seed(kind, make_gold, extra):
    gold = make_gold()
    first = mutate(kind, gold, MutationParams(**extra), rng(7))
    second = mutate(kind, gold, MutationParams(**extra), rng(7))
    assert dict(first.factors) == dict(second.factors)
    assert [(e.source.name, e.target.name, e.safe.to_string()) for e in first.faulty.edges] == \
           [(e.source.name, e.target.name, e.safe.to_string()) for e in second.faulty.edges]
    assert [(v.name, v.region.to_string()) for v in first.faulty.vertices] == \
           [(v.name, v.region.to_string()) for v in second.faulty.vertices]


def test_the_null_mutation_is_gold_and_valid():
    gold = gold_pocket()
    repair = null_mutation(gold)
    assert repair.kind is FaultKind.NULL and repair.faulty is not gold.graph
    assert_valid(repair, gold, rng(), n_samples=40)


def test_a_stricter_than_gold_mutant_is_caught_by_both_checks():
    gold = gold_pocket()
    repair = null_mutation(gold)
    vertex_named(repair.faulty, "goal").region = Box((6, 3), (7, 5))  # a strict subset of gold
    report = check_validity(repair, gold, rng(), n_samples=120)
    assert report.region_violations == ("vertex goal",)
    assert report.counterexample is not None
    with pytest.raises(ValidityError):
        assert_valid(repair, gold, rng(), n_samples=120)


def test_a_stricter_edge_safe_set_is_caught():
    gold = gold_trap()
    repair = null_mutation(gold)
    edge_named(repair.faulty, "start", "goal").safe = Not(Box((0, 0), (8, 8)))
    assert check_validity(repair, gold, rng(), n_samples=60).region_violations == ("edge start->goal",)

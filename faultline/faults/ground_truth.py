"""Types for parametric fault injection: kinds, loci, parameters, and the repair record."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import Predicate

LEVELS = ("low", "mid", "high")


class FaultKind(str, Enum):
    """The five mutation families, one per refinement operator, plus the null."""

    GOAL_INFLATION = "goal_inflation"
    HAZARD_UNLABELED = "hazard_unlabeled"
    DEAD_START = "dead_start"
    WAYPOINT_CONTRACTION = "waypoint_contraction"
    ROUTE_PRUNING = "route_pruning"
    NULL = "null"


# A priori taxonomy (spec 6.5): the operator whose repair matches the fault.
EXPECTED_OPERATORS: Mapping[FaultKind, frozenset[str]] = {
    FaultKind.GOAL_INFLATION: frozenset({"seq"}),
    FaultKind.HAZARD_UNLABELED: frozenset({"seq"}),
    FaultKind.DEAD_START: frozenset({"past"}),
    FaultKind.WAYPOINT_CONTRACTION: frozenset({"add"}),
    FaultKind.ROUTE_PRUNING: frozenset({"or"}),
    FaultKind.NULL: frozenset(),
}

# Fallbacks that mirror configs/protocol.toml; pass a protocol to override them.
DEFAULT_SEVERITY: Mapping[FaultKind, tuple[float, float, float]] = {
    FaultKind.GOAL_INFLATION: (0.10, 0.25, 0.50),
    FaultKind.DEAD_START: (0.10, 0.25, 0.50),
    FaultKind.HAZARD_UNLABELED: (0.25, 0.50, 0.75),
    FaultKind.WAYPOINT_CONTRACTION: (0.5, 0.25, 0.75),
}

_PROTOCOL_KEYS = {
    FaultKind.GOAL_INFLATION: "pocket_mass",
    FaultKind.DEAD_START: "pocket_mass",
    FaultKind.HAZARD_UNLABELED: "trap_unlabelled_fraction",
    FaultKind.WAYPOINT_CONTRACTION: "waypoint_position",
}


class MutationNotApplicable(Exception):
    """Raised when a gold spec has no site where a mutation can be placed.

    It is an expected outcome, counted in the funnel as a generation failure,
    never an error in the benchmark.
    """


@dataclass(frozen=True)
class Locus:
    """Where a fault sits: a vertex label or an edge.

    Args:
        kind: ``"vertex"``, ``"edge"`` or ``"none"``.
        name: Vertex name, or edge id ``"a->b"``.
    """

    kind: str
    name: str = ""


@dataclass(frozen=True)
class MutationParams:
    """Knobs of one mutation.

    Args:
        level: ``"low"``, ``"mid"`` or ``"high"``.
        severity: Numeric severity, or ``None`` to use the level's default.
        vertex: Vertex to mutate, or ``None`` for the family's default.
        edge: Edge id to mutate, or ``None`` for the family's default.
        pockets: Number of pockets (M3); two make the half-plane fit fail.
        n_dropped: Waypoints dropped (M4).
        trap_index: Which gold trap to partly unlabel (M2).
    """

    level: str = "mid"
    severity: float | None = None
    vertex: str | None = None
    edge: str | None = None
    pockets: int = 1
    n_dropped: int = 1
    trap_index: int = 0

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")


def severity_value(kind: FaultKind, params: MutationParams, protocol=None) -> float | None:
    """Return the numeric severity of a mutation.

    Args:
        kind: The mutation family.
        params: The mutation parameters.
        protocol: Optional loaded protocol; its severity grid overrides the defaults.

    How:
        An explicit ``params.severity`` wins. Otherwise the level indexes the
        family's grid. Route pruning has no numeric severity.

    Returns:
        The severity, or ``None`` for families without one.
    """
    if params.severity is not None:
        return float(params.severity)
    if kind not in _PROTOCOL_KEYS:
        return None
    grid = DEFAULT_SEVERITY[kind]
    if protocol is not None:
        grid = tuple(protocol["severity"][_PROTOCOL_KEYS[kind]])
    return float(grid[LEVELS.index(params.level)])


@dataclass(frozen=True, eq=False)
class GroundTruthRepair:
    """One mutant with everything needed to score a repair of it.

    Args:
        gold: The gold graph (phi*).
        faulty: The faulty graph (phi_f).
        kind: The mutation family.
        locus: Where the fault is.
        severity_level: ``low``, ``mid``, ``high`` or ``fixed``.
        severity: Numeric severity, or ``None``.
        factors: Recoverability and diagnostic factors recorded on the mutant.
        target: Gold region at each locus (R*).
        expected_operators: A priori operator set for this family.
        ceiling: Best repair the operator's geometry can reach with oracle
            witnesses; filled in by a later step.
        oracle_valid_operators: Operators that pass the threshold with oracle
            witnesses; filled in by a later step.
    """

    gold: AbstractGraph
    faulty: AbstractGraph
    kind: FaultKind
    locus: Locus
    severity_level: str
    severity: float | None
    factors: Mapping[str, object]
    target: Mapping[Locus, Predicate]
    expected_operators: frozenset[str]
    ceiling: Mapping[Locus, Predicate] | None = None
    oracle_valid_operators: frozenset[str] | None = None
    weak_edges: frozenset[str] = field(default_factory=frozenset)

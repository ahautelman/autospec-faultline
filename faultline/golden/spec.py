"""Records for a gold specification and its certificate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from faultline.envs.rooms.layout import RoomsLayout
from faultline.graph import AbstractGraph
from faultline.spectrl.predicates import Predicate

EdgeId = str  # "source->target"


def edge_id(source: str, target: str) -> EdgeId:
    """Return the canonical identifier of an edge.

    Args:
        source: Source vertex name.
        target: Target vertex name.

    How:
        Joins the two names with ``->``.

    Returns:
        The edge identifier.
    """
    return f"{source}->{target}"


@dataclass(frozen=True)
class Certificate:
    """What has been established about a gold spec, and under which protocol.

    Args:
        tier1_ok: Feasibility (exact, on the cell grid).
        tier2_ok: Learnability, or ``None`` when not run.
        tier3_ok: Fixpoint under Algorithm 1, or ``None`` when not run.
        cert_seeds: Seeds the learning tiers used.
        edge_cp_lower: Per-edge one-sided lower bounds, one per certification seed.
        p_gold: Minimum lower bound a gold edge must reach.
        p_alt: Minimum lower bound for the edges of the alternative route behind a
            weak-by-design edge.
        protocol_hash: Hash of the protocol file the run used.
        problems: Human-readable reasons for any failed check.
        notes: Remarks that do not fail a tier (for example, refinements AutoSpec
            made on a weak-by-design edge).

    How:
        "Certified" means certified under the protocol, never proven: refinement is
        undecidable in general and neither DiRL nor LSTS is complete.
    """

    tier1_ok: bool
    tier2_ok: bool | None = None
    tier3_ok: bool | None = None
    cert_seeds: tuple[int, ...] = ()
    edge_cp_lower: Mapping[EdgeId, tuple[float, ...]] = field(default_factory=dict)
    p_gold: float = 0.0
    p_alt: float = 0.0
    protocol_hash: str = ""
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def certified(self) -> bool:
        """Return whether all three tiers ran and passed.

        How:
            ``None`` (not run) counts as not certified.

        Returns:
            ``True`` only when every tier is ``True``.
        """
        return self.tier1_ok is True and self.tier2_ok is True and self.tier3_ok is True


@dataclass(frozen=True, eq=False)
class GoldSpec:
    """A gold specification built from environment ground truth.

    Args:
        id: Identifier used in reports.
        env_id: Name of the environment or world.
        layout: The physical layout shared with every faulty variant.
        traps: Physical absorbing traps, which the gold safe sets exclude.
        graph: The gold abstract graph.
        viability: Per-edge masks of the cells from which the edge can succeed.
        weak_edges: Edges that are weak by design (for example too long for the
            horizon). Each needs an alternative route of very reliable edges.
        shrinks: Per-vertex fraction of the intended region removed to make it
            feasible (``0.0`` means the intended box was kept as is).
        resolution: Cells per axis of the analysis grid.
        max_steps: Episode horizon the gold spec was built for.
        certificate: The certificate so far.
    """

    id: str
    env_id: str
    layout: RoomsLayout
    traps: tuple[Predicate, ...]
    graph: AbstractGraph
    viability: Mapping[EdgeId, np.ndarray]
    weak_edges: frozenset[EdgeId]
    shrinks: Mapping[str, float]
    resolution: int
    max_steps: int
    certificate: Certificate

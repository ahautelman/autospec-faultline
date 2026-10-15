"""Core data structures for FaultLine fault injection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from faultline.graph import AbstractGraph


class FaultType(str, Enum):
    GOAL_TRAP = "goal_trap"
    OVERWIDE_SAFETY = "overwide_safety"
    UNREACHABLE_START = "unreachable_start"
    MISSING_WAYPOINT = "missing_waypoint"
    MISSING_ALTERNATIVE = "missing_alternative"
    NONE = "none"


@dataclass(frozen=True)
class FaultCase:
    """One controlled benchmark instance with known ground truth."""

    name: str
    fault_type: FaultType
    expected_operator: str | None
    correct_graph: AbstractGraph
    faulty_graph: AbstractGraph
    description: str
    affected_vertex: str | None = None
    affected_edge: tuple[str, str] | None = None
    severity: float | None = None

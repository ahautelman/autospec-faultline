"""FaultLine controlled fault-injection API."""

from faultline.faults.base import FaultCase, FaultType
from faultline.faults.catalogue import (
    FaultBenchmarkCase,
    all_fault_cases,
    goal_trap_case,
    missing_alternative_case,
    missing_waypoint_case,
    overwide_safety_case,
    unreachable_start_case,
)
from faultline.faults.environments import HiddenHazardRooms9
from faultline.faults.injectors import (
    inject_goal_trap,
    inject_missing_alternative_path,
    inject_missing_waypoint,
    inject_overwide_safety,
    inject_unreachable_start,
    no_fault_control,
)
from faultline.faults.runner import FaultSummary, format_summary, summarize_fault

__all__ = [
    "FaultBenchmarkCase",
    "FaultCase",
    "FaultSummary",
    "FaultType",
    "HiddenHazardRooms9",
    "all_fault_cases",
    "format_summary",
    "goal_trap_case",
    "inject_goal_trap",
    "inject_missing_alternative_path",
    "inject_missing_waypoint",
    "inject_overwide_safety",
    "inject_unreachable_start",
    "missing_alternative_case",
    "missing_waypoint_case",
    "no_fault_control",
    "overwide_safety_case",
    "summarize_fault",
    "unreachable_start_case",
]

"""The 9-rooms validation scenarios and their runner (paper sections 8 and 11).

Re-exports the five scenario builders, the registry, and the runner so the
validation suite and the tests import from ``faultline.scenarios`` directly. Each
scenario isolates one symptom row of the section 11 table and maps to the
refinement operator that row expects.
"""

from faultline.scenarios.catalogue import (
    SCENARIOS,
    Scenario,
    blocked_two_path,
    long_corridor,
    mixed_start,
    narrow_door,
    resolve_scenario,
    scenario_names,
    trap_goal,
)
from faultline.scenarios.runner import EdgeOutcome, ScenarioResult, run_scenario

__all__ = [
    "SCENARIOS",
    "EdgeOutcome",
    "Scenario",
    "ScenarioResult",
    "blocked_two_path",
    "long_corridor",
    "mixed_start",
    "narrow_door",
    "resolve_scenario",
    "run_scenario",
    "scenario_names",
    "trap_goal",
]

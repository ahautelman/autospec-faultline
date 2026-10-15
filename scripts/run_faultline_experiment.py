"""Easy entry point for the FaultLine experiment.

Recommended order:
1. RUN_SET = "overwide" in quick mode to validate the revised safety case.
2. RUN_SET = "all" in quick mode for all five faults, one seed.
3. MODE = "full" only after every quick case has usable evidence.

Run from the project root:
    uv run --extra interop python scripts/run_faultline_experiment.py
"""

from faultline.experiments.faultline_benchmark import (
    FaultLineSettings,
    run_faultline_experiment,
)
from faultline.faults import (
    all_fault_cases,
    goal_trap_case,
    missing_alternative_case,
    missing_waypoint_case,
    overwide_safety_case,
    unreachable_start_case,
)


MODE = "full"          # "quick" or "full"
METHOD = "dirl"         # "base", "dirl", or "lsts"
LEARNER = "ppo"         # "ppo" or "tabular"

# "goal_trap", "overwide", "unreachable", "waypoint", "alternative", or "all"
RUN_SET = "all"


_CASES = {
    "goal_trap": (goal_trap_case(),),
    "overwide": (overwide_safety_case(),),
    "unreachable": (unreachable_start_case(),),
    "waypoint": (missing_waypoint_case(),),
    "alternative": (missing_alternative_case(),),
    "all": all_fault_cases(),
}

if RUN_SET not in _CASES:
    raise ValueError(f"Unknown RUN_SET {RUN_SET!r}")

CASES = _CASES[RUN_SET]


if MODE == "quick":
    SETTINGS = FaultLineSettings(
        learner=LEARNER,
        method=METHOD,
        threshold=0.80,
        budget=20_000,
        n_rollouts=30,
        avoid_tail_length=3,
        seeds=(0,),
        quality_threshold=0.80,
        grid_resolution=101,
        output_directory=f"results/faultline_quick_{METHOD}_{RUN_SET}",
        lsts_chunk_size=2_500,
        lsts_max_interactions=20_000,
    )

elif MODE == "full":
    SETTINGS = FaultLineSettings(
        learner=LEARNER,
        method=METHOD,
        threshold=0.80,
        budget=80_000,
        n_rollouts=100,
        avoid_tail_length=3,
        seeds=(0, 1, 2, 3, 4),
        quality_threshold=0.80,
        grid_resolution=121,
        output_directory=f"results/faultline_full_{METHOD}_{RUN_SET}",
        lsts_chunk_size=5_000,
        lsts_max_interactions=80_000,
    )

else:
    raise ValueError("MODE must be 'quick' or 'full'")


def main() -> None:
    run_faultline_experiment(CASES, SETTINGS)


if __name__ == "__main__":
    main()

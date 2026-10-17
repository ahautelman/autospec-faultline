"""Single entry point for the FaultLine experiments.

Recommended order:
1. Keep MODE="quick", METHOD="lsts", RUN_SET="all" and run one seed.
2. If all five cases finish sensibly, change MODE="full".
3. For a final apples-to-apples comparison, rerun DiRL with METHOD="dirl"
   after the evaluation-seed change in this patch.

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


# ---------------------------------------------------------------------------
# Change only these three values for normal use.
# ---------------------------------------------------------------------------

MODE = "full"           # "quick" or "full"
METHOD = "lsts"          # "dirl", "lsts", or "base"
RUN_SET = "all"          # one case name below, or "all"

LEARNER = "ppo"


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
        avoid_tail_length=1,
        seeds=(0,),
        quality_threshold=0.80,
        grid_resolution=101,
        output_directory=f"results/faultline_quick_{METHOD}_{RUN_SET}",
        lsts_chunk_size=4_096,
        # None is intentional: LSTS now receives budget per usable edge.
        lsts_max_interactions=None,
        # Quick mode keeps the useful per-run diagnostics on screen.
        verbose=True,
        write_individual_json=False,
    )

elif MODE == "full":
    SETTINGS = FaultLineSettings(
        learner=LEARNER,
        method=METHOD,
        threshold=0.80,
        budget=80_000,
        n_rollouts=100,
        avoid_tail_length=1,
        seeds=(0, 1, 2, 3, 4),
        quality_threshold=0.80,
        grid_resolution=121,
        output_directory=f"results/faultline_full_{METHOD}_{RUN_SET}",
        lsts_chunk_size=4_096,
        # Fair comparison with DiRL: LSTS total budget is computed as
        # 80k * number of usable graph edges.
        lsts_max_interactions=None,
        # Full runs print one compact line per seed instead of large blocks.
        verbose=False,
        write_individual_json=False,
    )

else:
    raise ValueError("MODE must be 'quick' or 'full'")


def main() -> None:
    run_faultline_experiment(CASES, SETTINGS)


if __name__ == "__main__":
    main()

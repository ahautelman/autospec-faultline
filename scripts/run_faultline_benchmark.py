"""Easy entry point for the FaultLine benchmark.

Run from the project root:

    uv run --extra interop python scripts/run_faultline_benchmark.py

For a very fast debugging run using the deterministic planner:

    set LEARNER = "tabular"

inside this file.

The real benchmark should use PPO.
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


# ============================================================================
# EDIT THESE SETTINGS
# ============================================================================

# Real experiment:
LEARNER = "ppo"

# Choose:
#
#   "base"  -> AutoSpec directly
#   "dirl"  -> AutoSpec + DiRL
#   "lsts"  -> AutoSpec + LSTS
#
METHOD = "dirl"


# ---------------------------------------------------------------------------
# QUICK DEVELOPMENT SETTINGS
# ---------------------------------------------------------------------------
#
# These are deliberately small.
# They are for checking whether the complete benchmark pipeline works.
#
# THRESHOLD = 0.50
THRESHOLD = 0.99
# BUDGET = 20_000
BUDGET = 80_000
ROLLOUTS = 50

# SEEDS = (
#     0,
# )

SEEDS = (
    0, 1, 2, 3, 4,
)


# ---------------------------------------------------------------------------
# WHICH CASES TO RUN
# ---------------------------------------------------------------------------

# Start with ONE case while debugging:
# CASES = (
#     goal_trap_case(),
# )
CASES = all_fault_cases()


# Once that works, replace the block above with:
#
# CASES = all_fault_cases()


# ---------------------------------------------------------------------------
# LSTS SETTINGS
# ---------------------------------------------------------------------------

LSTS_EPSILON = 0.30
LSTS_ALPHA = 0.20
LSTS_TOLERANCE = 0.10
LSTS_CHUNK_SIZE = 5_000

# None means use BUDGET as the total LSTS interaction budget.
LSTS_MAX_INTERACTIONS = None


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

OUTPUT_DIRECTORY = (
    "results/faultline_dirl"
)


# ============================================================================
# RUN
# ============================================================================


def main() -> None:

    settings = FaultLineSettings(
        learner=LEARNER,
        method=METHOD,

        threshold=THRESHOLD,
        budget=BUDGET,
        n_rollouts=ROLLOUTS,

        seeds=SEEDS,

        quality_threshold=0.80,
        grid_resolution=121,

        output_directory=(
            OUTPUT_DIRECTORY
        ),

        lsts_epsilon=(
            LSTS_EPSILON
        ),

        lsts_alpha=(
            LSTS_ALPHA
        ),

        lsts_tolerance=(
            LSTS_TOLERANCE
        ),

        lsts_chunk_size=(
            LSTS_CHUNK_SIZE
        ),

        lsts_max_interactions=(
            LSTS_MAX_INTERACTIONS
        ),
    )

    run_faultline_experiment(
        CASES,
        settings,
    )


if __name__ == "__main__":
    main()
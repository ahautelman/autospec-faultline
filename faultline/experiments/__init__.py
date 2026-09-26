"""Runnable reporting experiments for FaultLine.

The 9-rooms report trains PPO edge policies with periodic checkpoints and writes
comparison figures against the paper's section 11 reference numbers. This package
re-exports the pipeline entry points, the record types, and the figure writer so
callers and the CLI share one surface.

The training functions need the ``interop`` extra and the figures need the
``reports`` extra; the package itself imports neither, so importing it is cheap.
"""

from faultline.experiments.figures import write_figures
from faultline.experiments.pipeline import (
    default_workers,
    route_product,
    run_multi_seed,
    run_reported_scenario,
    run_single_seed,
    source_edge,
)
from faultline.experiments.records import (
    Checkpoint,
    Overlay,
    ScenarioReport,
    ScenarioSpread,
    SeedSummary,
    success_at,
)
from faultline.experiments.training import (
    DEFAULT_BUDGET,
    DEFAULT_INTERMEDIATE_ROLLOUTS,
    DEFAULT_INTERVAL,
    DEFAULT_ROLLOUTS,
    ReportSettings,
    TrainingRecord,
    train_with_checkpoints,
)

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_INTERMEDIATE_ROLLOUTS",
    "DEFAULT_INTERVAL",
    "DEFAULT_ROLLOUTS",
    "Checkpoint",
    "Overlay",
    "ReportSettings",
    "ScenarioReport",
    "ScenarioSpread",
    "SeedSummary",
    "TrainingRecord",
    "default_workers",
    "route_product",
    "run_multi_seed",
    "run_reported_scenario",
    "run_single_seed",
    "source_edge",
    "success_at",
    "train_with_checkpoints",
    "write_figures",
]

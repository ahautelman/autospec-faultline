"""Find saved runs in a data directory and turn them into run views.

A data directory may hold the full-loop reports, the toy single-bottleneck reports, or
the toy multi-seed summaries. Files are recognised by what they contain, not by name, so
a renamed or relocated file still loads, and anything unrecognised is reported rather
than guessed at.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from faultline.experiments.autospec.records import report_from_dict as autospec_report_from_dict
from faultline.experiments.records import (
    read_json,
    report_from_dict as scenario_report_from_dict,
    spread_from_dict,
)
from faultline.scenarios import SCENARIOS, Scenario, resolve_scenario
from faultline.viz.run import (
    RunView,
    run_view_from_report,
    run_view_from_scenario_report,
    run_views_from_spread,
)

_AUTOSPEC_KEYS = {"steps", "best_path_before", "scenario"}
_SCENARIO_REPORT_KEYS = {"overlay", "operator", "before", "after"}
_SPREAD_KEYS = {"summaries", "budget", "scenario"}


@dataclass(frozen=True)
class RunData:
    """Everything found in a data directory.

    Args:
        runs: One view per saved run, used for the overlay and learning curves.
        seed_runs: One view per seed from a multi-seed summary, used for the per-seed
            and paper-comparison figures when no full reports cover those seeds.
        notes: Files that were skipped, each with the reason.
    """

    runs: tuple[RunView, ...] = ()
    seed_runs: tuple[RunView, ...] = ()
    notes: tuple[str, ...] = field(default=())


def load_data(directory: Path, catalogue: Mapping[str, Scenario] = SCENARIOS) -> RunData:
    """Load every recognisable run under a directory.

    Args:
        directory: The folder to search, recursively, for ``.json`` files.
        catalogue: Scenarios used for runs that did not record their own room.

    How:
        Reads each JSON file and decides what it is from its keys. A toy report or
        summary needs its scenario's room, so one whose scenario is not in the
        catalogue is skipped with a note, as is any file that is not a run.

    Returns:
        The loaded views, ordered as the catalogue lists its scenarios and then by
        seed so figures list them the same way every time, and the notes for skipped
        files.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        ValueError: If two runs claim the same scenario and seed, which means the
            directory mixes different experiments.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory {directory} does not exist")
    runs: list[RunView] = []
    seed_runs: list[RunView] = []
    notes: list[str] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            _load_file(path, catalogue, runs, seed_runs)
        except _Skip as skip:
            notes.append(f"skipped {path.name}: {skip}")
    _reject_duplicates(runs)
    order = {name: index for index, name in enumerate(catalogue)}
    return RunData(
        tuple(sorted(runs, key=lambda run: _order_key(run, order))),
        tuple(sorted(seed_runs, key=lambda run: _order_key(run, order))),
        tuple(notes),
    )


def _order_key(run: RunView, order: Mapping[str, int]) -> tuple[int, str, int]:
    """Sort catalogue scenarios in catalogue order, then others by name, then by seed."""
    return order.get(run.scenario, len(order)), run.scenario, run.seed


class _Skip(Exception):
    """A file that is not a drawable run, with the reason."""


def _load_file(
    path: Path, catalogue: Mapping[str, Scenario], runs: list[RunView], seed_runs: list[RunView]
) -> None:
    """Classify one file and add its views to the right list."""
    payload = read_json(path)
    keys = set(payload) if isinstance(payload, dict) else set()
    if _AUTOSPEC_KEYS <= keys:
        runs.append(run_view_from_report(autospec_report_from_dict(payload), catalogue))
    elif _SCENARIO_REPORT_KEYS <= keys:
        report = scenario_report_from_dict(payload)
        runs.append(run_view_from_scenario_report(report, _scenario(report.scenario, catalogue)))
    elif _SPREAD_KEYS <= keys:
        spread = spread_from_dict(payload)
        seed_runs.extend(run_views_from_spread(spread, _scenario(spread.scenario, catalogue)))
    else:
        raise _Skip("not a run report")


def _scenario(name: str, catalogue: Mapping[str, Scenario]) -> Scenario:
    """Look up a scenario for a toy file, or skip the file.

    How:
        Prefers the caller's catalogue and then the shared resolver, so an
        n-rooms toy file whose name is only ``rooms100_random_<seed>`` is still
        rebuildable from its seed.
    """
    scenario = catalogue.get(name)
    if scenario is not None:
        return scenario
    try:
        return resolve_scenario(name)
    except KeyError as missing:
        raise _Skip(f"scenario {name!r} is not in the catalogue") from missing


def _reject_duplicates(runs: list[RunView]) -> None:
    """Fail if two runs share a scenario and seed."""
    seen: set[tuple[str, int]] = set()
    for run in runs:
        key = (run.scenario, run.seed)
        if key in seen:
            raise ValueError(
                f"Two runs for scenario {run.scenario!r} seed {run.seed}; the directory "
                "seems to mix different experiments"
            )
        seen.add(key)

"""The per-scenario summary table of a full-loop batch."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from faultline.experiments.autospec.records import AutoSpecReport

HEADER = (
    "scenario",
    "expected",
    "matched",
    "best path before",
    "best path after",
    "paper before",
    "paper after",
    "mean seconds",
)


def summary_table(reports: Sequence[AutoSpecReport]) -> str:
    """Render the reports as a Markdown table, one row per scenario.

    Args:
        reports: Every report of the batch, any order.

    How:
        Groups reports by scenario in first-seen order and averages the best-path
        success and the run time over seeds. ``matched`` counts the seeds that
        committed the paper's expected operator at least once.

    Returns:
        The Markdown table text.
    """
    by_scenario: dict[str, list[AutoSpecReport]] = {}
    for report in sorted(reports, key=lambda r: r.seed):
        by_scenario.setdefault(report.scenario, []).append(report)
    rows = [HEADER, tuple("---" for _ in HEADER)]
    rows += [_row(group) for group in by_scenario.values()]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _row(group: list[AutoSpecReport]) -> tuple[str, ...]:
    """Build one scenario's table row from its per-seed reports.

    Args:
        group: The scenario's reports, one per seed.

    How:
        Counts the seeds that matched the expected operator and averages the
        best-path success and run time. A 100-rooms world has no single expected
        operator: its per-edge expectations are printed from the recorded
        scenario metadata and the match column says ``n/a``, because the plan
        reports those expectations without scoring against them. A missing paper
        reference is printed as an em dash.

    Returns:
        The row's cells, as text.
    """
    first = group[0]
    matched = "n/a" if first.expected_operator is None else (
        f"{sum(report.matched_expected for report in group)}/{len(group)}"
    )
    return (
        first.scenario,
        _expected_text(first),
        matched,
        f"{mean(r.best_path_before for r in group):.3f}",
        f"{mean(r.best_path_after for r in group):.3f}",
        _reference_text(first.reference_before),
        _reference_text(first.reference_after),
        f"{mean(r.elapsed_seconds for r in group):.0f}",
    )


def _expected_text(report: AutoSpecReport) -> str:
    """Return the expected-operator cell for one scenario.

    How:
        Prefers the per-edge expectations recorded in the scenario metadata,
        which is what the designed 100-rooms worlds carry. Falls back to the
        single expected operator of a 9-rooms scenario, then to ``none``.
    """
    metadata = report.scenario_metadata or {}
    per_edge = metadata.get("expected_operators") or {}
    if per_edge:
        return ", ".join(
            f"{edge}: {'/'.join(operators)}" for edge, operators in per_edge.items()
        )
    return report.expected_operator or "none"


def _reference_text(value: float | None) -> str:
    """Return a paper-reference cell, or an em dash when there is no reference."""
    return "—" if value is None else f"{value:.2f}"

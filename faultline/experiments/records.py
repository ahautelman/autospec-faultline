"""Report records and JSON persistence for the 9-rooms report.

The reporting pipeline is split so training and plotting are decoupled: a training
pass writes these records to JSON, and a plotting pass reads them back. Every
record holds plain numbers and strings, so a figure is regenerated without
retraining a policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The paper's comparable budget: the section 11 table was produced at 80k steps,
# so the reference comparison reads each series at this step.
REFERENCE_STEP = 80_000


@dataclass(frozen=True)
class Checkpoint:
    """One success estimate taken mid-training.

    Args:
        step: Environment steps completed when the estimate was taken.
        probability: Reach success over the evaluation rollouts.

    How:
        A point on a learning curve; a series is ordered by ``step``.
    """

    step: int
    probability: float


def success_at(series: tuple[Checkpoint, ...], step: int) -> float:
    """Return the series value at ``step``.

    Args:
        series: An ordered checkpoint series.
        step: The step to read.

    How:
        Prefers the exact step; otherwise falls back to the latest checkpoint at
        or before ``step``, and finally to the first checkpoint. That fallback
        keeps a short smoke run plottable when the reference step was never
        reached.

    Returns:
        The success probability for ``step``.

    Raises:
        ValueError: If ``series`` is empty.
    """
    if not series:
        raise ValueError("Checkpoint series is empty")
    earlier = [checkpoint for checkpoint in series if checkpoint.step <= step]
    if earlier:
        return max(earlier, key=lambda checkpoint: checkpoint.step).probability
    for checkpoint in series:
        if checkpoint.step >= step:
            return checkpoint.probability
    return series[-1].probability


@dataclass(frozen=True)
class Overlay:
    """The geometry a refinement operator consumed and produced.

    Args:
        kind: ``"hull"``, ``"waypoint"``, ``"half_plane"``, or ``"edge"``.
        points: Sampled states the operator consumed, as ``(x, y)`` pairs. For a
            ``seq`` overlay these are the target-region visits ReachRefine hulls.
        counter_points: Contrasting samples, non-empty only for a half-plane
            split, where they are the failed starts.
        normal: Half-plane normal, or ``None`` for the other kinds.
        offset: Half-plane offset, or ``None`` for the other kinds.
        added_edge: Vertex names of the structural edge OrRefine added, or
            ``None``.
        avoid_points: Unsafe trajectory-tail states, as ``(x, y)`` pairs, that
            AvoidRefine hulls and subtracts from the safe region. Non-empty only
            for a ``seq`` overlay with at least one rollout that ended outside the
            safe region, so an edge whose only sub-effect is ReachRefine keeps an
            empty tuple.

    How:
        Stores primitive values only, so the overlay reloads from JSON and the
        figure layer redraws the geometry without the original rollouts. Both
        ``points`` and ``avoid_points`` are captured for a ``seq`` overlay, so the
        figure can show both halves of the operator even when only one of them
        changed the graph.
    """

    kind: str
    points: tuple[tuple[float, float], ...]
    counter_points: tuple[tuple[float, float], ...]
    normal: tuple[float, float] | None
    offset: float | None
    added_edge: tuple[str, str] | None
    avoid_points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ScenarioReport:
    """The single-seed report for one scenario.

    Args:
        scenario: Scenario name.
        operator: Refinement operator applied to the bottleneck edge.
        seed: Training seed.
        budget: Training budget in steps for each edge.
        before: The pre-refinement edge's checkpoint series.
        after: The refined route's success series, the route product.
        overlay: Geometry for the overlay figure.
        reference_before: The paper's before number from the scenario, or ``None``
            when the paper gives none.
        reference_after: The paper's after number from the scenario, or ``None``.
        operator_refused: Whether the scenario's operator was attempted but
            declined for lack of a witness; ``False`` when there was no operator
            or the refinement succeeded.
        satisfaction_before: Fraction of end-to-end episodes in which the
            unrefined policy satisfies the original edge's specification (region
            semantics, starts from the original source region), or ``None`` for an
            older report.
        satisfaction_after: The same fraction for the refined route run end to
            end, judged against the original edge; equal to
            ``satisfaction_before`` when nothing was refined.

    How:
        Bundles the two curves, the overlay inputs, and the paper's reference so
        one JSON file per scenario carries everything the figures need. The two
        curves score the learner's goal-point task per edge (the paper's
        reporting); the two satisfaction numbers score the original
        specification, so a refinement cannot look better merely by sampling
        easier starts or goals.
    """

    scenario: str
    operator: str
    seed: int
    budget: int
    before: tuple[Checkpoint, ...]
    after: tuple[Checkpoint, ...]
    overlay: Overlay
    reference_before: float | None
    reference_after: float | None
    operator_refused: bool = False
    satisfaction_before: float | None = None
    satisfaction_after: float | None = None


@dataclass(frozen=True)
class SeedSummary:
    """One seed's before and after success at the two reported checkpoints.

    Args:
        seed: Training seed.
        before_80k: Pre-refinement success at the reference step.
        after_80k: Refined success at the reference step.
        before_final: Pre-refinement success at the budget.
        after_final: Refined success at the budget.

    How:
        The four numbers the per-seed spread figure aggregates.
    """

    seed: int
    before_80k: float
    after_80k: float
    before_final: float
    after_final: float


@dataclass(frozen=True)
class ScenarioSpread:
    """The multi-seed summaries for one scenario.

    Args:
        scenario: Scenario name.
        budget: Training budget used.
        summaries: One :class:`SeedSummary` per seed, in seed order.

    How:
        Holds the per-seed numbers so the spread figure is redrawn from JSON.
    """

    scenario: str
    budget: int
    summaries: tuple[SeedSummary, ...]


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload, creating the parent directory first.

    Args:
        path: Destination file path.
        payload: JSON-serialisable mapping.

    How:
        Creates the parent directory, then writes indented JSON. A filesystem
        error is re-raised as a :class:`RuntimeError` naming the path.

    Raises:
        RuntimeError: If the directory or file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as err:
        raise RuntimeError(f"Could not write {path}: {err}") from err


def read_json(path: Path) -> dict:
    """Read a JSON object from disk.

    Args:
        path: Source file path.

    How:
        Reads the file and parses it. A filesystem or parse error is re-raised as
        a :class:`RuntimeError` naming the path.

    Returns:
        The parsed JSON mapping.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise RuntimeError(f"Could not read {path}: {err}") from err


def report_to_dict(report: ScenarioReport) -> dict:
    """Convert a report to a JSON-serialisable mapping.

    Args:
        report: The single-seed report to render.

    How:
        Renders both curves as pair lists and the overlay as a nested mapping,
        keeping the scenario metadata alongside them.

    Returns:
        The report as a JSON-ready mapping.
    """
    return {
        "scenario": report.scenario,
        "operator": report.operator,
        "seed": report.seed,
        "budget": report.budget,
        "before": _series_to_pairs(report.before),
        "after": _series_to_pairs(report.after),
        "overlay": _overlay_to_dict(report.overlay),
        "reference_before": report.reference_before,
        "reference_after": report.reference_after,
        "operator_refused": report.operator_refused,
        "satisfaction_before": report.satisfaction_before,
        "satisfaction_after": report.satisfaction_after,
    }


def report_from_dict(payload: dict) -> ScenarioReport:
    """Rebuild a report from its JSON mapping.

    Args:
        payload: The mapping written by :func:`report_to_dict`.

    How:
        Restores both checkpoint series and the overlay, leaving the metadata as
        stored.

    Returns:
        The rebuilt report.
    """
    return ScenarioReport(
        scenario=payload["scenario"],
        operator=payload["operator"],
        seed=payload["seed"],
        budget=payload["budget"],
        before=_pairs_to_series(payload["before"]),
        after=_pairs_to_series(payload["after"]),
        overlay=_overlay_from_dict(payload["overlay"]),
        reference_before=payload["reference_before"],
        reference_after=payload["reference_after"],
        operator_refused=bool(payload.get("operator_refused", False)),
        satisfaction_before=payload.get("satisfaction_before"),
        satisfaction_after=payload.get("satisfaction_after"),
    )


def spread_to_dict(spread: ScenarioSpread) -> dict:
    """Convert a multi-seed spread to a JSON-serialisable mapping.

    Args:
        spread: The multi-seed spread to render.

    How:
        Renders each seed summary as a plain mapping, so the file is a flat list
        of the four reported numbers per seed.

    Returns:
        The spread as a JSON-ready mapping.
    """
    return {
        "scenario": spread.scenario,
        "budget": spread.budget,
        "summaries": [
            {
                "seed": summary.seed,
                "before_80k": summary.before_80k,
                "after_80k": summary.after_80k,
                "before_final": summary.before_final,
                "after_final": summary.after_final,
            }
            for summary in spread.summaries
        ],
    }


def spread_from_dict(payload: dict) -> ScenarioSpread:
    """Rebuild a multi-seed spread from its JSON mapping.

    Args:
        payload: The mapping written by :func:`spread_to_dict`.

    How:
        Restores one :class:`SeedSummary` per entry, in the stored order.

    Returns:
        The rebuilt spread.
    """
    return ScenarioSpread(
        scenario=payload["scenario"],
        budget=payload["budget"],
        summaries=tuple(
            SeedSummary(
                seed=entry["seed"],
                before_80k=entry["before_80k"],
                after_80k=entry["after_80k"],
                before_final=entry["before_final"],
                after_final=entry["after_final"],
            )
            for entry in payload["summaries"]
        ),
    )


def _series_to_pairs(series: tuple[Checkpoint, ...]) -> list[list[float]]:
    """Render a checkpoint series as JSON pairs.

    Args:
        series: The checkpoint series to render.

    How:
        Maps each checkpoint to a ``[step, probability]`` pair, keeping the
        on-disk file readable and independent of the dataclass shape.

    Returns:
        The series as ``[[step, probability], ...]``.
    """
    return [[checkpoint.step, checkpoint.probability] for checkpoint in series]


def _pairs_to_series(pairs: list) -> tuple[Checkpoint, ...]:
    """Rebuild a checkpoint series from JSON pairs.

    Args:
        pairs: ``[[step, probability], ...]`` as loaded from disk.

    How:
        Coerces each pair back to an int step and a float probability, so a JSON
        number type never leaks past this boundary.

    Returns:
        The rebuilt checkpoint series.
    """
    return tuple(Checkpoint(int(step), float(probability)) for step, probability in pairs)


def _overlay_to_dict(overlay: Overlay) -> dict:
    """Render an overlay as a JSON-serialisable mapping.

    Args:
        overlay: The overlay to render.

    How:
        Converts point tuples to lists and leaves the optional normal, offset, and
        added edge as ``None`` when they are absent.

    Returns:
        The overlay as a JSON-ready mapping.
    """
    return {
        "kind": overlay.kind,
        "points": [list(point) for point in overlay.points],
        "counter_points": [list(point) for point in overlay.counter_points],
        "normal": list(overlay.normal) if overlay.normal is not None else None,
        "offset": overlay.offset,
        "added_edge": list(overlay.added_edge) if overlay.added_edge is not None else None,
        "avoid_points": [list(point) for point in overlay.avoid_points],
    }


def _overlay_from_dict(payload: dict) -> Overlay:
    """Rebuild an overlay from its JSON mapping.

    Args:
        payload: The mapping written by :func:`_overlay_to_dict`.

    How:
        Restores the point tuples as floats and rebuilds the optional normal and
        added edge when present. ``avoid_points`` is read with a default, so a
        record written before that field existed still loads.

    Returns:
        The rebuilt overlay.
    """
    normal = payload["normal"]
    added_edge = payload["added_edge"]
    return Overlay(
        kind=payload["kind"],
        points=tuple((float(x), float(y)) for x, y in payload["points"]),
        counter_points=tuple((float(x), float(y)) for x, y in payload["counter_points"]),
        normal=(float(normal[0]), float(normal[1])) if normal is not None else None,
        offset=payload["offset"],
        added_edge=(added_edge[0], added_edge[1]) if added_edge is not None else None,
        avoid_points=tuple(
            (float(x), float(y)) for x, y in payload.get("avoid_points", ())
        ),
    )



"""RQ2 data log: per-edge rollouts plus every label a fault-diagnosis model could learn.

Layout under ``<out>/rq2/``:

* ``<instance>.jsonl``: one JSON row per (phase, run seed, edge). See :func:`edge_row`.
* ``rollouts/<sha256>.npz``: the edge's full estimate (every rollout's states, ``reached``,
  ``safe``, ``final_safe``), encoded losslessly by
  :func:`faultline.learners.cache.encode_estimate` and content addressed, so identical
  estimates (a gate baseline and the AutoSpec first phase on the same seed) are stored once.

Phases:

* ``gold``: the gold graph's baseline on a gate seed (negative examples, no fault).
* ``gate``: the faulty graph's baseline on a gate seed (what the effectiveness gate saw).
* ``initial``: AutoSpec's first phase on an evaluation seed.
* ``decision``: a weak edge the loop examined, with the estimate it refined from and the
  score of every operator (``operators``). With ``score_all_operators`` the operators the
  loop skipped after committing are scored too (``by_loop: false``), so every decision row
  carries the full label vector.

Features are not computed here; :mod:`faultline.rq2.features` derives them offline from
the rollouts, so a new feature never needs a rerun.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from faultline.faults.ground_truth import EXPECTED_OPERATORS
from faultline.geometry import region_bounds
from faultline.golden.spec import edge_id
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners.base import EdgeEstimate
from faultline.learners.cache import encode_estimate
from faultline.spectrl.predicates import PLANAR_DIMS

SCHEMA_VERSION = 1


def _bounds(region) -> list[list[float]] | None:
    bounds = region_bounds(region, PLANAR_DIMS)
    return None if bounds is None else [np.asarray(bounds[0]).tolist(), np.asarray(bounds[1]).tolist()]


def _centre(region) -> np.ndarray | None:
    bounds = region_bounds(region, PLANAR_DIMS)
    return None if bounds is None else (np.asarray(bounds[0]) + np.asarray(bounds[1])) / 2.0


def _box_distance(point: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(low - point, 0.0), point - high)))


def edge_geometry(edge: Edge, layout, traps) -> dict[str, Any]:
    """Return static geometry of an edge: boxes, straight-line length, wall crossing, trap distance."""
    a, b = _centre(edge.source.region), _centre(edge.target.region)
    geometry: dict[str, Any] = {
        "source_bounds": _bounds(edge.source.region), "target_bounds": _bounds(edge.target.region),
        "safe": repr(edge.safe), "length": None, "crosses_wall": None, "trap_distance": None,
    }
    if a is None or b is None:
        return geometry
    geometry["length"] = float(np.linalg.norm(b - a))
    geometry["crosses_wall"] = bool(layout.segment_is_blocked(a, b))
    distances = []
    for trap in traps:
        bounds = region_bounds(trap, PLANAR_DIMS)
        if bounds is not None:
            low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
            distances += [_box_distance(a + t * (b - a), low, high) for t in np.linspace(0, 1, 65)]
    geometry["trap_distance"] = min(distances) if distances else None
    return geometry


class Rq2Log:
    """Writes the RQ2 rows of one instance (or one gold) and their rollouts.

    Args:
        directory: The ``rq2`` directory.
        name: File stem (the instance id, made filename safe).
        labels: Instance-level labels copied onto every row (fault kind, locus, ...).
        layout: The layout, for edge geometry.
        traps: Physical traps, for edge geometry.
        threshold: The refinement threshold ``p``.

    How:
        Rows are appended to ``<name>.jsonl`` as they are produced, so a killed run keeps
        what it logged; :meth:`reset` truncates the file when an instance is rerun.
    """

    def __init__(self, directory: Path, name: str, labels: dict[str, Any], layout, traps, threshold: float) -> None:
        self.directory = Path(directory)
        self.path = self.directory / f"{name}.jsonl"
        self.labels = labels
        self.layout = layout
        self.traps = tuple(traps)
        self.threshold = float(threshold)

    def reset(self) -> None:
        """Start the instance's file afresh."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def _rollouts_file(self, estimate: EdgeEstimate) -> str:
        data = encode_estimate(estimate)
        digest = hashlib.sha256(data).hexdigest()
        path = self.directory / "rollouts" / f"{digest}.npz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            os.replace(tmp, path)
        return f"rollouts/{digest}.npz"

    def edge_row(self, phase: str, seed: int, edge: Edge, estimate: EdgeEstimate | None,
                 operators: dict[str, dict] | None = None) -> dict[str, Any]:
        """Build and append one row.

        Row fields: ``schema``, the instance labels, ``phase``, ``seed``, ``edge``, ``source``,
        ``target``, ``learned``, ``p_hat``, ``is_weak`` (``p_hat < threshold``), ``threshold``,
        ``n_rollouts``, ``rollouts`` (relative path), ``touches_locus`` (the edge is the locus
        edge or touches the locus vertex), the edge geometry, and for decision rows
        ``operators`` (``{name: {"score": float | None, "committed": bool, "by_loop": bool}}``),
        ``committed_operator`` and ``operators_clearing`` (scores above the threshold).
        """
        locus_kind, locus_name = self.labels.get("locus", ["none", None])
        eid = edge_id(edge.source.name, edge.target.name)
        touches = (locus_kind == "edge" and locus_name == eid) or (
            locus_kind == "vertex" and locus_name in (edge.source.name, edge.target.name))
        row: dict[str, Any] = {
            "schema": SCHEMA_VERSION, **self.labels, "phase": phase, "seed": int(seed), "edge": eid,
            "source": edge.source.name, "target": edge.target.name, "learned": estimate is not None,
            "p_hat": None if estimate is None else float(estimate.success_probability),
            "threshold": self.threshold,
            "is_weak": None if estimate is None else bool(estimate.success_probability < self.threshold),
            "n_rollouts": 0 if estimate is None else len(estimate.rollouts),
            "rollouts": None if estimate is None else self._rollouts_file(estimate),
            "touches_locus": bool(touches),
            **edge_geometry(edge, self.layout, self.traps),
        }
        if operators is not None:
            row["operators"] = operators
            row["committed_operator"] = next((n for n, o in operators.items() if o["committed"]), None)
            row["operators_clearing"] = sorted(
                n for n, o in operators.items() if o["score"] is not None and o["score"] > self.threshold)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            out.write(json.dumps(row) + "\n")
        return row

    def baseline(self, phase: str, seed: int, graph: AbstractGraph, estimates) -> None:
        """Log a baseline (``{edge id: estimate | None}``) of ``graph``."""
        for edge in graph.edges:
            self.edge_row(phase, seed, edge, estimates.get(edge_id(edge.source.name, edge.target.name)))

    def observer(self, seed: int, score_all_operators: bool = True) -> "LoopRecorder":
        """Return a loop observer that logs one AutoSpec run on ``seed``."""
        return LoopRecorder(self, seed, score_all_operators)


class LoopRecorder:
    """The :class:`~faultline.refinement.loop.LoopObserver` behind :meth:`Rq2Log.observer`."""

    def __init__(self, log: Rq2Log, seed: int, score_all_operators: bool) -> None:
        self.log = log
        self.seed = seed
        self.score_all_operators = score_all_operators
        self._edge: Edge | None = None
        self._estimate: EdgeEstimate | None = None
        self._operators: dict[str, dict] = {}

    def initial(self, graph: AbstractGraph, estimates: dict[Edge, EdgeEstimate]) -> None:
        for edge in graph.edges:
            self.log.edge_row("initial", self.seed, edge, estimates.get(edge))

    def candidate(self, edge: Edge, estimate: EdgeEstimate, operator: str, score: float | None,
                  committed: bool, by_loop: bool) -> None:
        if self._edge is not None and (edge is not self._edge or operator in self._operators):
            self.flush()
        self._edge, self._estimate = edge, estimate
        self._operators[operator] = {"score": None if score is None else float(score),
                                     "committed": bool(committed), "by_loop": bool(by_loop)}

    def flush(self) -> None:
        """Write the pending decision row (call once after the run)."""
        if self._edge is not None:
            self.log.edge_row("decision", self.seed, self._edge, self._estimate, self._operators)
        self._edge, self._estimate, self._operators = None, None, {}


def instance_labels(instance) -> dict[str, Any]:
    """Instance-level labels for every row of an instance."""
    repair = instance.repair
    return {
        "instance": instance.id, "gold": instance.gold.id, "fault_kind": instance.kind.value,
        "level": instance.level, "instance_seed": instance.seed, "severity": repair.severity,
        "locus": [repair.locus.kind, repair.locus.name],
        "expected_operators": sorted(EXPECTED_OPERATORS[instance.kind]),
    }


def gold_labels(gold) -> dict[str, Any]:
    """Labels for the gold baseline rows (no fault)."""
    return {
        "instance": f"{gold.id}|gold", "gold": gold.id, "fault_kind": "none", "level": None,
        "instance_seed": None, "severity": None, "locus": ["none", None], "expected_operators": [],
    }

"""Offline features of one logged edge, computed from its rollouts (RQ2).

Each feature targets a failure signature the refinement operators react to:

* where failures end along the edge (start, middle, goal third): a dead start fails near
  the source, an inflated goal near the target;
* whether failures end inside the target box but miss the goal point (goal inflation);
* whether success depends on the start position (dead start: part of the source region
  cannot reach the target);
* violations versus timeouts versus being stuck at a wall (hazards and missing waypoints);
* how concentrated the failure end points are (entropy over a unit grid).
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from faultline.learners.cache import decode_estimate

STUCK_STEPS = 5
STUCK_DISTANCE = 0.1


def _projection(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Position of each point along the segment ``a -> b`` (0 at ``a``, 1 at ``b``), clipped."""
    direction = b - a
    length = float(direction @ direction)
    if length == 0.0:
        return np.zeros(len(points))
    return np.clip((points - a) @ direction / length, 0.0, 1.0)


def _inside(points: np.ndarray, bounds) -> np.ndarray:
    low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
    return np.all((points >= low) & (points <= high), axis=1)


def _entropy(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    counts = Counter(map(tuple, np.floor(points).astype(int)))
    p = np.array(list(counts.values()), dtype=float) / len(points)
    return float(-(p * np.log(p)).sum())


def edge_features(row: dict[str, Any], rq2_dir: Path) -> dict[str, Any]:
    """Return the features of one logged row (empty for an unlearned edge).

    Args:
        row: A row of ``<instance>.jsonl``.
        rq2_dir: The ``rq2`` directory holding ``rollouts/``.
    """
    if not row.get("rollouts"):
        return {}
    estimate = decode_estimate(Path(rq2_dir) / row["rollouts"])
    rollouts = estimate.rollouts
    n = len(rollouts)
    starts = np.array([r.states[0][:2] for r in rollouts])
    ends = np.array([r.states[-1][:2] for r in rollouts])
    success = np.array([r.success for r in rollouts])
    violated = np.array([not r.safe for r in rollouts])
    timeout = ~success & ~violated
    stuck = np.array([
        len(r.states) > STUCK_STEPS
        and np.linalg.norm(r.states[-1][:2] - r.states[-1 - STUCK_STEPS][:2]) < STUCK_DISTANCE
        for r in rollouts
    ]) & ~success
    failed = ~success
    sb, tb = row["source_bounds"], row["target_bounds"]
    a = (np.asarray(sb[0]) + np.asarray(sb[1])) / 2 if sb else starts.mean(axis=0)
    b = (np.asarray(tb[0]) + np.asarray(tb[1])) / 2 if tb else ends.mean(axis=0)
    progress = _projection(ends[failed], a, b) if failed.any() else np.zeros(0)
    features: dict[str, Any] = {
        "p_hat": float(success.mean()),
        "frac_violation": float(violated.mean()),
        "frac_timeout": float(timeout.mean()),
        "frac_stuck": float(stuck.mean()),
        "mean_length": float(np.mean([len(r.states) for r in rollouts])),
        "fail_progress_mean": float(progress.mean()) if len(progress) else math.nan,
        "fail_progress_std": float(progress.std()) if len(progress) else math.nan,
        "fail_near_start": float((progress < 1 / 3).mean()) if len(progress) else math.nan,
        "fail_mid": float(((progress >= 1 / 3) & (progress <= 2 / 3)).mean()) if len(progress) else math.nan,
        "fail_near_goal": float((progress > 2 / 3).mean()) if len(progress) else math.nan,
        "fail_in_target_box": float(_inside(ends[failed], tb).mean()) if failed.any() and tb else math.nan,
        "fail_end_spread": float(ends[failed].std(axis=0).sum()) if failed.sum() > 1 else math.nan,
        "fail_end_entropy": _entropy(ends[failed]),
        "n_witnesses": int(success.sum()),
        "n_rollouts": n,
    }
    # Start dependence: success rate per quadrant of the source box (max minus min).
    if sb:
        mid = (np.asarray(sb[0]) + np.asarray(sb[1])) / 2
        quadrant = (starts[:, 0] > mid[0]).astype(int) * 2 + (starts[:, 1] > mid[1]).astype(int)
        rates = [success[quadrant == q].mean() for q in range(4) if (quadrant == q).sum() >= 10]
        features["start_quadrant_gap"] = float(max(rates) - min(rates)) if len(rates) > 1 else math.nan
    for name in ("length", "crosses_wall", "trap_distance"):
        value = row.get(name)
        features[name] = math.nan if value is None else float(value)
    return features

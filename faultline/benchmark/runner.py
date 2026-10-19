"""Run one benchmark instance: gates, Algorithm 1, baselines, metrics, JSON record."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from faultline.baselines import graph_baselines, random_operator
from faultline.benchmark.instances import Instance, evaluation_seeds, instance_rng
from faultline.faults.gates import Baseline, GateResult, Stage, learn_baseline, screen_with_learning
from faultline.faults.ground_truth import FaultKind
from faultline.golden.env import make_env
from faultline.golden.raster import CellGrid
from faultline.golden.spec import edge_id
from faultline.graph import AbstractGraph
from faultline.learners import DiRL, PPOEdgeLearner, TabularLearner
from faultline.learners.cache import CachedLearner, EstimateStore
from faultline.metrics.evaluate import evaluate_repair
from faultline.protocol import Protocol
from faultline.refinement import AutoSpecConfig, run_autospec
from faultline.rq2.log import Rq2Log, gold_labels, instance_labels


def git_hash() -> str:
    """Return the current commit hash, with ``-dirty`` when the tree has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class Settings:
    """Everything a run needs beyond the instance, read from the protocol."""

    protocol: Protocol
    learner: str = "ppo"
    method: str = "dirl"
    gate_seeds: tuple[int, ...] = (0, 1, 2)
    budget: int | None = None
    n_rollouts: int | None = None
    audit_samples: int = 200
    cache_dir: str | None = None
    score_all_operators: bool = True

    @property
    def config(self) -> AutoSpecConfig:
        """The loop configuration (seed is set per run)."""
        a = self.protocol["autospec"]
        return AutoSpecConfig(
            threshold=a["threshold"], budget=self.budget or a["budget"],
            n_rollouts=self.n_rollouts or a["n_rollouts"], avoid_tail_length=a["avoid_tail_length"],
        )

    def edge_learner(self, seed: int):
        """Return a fresh edge learner, wrapped in the exact estimate cache when one is set."""
        if self.learner == "ppo":
            learner = PPOEdgeLearner(device="cpu", seed=seed)
        elif self.learner == "tabular":
            learner = TabularLearner()
        else:
            raise ValueError("learner must be 'ppo' or 'tabular'")
        if self.cache_dir is None:
            return learner
        return CachedLearner(learner, EstimateStore(self.cache_dir))

    def graph_learner(self):
        """Return a fresh graph learner, or ``None`` for the base method."""
        if self.method == "base":
            return None
        if self.method == "dirl":
            return DiRL()
        raise ValueError("method must be 'base' or 'dirl'")


def baseline_for(settings: Settings, gold_graph: AbstractGraph, env: Any, seed: int) -> Baseline:
    """Learn the unrefined baseline of a graph on one seed."""
    cfg = settings.config
    return learn_baseline(gold_graph, env, settings.edge_learner(seed), settings.graph_learner(),
                          budget=cfg.budget, n_rollouts=cfg.n_rollouts, seed=seed)


def gold_baselines(settings: Settings, gold, rq2_dir: Path | None = None) -> dict[int, Baseline]:
    """Learn gold's baseline on every gate seed, logging the RQ2 ``gold`` rows when asked."""
    env = make_env(gold)
    out = {s: baseline_for(settings, gold.graph, env, s) for s in settings.gate_seeds}
    if rq2_dir is not None:
        log = Rq2Log(rq2_dir, _safe_name(gold.id + "|gold"), gold_labels(gold), gold.layout, gold.traps,
                     settings.config.threshold)
        log.reset()
        for seed, baseline in out.items():
            log.baseline("gold", seed, gold.graph, baseline)
    return out


def _safe_name(identifier: str) -> str:
    return identifier.replace("|", "__").replace("/", "_")


def _graph_dict(graph: AbstractGraph) -> dict:
    return {
        "vertices": {v.name: repr(v.region) for v in graph.vertices},
        "edges": [{"source": e.source.name, "target": e.target.name, "safe": repr(e.safe)} for e in graph.edges],
    }


def _run_dict(run) -> dict:
    return {
        "best_path_success": run.best_path_success,
        "initial": {edge_id(e.source, e.target): e.probability for e in run.initial_edge_success},
        "final": {edge_id(e.source, e.target): e.probability for e in run.edge_success},
        "steps": [
            {"edge": edge_id(s.source, s.target), "tried": list(s.operators_tried), "operator": s.operator,
             "before": s.before, "after": s.after}
            for s in run.steps
        ],
    }


def run_instance(
    instance: Instance,
    settings: Settings,
    gold_base: Mapping[int, Baseline],
    *,
    seeds: tuple[int, ...] | None = None,
    skip_gates: bool = False,
    gates_only: bool = False,
    log: Callable[[str], None] = lambda _msg: None,
    rq2_dir: Path | None = None,
) -> dict[str, Any]:
    """Screen one instance, run AutoSpec and the baselines on it, and score everything.

    Args:
        instance: A valid instance.
        settings: Run settings.
        gold_base: Gold baselines per gate seed (from :func:`gold_baselines`).
        seeds: Evaluation seeds; the protocol's by default.
        skip_gates: Run the loop even when a gate rejects (for ablations); the stage is
            still recorded.
        gates_only: Stop after the gates (the funnel checkpoint before any evaluation run).
        log: Progress sink.
        rq2_dir: When given, the RQ2 rows of this instance (gate baselines, AutoSpec first
            phase and decisions) are written there (:mod:`faultline.rq2.log`).

    How:
        Gates run first on the unrefined faulty graph. A rejected instance is recorded with
        its reason and, unless ``skip_gates``, not run. An accepted or unobservable one
        runs the real loop and the random-operator baseline on every evaluation seed and
        is scored with :func:`evaluate_repair`; the graph baselines (unrefined, oracle,
        trivializer) are scored once. Nothing is thrown away: the record carries the
        gate reports, so a reader can recompute any stratum.

    Returns:
        A JSON-serialisable record.
    """
    protocol = settings.protocol
    cfg = settings.config
    gates = protocol["gates"]
    repair, gold = instance.repair, instance.gold
    grid = CellGrid(gold.layout, protocol["metrics"]["grid_resolution"])
    record: dict[str, Any] = {
        "id": instance.id, "gold": gold.id, "kind": instance.kind.value, "level": instance.level,
        "seed": instance.seed, "severity": repair.severity, "factors": {k: _jsonable(v) for k, v in repair.factors.items()},
        "locus": [repair.locus.kind, repair.locus.name], "weak_edges": sorted(repair.weak_edges),
        "protocol_hash": protocol.hash, "git": git_hash(),
        "learner": settings.learner, "method": settings.method,
        "faulty_graph": _graph_dict(repair.faulty),
    }
    env = make_env(gold)
    rq2 = None
    if rq2_dir is not None:
        rq2 = Rq2Log(rq2_dir, _safe_name(instance.id), instance_labels(instance), gold.layout, gold.traps,
                     cfg.threshold)
        rq2.reset()

    def learn_gate(graph: AbstractGraph, seed: int) -> Baseline:
        baseline = baseline_for(settings, graph, env, seed)
        if rq2 is not None:
            rq2.baseline("gate", seed, graph, baseline)
        return baseline

    started = time.perf_counter()
    if instance.kind is FaultKind.NULL:
        gate = GateResult(Stage.ACCEPTED)      # the null is gold itself: nothing to screen
    else:
        gate = screen_with_learning(
            repair, gold_base, settings.gate_seeds, learn_gate,
            threshold=cfg.threshold, min_gap=gates["effectiveness_min_gap"],
            seed_fraction=gates["effectiveness_seed_fraction"], min_witnesses=gates["observability_min_witnesses"],
        )
    record["gate"] = {
        "stage": gate.stage.value, "reason": gate.reason,
        "effectiveness": None if gate.effectiveness is None else {
            "fraction": gate.effectiveness.fraction, "gaps": dict(gate.effectiveness.gaps),
            "triggers": dict(gate.effectiveness.triggers)},
        "observability": None if gate.observability is None else {
            "witnesses": dict(gate.observability.witnesses), "kind": gate.observability.kind_of_witness},
        "seconds": time.perf_counter() - started,
    }
    log(f"{instance.id}: gate {gate.stage.value}")
    # Hazard-unlabeled cannot bite under an absorbing trap that is identical for gold and
    # faulty (PPO is penalised by the physical trap either way), so the effectiveness gate
    # is expected to reject it. Those instances still run and form their own stratum.
    by_construction = instance.kind is FaultKind.HAZARD_UNLABELED
    record["stratum"] = (
        "ineffective_by_construction" if by_construction else gate.stage.value
    )
    if gates_only or (gate.stage is Stage.INEFFECTIVE and not (skip_gates or by_construction)):
        return record

    seeds = tuple(seeds) if seeds is not None else evaluation_seeds(instance, protocol["seeds"]["evaluation"])
    repair_full = _with_ceiling(repair, grid)
    base = graph_baselines(repair, grid)
    record["baselines"] = {
        name: evaluate_repair(repair_full, graph, grid, instance_rng(instance.id + "|" + name), audit_samples=settings.audit_samples)
        for name, graph in (("unrefined", base.unrefined), ("oracle", base.oracle), ("trivializer", base.trivializer))
    }
    record["runs"] = []
    for seed in seeds:
        t0 = time.perf_counter()
        observer = None if rq2 is None else rq2.observer(seed, settings.score_all_operators)
        run = run_autospec(repair.faulty, make_env(gold), settings.edge_learner(seed), replace(cfg, seed=seed),
                           settings.graph_learner(), observer=observer)
        if observer is not None:
            observer.flush()
        rand = random_operator(repair.faulty, make_env(gold), settings.edge_learner(seed), replace(cfg, seed=seed),
                               instance_rng(f"{instance.id}|order|{seed}"), settings.graph_learner())
        record["runs"].append({
            "seed": seed, "seconds": time.perf_counter() - t0,
            "autospec": _run_dict(run),
            "autospec_eval": evaluate_repair(repair_full, run.graph, grid, instance_rng(f"{instance.id}|eval|{seed}"), audit_samples=settings.audit_samples),
            "random_operator": _run_dict(rand),
            "random_operator_eval": evaluate_repair(repair_full, rand.graph, grid, instance_rng(f"{instance.id}|reval|{seed}"), audit_samples=settings.audit_samples),
        })
        log(f"{instance.id}: seed {seed} done in {record['runs'][-1]['seconds']:.0f}s")
    return record


def _with_ceiling(repair, grid):
    from faultline.metrics.ceiling import with_ceiling
    return with_ceiling(repair, grid)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def write_record(record: Mapping[str, Any], directory: Path) -> Path:
    """Write a record as ``<id>.json`` (identifier made filename-safe)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_safe_name(record["id"]) + ".json")
    path.write_text(json.dumps(record, indent=2, default=repr))
    return path

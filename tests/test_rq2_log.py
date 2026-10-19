"""The RQ2 log: rows and rollouts are written, labels are complete, and logging changes nothing."""

from __future__ import annotations

import json
import math

from faultline.benchmark.instances import Instance, instance_rng
from faultline.benchmark.runner import Settings, gold_baselines, run_instance
from faultline.envs.rooms.layout import RoomsLayout
from faultline.faults.ground_truth import FaultKind, MutationParams
from faultline.faults.mutations import mutate
from faultline.golden.builder import build_gold, chain_graph
from faultline.golden.env import make_env
from faultline.learners import DiRL, TabularLearner
from faultline.protocol import load_protocol
from faultline.refinement import AutoSpecConfig, run_autospec
from faultline.rq2 import Rq2Log, edge_features, instance_labels
from faultline.spectrl.predicates import Box

from test_edge_cache import _summary


def _instance():
    gold = build_gold("tiny", RoomsLayout(doors={}),
                      chain_graph(Box((1, 1), (3, 3)), (Box((4, 1), (6, 3)),), Box((4, 5), (8, 7))),
                      resolution=24, max_steps=20)
    repair = mutate(FaultKind.GOAL_INFLATION, gold, MutationParams(level="high"), instance_rng("cache"))
    return Instance("tiny|goal_inflation|high|0", gold, FaultKind.GOAL_INFLATION, "high", 0, repair)


def test_observer_scores_every_operator_and_changes_nothing(tmp_path):
    inst = _instance()
    config = AutoSpecConfig(threshold=0.8, budget=0, n_rollouts=40, seed=3)
    plain = run_autospec(inst.repair.faulty, make_env(inst.gold), TabularLearner(), config, DiRL())
    log = Rq2Log(tmp_path, "x", instance_labels(inst), inst.gold.layout, inst.gold.traps, 0.8)
    observer = log.observer(3)
    logged = run_autospec(inst.repair.faulty, make_env(inst.gold), TabularLearner(), config, DiRL(), observer=observer)
    observer.flush()
    assert _summary(plain) == _summary(logged)
    rows = [json.loads(line) for line in (tmp_path / "x.jsonl").read_text().splitlines()]
    decisions = [r for r in rows if r["phase"] == "decision"]
    assert {r["phase"] for r in rows} == {"initial", "decision"}
    assert decisions and all(set(r["operators"]) == {"seq", "add", "past", "or"} for r in decisions)
    committed = [r for r in decisions if r["committed_operator"]]
    assert committed and any(not o["by_loop"] for r in committed for o in r["operators"].values())
    assert all(r["fault_kind"] == "goal_inflation" and r["expected_operators"] == ["seq"] for r in rows)


def test_runner_writes_gold_gate_and_loop_rows_with_features(tmp_path):
    inst = _instance()
    settings = Settings(load_protocol(), learner="tabular", method="dirl", gate_seeds=(0,), budget=0,
                        n_rollouts=40, audit_samples=40, cache_dir=str(tmp_path / "cache"))
    rq2 = tmp_path / "rq2"
    base = gold_baselines(settings, inst.gold, rq2_dir=rq2)
    run_instance(inst, settings, base, seeds=(0,), skip_gates=True, rq2_dir=rq2)
    rows = [json.loads(line) for path in sorted(rq2.glob("*.jsonl")) for line in path.read_text().splitlines()]
    assert {"gold", "gate", "initial", "decision"} <= {r["phase"] for r in rows}
    assert any(r["touches_locus"] for r in rows)
    learned = [r for r in rows if r["learned"]]
    features = edge_features(learned[0], rq2)
    assert math.isclose(features["p_hat"], learned[0]["p_hat"]) and features["n_rollouts"] == 40


def test_analyze_writes_tables_and_figures(tmp_path):
    from faultline.benchmark.analyze import analyze
    from faultline.benchmark.runner import write_record

    inst = _instance()
    settings = Settings(load_protocol(), learner="tabular", method="dirl", gate_seeds=(0,), budget=0,
                        n_rollouts=40, audit_samples=40)
    record = run_instance(inst, settings, gold_baselines(settings, inst.gold), seeds=(0, 1), skip_gates=True)
    write_record(record, tmp_path)
    analyze(tmp_path, threshold=0.8, resamples=200)
    summary = (tmp_path / "analysis" / "summary.md").read_text()
    assert "Goal inflation" in summary and (tmp_path / "analysis" / "runs.csv").exists()

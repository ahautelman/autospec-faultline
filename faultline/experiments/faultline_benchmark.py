"""End-to-end FaultLine benchmark runner.

Connects injected ``FaultCase`` objects to AutoSpec, records the trajectory
evidence available on the injected fault, evaluates the repaired specification
against the known correct graph, and writes detailed JSON plus a flat CSV.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faultline.experiments.faultline_metrics import evaluate_recovery
from faultline.faults import FaultBenchmarkCase, FaultCase, FaultType
from faultline.graph import AbstractGraph, best_path_product
from faultline.learners import DiRL, LSTS, PPOEdgeLearner, TabularLearner
from faultline.refinement.loop import AutoSpecConfig, AutoSpecRun, run_autospec


@dataclass(frozen=True)
class FaultLineSettings:
    """Configuration shared by one FaultLine experiment batch."""

    learner: str = "ppo"          # "ppo" or "tabular"
    method: str = "dirl"          # "base", "dirl", or "lsts"
    threshold: float = 0.80
    budget: int = 40_000
    n_rollouts: int = 50
    avoid_tail_length: int = 3
    seeds: tuple[int, ...] = (0,)
    quality_threshold: float = 0.80
    grid_resolution: int = 121
    output_directory: str = "results/faultline"

    lsts_epsilon: float = 0.30
    lsts_alpha: float = 0.20
    lsts_tolerance: float = 0.10
    lsts_chunk_size: int = 5_000
    lsts_max_interactions: int | None = None

    max_recorded_visit_states: int = 200_000


@dataclass(frozen=True)
class EstimateRecord:
    """One edge-policy evaluation captured by ``CountingLearner``."""

    source: str
    target: str
    probability: float
    reached_rollouts: int
    successes: int
    failures: int
    unsafe_rollouts: int
    unsafe_reached_rollouts: int
    visit_states: tuple[np.ndarray, ...]


class CountingLearner:
    """Transparent learner wrapper that records cost and rollout evidence."""

    def __init__(
        self,
        learner: Any,
        *,
        max_recorded_visit_states: int = 200_000,
    ) -> None:
        self.learner = learner
        self.train_edge_calls = 0
        self.continue_edge_calls = 0
        self.estimate_calls = 0
        self.requested_training_steps = 0
        self.evaluation_rollouts = 0
        self.max_recorded_visit_states = int(max_recorded_visit_states)
        self.estimate_records: list[EstimateRecord] = []

    def train_edge(self, env, edge, budget, *args, **kwargs):
        self.train_edge_calls += 1
        if budget is not None:
            self.requested_training_steps += int(budget)
        return self.learner.train_edge(env, edge, budget, *args, **kwargs)

    def continue_edge(self, env, edge, policy, budget, *args, **kwargs):
        if not hasattr(self.learner, "continue_edge"):
            raise AttributeError("Wrapped learner does not implement continue_edge")
        self.continue_edge_calls += 1
        if budget is not None:
            self.requested_training_steps += int(budget)
        return self.learner.continue_edge(env, edge, policy, budget, *args, **kwargs)

    def estimate_reach(self, env, edge, policy, n_rollouts, seed):
        self.estimate_calls += 1
        self.evaluation_rollouts += int(n_rollouts)
        estimate = self.learner.estimate_reach(env, edge, policy, n_rollouts, seed)

        states: list[np.ndarray] = []
        for rollout in estimate.rollouts:
            for state in rollout.states:
                if len(states) >= self.max_recorded_visit_states:
                    break
                states.append(np.asarray(state, dtype=float).copy())
            if len(states) >= self.max_recorded_visit_states:
                break

        reached = sum(1 for rollout in estimate.rollouts if rollout.reached)
        successes = sum(1 for rollout in estimate.rollouts if rollout.success)
        unsafe = sum(1 for rollout in estimate.rollouts if not rollout.safe)
        unsafe_reached = sum(
            1
            for rollout in estimate.rollouts
            if rollout.reached and not rollout.safe
        )

        self.estimate_records.append(
            EstimateRecord(
                source=edge.source.name,
                target=edge.target.name,
                probability=float(estimate.success_probability),
                reached_rollouts=reached,
                successes=successes,
                failures=len(estimate.rollouts) - successes,
                unsafe_rollouts=unsafe,
                unsafe_reached_rollouts=unsafe_reached,
                visit_states=tuple(states),
            )
        )
        return estimate

    def __getattr__(self, name: str):
        return getattr(self.learner, name)


def run_fault_case(
    benchmark: FaultBenchmarkCase,
    settings: FaultLineSettings,
    seed: int,
) -> dict[str, Any]:
    """Run one injected fault through AutoSpec and evaluate the result."""

    case = benchmark.fault
    trigger_edge = _fault_trigger_edge(case)

    print("\n" + "=" * 88)
    print(
        f"FAULTLINE | case={case.name} | fault={case.fault_type.value} | "
        f"method={settings.method} | learner={settings.learner} | seed={seed}"
    )
    print("=" * 88)
    print(f"Expected operator : {case.expected_operator}")
    print(f"Fault trigger edge: {trigger_edge}")
    print(f"Description       : {case.description}")
    print(
        f"Graphs            : correct V={len(case.correct_graph.vertices)} "
        f"E={len(case.correct_graph.edges)} | faulty V={len(case.faulty_graph.vertices)} "
        f"E={len(case.faulty_graph.edges)}"
    )

    env = benchmark.make_env()
    counted = CountingLearner(
        _make_edge_learner(settings, seed),
        max_recorded_visit_states=settings.max_recorded_visit_states,
    )
    graph_learner = _make_graph_learner(settings)
    config = AutoSpecConfig(
        threshold=settings.threshold,
        budget=settings.budget,
        n_rollouts=settings.n_rollouts,
        avoid_tail_length=settings.avoid_tail_length,
        seed=seed,
    )

    print(
        f"Config            : threshold={settings.threshold:.2f}, "
        f"budget={settings.budget}, rollouts={settings.n_rollouts}, "
        f"avoid_tail={settings.avoid_tail_length}, max_steps={env.max_steps}"
    )
    print(
        "Graph learner     : "
        f"{type(graph_learner).__name__ if graph_learner is not None else 'base/direct'}"
    )
    print("Running AutoSpec...")

    started = time.perf_counter()
    try:
        run = run_autospec(
            graph=case.faulty_graph,
            env=env,
            learner=counted,
            config=config,
            graph_learner=graph_learner,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"RUN FAILED        : {type(exc).__name__}: {exc}")
        return _failed_record(case, settings, seed, elapsed, counted, exc)
    elapsed = time.perf_counter() - started

    baseline_record = _baseline_evidence(case, counted.estimate_records)
    evidence = _evidence_summary(case, baseline_record)
    baseline_states = baseline_record.visit_states if baseline_record is not None else ()

    initial_best_path = _initial_best_path(case.faulty_graph, run)
    committed = tuple(step.operator for step in run.steps if step.operator is not None)
    first_committed_operator = committed[0] if committed else None
    trigger_step = _fault_trigger_step(case, run)
    fault_trigger_operator = trigger_step.operator if trigger_step is not None else None
    expected_operator_used = any(
        step.operator == case.expected_operator
        for step in run.steps
    )

    # "actual_operator" is deliberately fault-local.  A multi-edge graph may
    # undergo an unrelated refinement before the injected bottleneck is handled.
    actual_operator = fault_trigger_operator
    operator_correct = actual_operator == case.expected_operator

    recovery = evaluate_recovery(
        case,
        run.graph,
        baseline_states,
        world_size=float(env.layout.world_size),
        grid_resolution=settings.grid_resolution,
        quality_threshold=settings.quality_threshold,
    )

    outcome_category = _classify_outcome(
        refinement_possible=bool(evidence["refinement_possible"]),
        operator_correct=operator_correct,
        recovered=bool(recovery["recovered"]),
        actual_operator=actual_operator,
    )

    print("\n--- Baseline evidence on the injected fault ---")
    if baseline_record is None:
        print("Affected edge      : not learned/evaluated by the graph learner")
    else:
        print(f"Affected edge      : {baseline_record.source}->{baseline_record.target}")
        print(f"Success probability: {baseline_record.probability:.3f}")
        print(
            "Rollouts           : "
            f"reached={baseline_record.reached_rollouts}, "
            f"success={baseline_record.successes}, "
            f"failure={baseline_record.failures}, "
            f"unsafe={baseline_record.unsafe_rollouts}, "
            f"unsafe+reached={baseline_record.unsafe_reached_rollouts}"
        )
    print(f"Refinement data ok : {evidence['refinement_possible']}")
    print(f"Evidence reason    : {evidence['reason']}")

    print("\n--- AutoSpec decisions ---")
    if not run.steps:
        print("No learned weak edge was examined for refinement.")
    for index, step in enumerate(run.steps, start=1):
        marker = " <-- injected fault" if (
            trigger_edge is not None
            and (step.source, step.target) == trigger_edge
        ) else ""
        print(
            f"[{index}] {step.source}->{step.target} | before={step.before:.3f} | "
            f"tried={list(step.operators_tried)} | committed={step.operator} | "
            f"after={step.after:.3f}{marker}"
        )

    print("\n--- Recovery ---")
    print(f"Expected operator : {case.expected_operator}")
    print(f"First committed   : {first_committed_operator}")
    print(f"Trigger operator  : {fault_trigger_operator}")
    print(f"Expected used     : {expected_operator_used}")
    print(f"Operator correct  : {operator_correct}")
    print(f"Initial best path : {initial_best_path:.3f}")
    print(f"Final best path   : {run.best_path_success:.3f}")
    print(f"Metric type       : {recovery['metric_type']}")
    _print_optional("Baseline IoU      ", recovery.get("baseline_iou"))
    _print_optional("Repaired IoU      ", recovery.get("repaired_iou"))
    _print_optional("Recovery gain     ", recovery.get("normalized_recovery_gain"))
    _print_optional("Repaired precision", recovery.get("repaired_precision"))
    _print_optional("Repaired recall   ", recovery.get("repaired_recall"))
    _print_optional("Baseline wIoU     ", recovery.get("baseline_weighted_iou"))
    _print_optional("Repaired wIoU     ", recovery.get("repaired_weighted_iou"))
    _print_optional("Graph edit dist.  ", recovery.get("normalized_graph_edit_distance"))
    _print_optional("Graph similarity  ", recovery.get("graph_similarity"))
    print(f"Recovered         : {recovery['recovered']}")
    print(f"Recovery reason   : {recovery['recovery_reason']}")
    print(f"Outcome category  : {outcome_category}")

    print("\n--- Cost ---")
    print(f"train_edge calls  : {counted.train_edge_calls}")
    print(f"continue calls    : {counted.continue_edge_calls}")
    print(f"estimate calls    : {counted.estimate_calls}")
    print(f"requested steps   : {counted.requested_training_steps}")
    print(f"evaluation rollouts: {counted.evaluation_rollouts}")
    print(f"wall-clock        : {elapsed:.2f}s")

    return {
        "status": "ok",
        "case": case.name,
        "fault_type": case.fault_type.value,
        "severity": case.severity,
        "seed": seed,
        "learner": settings.learner,
        "method": settings.method,
        "expected_operator": case.expected_operator,
        "actual_operator": actual_operator,
        "first_committed_operator": first_committed_operator,
        "fault_trigger_operator": fault_trigger_operator,
        "fault_trigger_edge": list(trigger_edge) if trigger_edge is not None else None,
        "expected_operator_used": expected_operator_used,
        "operator_correct": operator_correct,
        "outcome_category": outcome_category,
        "refinement_possible": evidence["refinement_possible"],
        "evidence_reason": evidence["reason"],
        "baseline_reached_trajectories": evidence["reached_rollouts"],
        "baseline_successful_trajectories": evidence["successes"],
        "baseline_failed_trajectories": evidence["failures"],
        "baseline_unsafe_trajectories": evidence["unsafe_rollouts"],
        "baseline_unsafe_reached_trajectories": evidence["unsafe_reached_rollouts"],
        "baseline_affected_edge_probability": evidence["probability"],
        "committed_operators": list(committed),
        "operators_examined": [
            {
                "source": step.source,
                "target": step.target,
                "operators_tried": list(step.operators_tried),
                "committed_operator": step.operator,
                "success_before": step.before,
                "success_after": step.after,
                "is_fault_trigger": bool(
                    trigger_edge is not None
                    and (step.source, step.target) == trigger_edge
                ),
            }
            for step in run.steps
        ],
        "initial_best_path_success": initial_best_path,
        "final_best_path_success": run.best_path_success,
        "best_path_improvement": run.best_path_success - initial_best_path,
        "initial_edge_success": [
            {
                "source": edge.source,
                "target": edge.target,
                "probability": edge.probability,
            }
            for edge in run.initial_edge_success
        ],
        "final_edge_success": [
            {
                "source": edge.source,
                "target": edge.target,
                "probability": edge.probability,
            }
            for edge in run.edge_success
        ],
        "recovery": recovery,
        "cost": {
            "wall_clock_seconds": elapsed,
            "train_edge_calls": counted.train_edge_calls,
            "continue_edge_calls": counted.continue_edge_calls,
            "estimate_calls": counted.estimate_calls,
            "requested_training_steps": counted.requested_training_steps,
            "evaluation_rollouts": counted.evaluation_rollouts,
        },
        "config": {
            "threshold": settings.threshold,
            "budget": settings.budget,
            "n_rollouts": settings.n_rollouts,
            "avoid_tail_length": settings.avoid_tail_length,
            "quality_threshold": settings.quality_threshold,
            "grid_resolution": settings.grid_resolution,
        },
        "fault": {
            "description": case.description,
            "affected_vertex": case.affected_vertex,
            "affected_edge": list(case.affected_edge) if case.affected_edge is not None else None,
        },
        "graphs": {
            "correct": graph_to_dict(case.correct_graph),
            "faulty": graph_to_dict(case.faulty_graph),
            "repaired": graph_to_dict(run.graph),
        },
    }


def run_faultline_experiment(
    cases: tuple[FaultBenchmarkCase, ...],
    settings: FaultLineSettings,
) -> list[dict[str, Any]]:
    """Run all requested fault cases/seeds and persist JSON plus CSV outputs."""

    output_dir = Path(settings.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 88)
    print("FAULTLINE EXPERIMENT")
    print("#" * 88)
    print(f"Cases             : {len(cases)}")
    print(f"Seeds             : {settings.seeds}")
    print(f"Learner / method  : {settings.learner} / {settings.method}")
    print(f"Output directory  : {output_dir.resolve()}")

    records: list[dict[str, Any]] = []
    total = len(cases) * len(settings.seeds)
    index = 0

    for benchmark in cases:
        for seed in settings.seeds:
            index += 1
            print(f"\nStarting run {index}/{total}")
            record = run_fault_case(benchmark, settings, seed)
            records.append(record)
            _write_json(
                output_dir / f"{record['case']}__{settings.method}__seed{seed}.json",
                record,
            )

    _write_json(output_dir / "all_runs.json", records)
    _write_summary_csv(output_dir / "summary.csv", records)
    _print_batch_summary(records, output_dir)
    return records


def _fault_trigger_edge(case: FaultCase) -> tuple[str, str] | None:
    """Return the weak edge on which the injected fault should be diagnosed."""

    if case.fault_type in {FaultType.OVERWIDE_SAFETY, FaultType.MISSING_WAYPOINT}:
        return case.affected_edge

    if case.fault_type is FaultType.MISSING_ALTERNATIVE:
        if case.affected_edge is None or case.faulty_graph.target is None:
            return None
        return (case.affected_edge[0], case.faulty_graph.target.name)

    if case.fault_type is FaultType.GOAL_TRAP:
        vertex = _vertex_named(case.faulty_graph, case.affected_vertex)
        if vertex is None:
            return None
        incoming = case.faulty_graph.predecessors(vertex)
        if len(incoming) == 1:
            edge = incoming[0]
            return (edge.source.name, edge.target.name)
        return None

    if case.fault_type is FaultType.UNREACHABLE_START:
        vertex = _vertex_named(case.faulty_graph, case.affected_vertex)
        if vertex is None:
            return None
        outgoing = case.faulty_graph.successors(vertex)
        if len(outgoing) == 1:
            edge = outgoing[0]
            return (edge.source.name, edge.target.name)
        return None

    return case.affected_edge


def _fault_trigger_step(case: FaultCase, run: AutoSpecRun):
    trigger = _fault_trigger_edge(case)
    if trigger is None:
        return None
    matching = [
        step
        for step in run.steps
        if (step.source, step.target) == trigger
    ]
    return matching[-1] if matching else None


def _baseline_evidence(
    case: FaultCase,
    records: list[EstimateRecord],
) -> EstimateRecord | None:
    """Return the first evaluation of the fault's trigger edge."""

    trigger = _fault_trigger_edge(case)
    if trigger is None:
        return None
    for record in records:
        if (record.source, record.target) == trigger:
            return record
    return None


def _evidence_summary(
    case: FaultCase,
    record: EstimateRecord | None,
) -> dict[str, Any]:
    if record is None:
        return {
            "refinement_possible": False,
            "reason": "fault trigger edge was not learned/evaluated",
            "reached_rollouts": 0,
            "successes": 0,
            "failures": 0,
            "unsafe_rollouts": 0,
            "unsafe_reached_rollouts": 0,
            "probability": None,
        }

    if case.fault_type is FaultType.GOAL_TRAP:
        possible = record.successes > 0 and record.failures > 0
        reason = (
            "both reachable and unreachable target samples available"
            if possible
            else "ReachRefine benchmark needs both successful and failed target samples"
        )

    elif case.fault_type is FaultType.OVERWIDE_SAFETY:
        possible = record.unsafe_rollouts > 0
        reason = (
            "unsafe trajectory tails available"
            if possible
            else "no unsafe trajectories"
        )

    elif case.fault_type is FaultType.UNREACHABLE_START:
        possible = record.successes > 0 and record.failures > 0
        reason = (
            "both successful and failed starts available"
            if possible
            else "PastRefine needs both successes and failures"
        )

    elif case.fault_type is FaultType.MISSING_WAYPOINT:
        possible = record.successes > 0
        reason = (
            "successful midpoints available"
            if possible
            else "AddRefine needs at least one successful trajectory"
        )

    elif case.fault_type is FaultType.MISSING_ALTERNATIVE:
        possible = True
        reason = "OrRefine is structural; bottleneck edge was evaluated"

    else:
        possible = True
        reason = "not applicable"

    return {
        "refinement_possible": possible,
        "reason": reason,
        "reached_rollouts": record.reached_rollouts,
        "successes": record.successes,
        "failures": record.failures,
        "unsafe_rollouts": record.unsafe_rollouts,
        "unsafe_reached_rollouts": record.unsafe_reached_rollouts,
        "probability": record.probability,
    }


def _classify_outcome(
    *,
    refinement_possible: bool,
    operator_correct: bool,
    recovered: bool,
    actual_operator: str | None,
) -> str:
    if not refinement_possible:
        return "evidence_starved"
    if operator_correct and recovered:
        return "correct_operator_and_recovery"
    if operator_correct and not recovered:
        return "correct_operator_inexact_repair"
    if actual_operator is None:
        return "no_refinement_committed"
    return "wrong_operator"


def _make_edge_learner(settings: FaultLineSettings, seed: int):
    if settings.learner == "ppo":
        return PPOEdgeLearner(device="cpu", seed=seed)
    if settings.learner == "tabular":
        return TabularLearner()
    raise ValueError("learner must be 'ppo' or 'tabular'")


def _make_graph_learner(settings: FaultLineSettings):
    if settings.method == "base":
        return None
    if settings.method == "dirl":
        return DiRL()
    if settings.method == "lsts":
        if settings.learner != "ppo":
            raise ValueError("LSTS requires PPOEdgeLearner because it needs continue_edge()")
        total = settings.lsts_max_interactions or settings.budget
        return LSTS(
            epsilon=settings.lsts_epsilon,
            alpha=settings.lsts_alpha,
            threshold=settings.threshold,
            tolerance=settings.lsts_tolerance,
            chunk_size=settings.lsts_chunk_size,
            max_interactions=total,
        )
    raise ValueError("method must be 'base', 'dirl', or 'lsts'")


def _initial_best_path(graph: AbstractGraph, run: AutoSpecRun) -> float:
    lookup = {
        (item.source, item.target): item.probability
        for item in run.initial_edge_success
    }
    probabilities = {
        edge: (
            float(lookup[(edge.source.name, edge.target.name)])
            if lookup.get((edge.source.name, edge.target.name)) is not None
            else 0.0
        )
        for edge in graph.edges
    }
    return best_path_product(graph, probabilities)


def graph_to_dict(graph: AbstractGraph) -> dict[str, Any]:
    return {
        "source": graph.source.name if graph.source is not None else None,
        "target": graph.target.name if graph.target is not None else None,
        "vertices": [
            {"name": vertex.name, "region": str(vertex.region)}
            for vertex in graph.vertices
        ],
        "edges": [
            {
                "source": edge.source.name,
                "target": edge.target.name,
                "safe": str(edge.safe),
            }
            for edge in graph.edges
        ],
    }


def _vertex_named(graph: AbstractGraph, name: str | None):
    if name is None:
        return None
    for vertex in graph.vertices:
        if vertex.name == name:
            return vertex
    return None


def _failed_record(
    case: FaultCase,
    settings: FaultLineSettings,
    seed: int,
    elapsed: float,
    counted: CountingLearner,
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "case": case.name,
        "fault_type": case.fault_type.value,
        "severity": case.severity,
        "seed": seed,
        "learner": settings.learner,
        "method": settings.method,
        "expected_operator": case.expected_operator,
        "actual_operator": None,
        "first_committed_operator": None,
        "fault_trigger_operator": None,
        "fault_trigger_edge": (
            list(_fault_trigger_edge(case))
            if _fault_trigger_edge(case) is not None
            else None
        ),
        "expected_operator_used": False,
        "operator_correct": False,
        "outcome_category": "run_failed",
        "refinement_possible": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "cost": {
            "wall_clock_seconds": elapsed,
            "train_edge_calls": counted.train_edge_calls,
            "continue_edge_calls": counted.continue_edge_calls,
            "estimate_calls": counted.estimate_calls,
            "requested_training_steps": counted.requested_training_steps,
            "evaluation_rollouts": counted.evaluation_rollouts,
        },
    }


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "status",
        "case",
        "fault_type",
        "severity",
        "seed",
        "learner",
        "method",
        "expected_operator",
        "actual_operator",
        "first_committed_operator",
        "fault_trigger_operator",
        "fault_trigger_source",
        "fault_trigger_target",
        "expected_operator_used",
        "operator_correct",
        "outcome_category",
        "refinement_possible",
        "evidence_reason",
        "baseline_reached_trajectories",
        "baseline_successful_trajectories",
        "baseline_failed_trajectories",
        "baseline_unsafe_trajectories",
        "baseline_unsafe_reached_trajectories",
        "baseline_affected_edge_probability",
        "initial_best_path_success",
        "final_best_path_success",
        "best_path_improvement",
        "metric_type",
        "baseline_iou",
        "repaired_iou",
        "iou_improvement",
        "normalized_recovery_gain",
        "baseline_precision",
        "baseline_recall",
        "repaired_precision",
        "repaired_recall",
        "baseline_weighted_iou",
        "repaired_weighted_iou",
        "weighted_iou_improvement",
        "subset_fraction",
        "normalized_graph_edit_distance",
        "graph_similarity",
        "missing_edge_recovered",
        "topology_recovered",
        "recovered",
        "recovery_reason",
        "wall_clock_seconds",
        "train_edge_calls",
        "continue_edge_calls",
        "estimate_calls",
        "requested_training_steps",
        "evaluation_rollouts",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for record in records:
            recovery = record.get("recovery", {})
            cost = record.get("cost", {})
            trigger = record.get("fault_trigger_edge") or [None, None]
            row = {column: None for column in columns}

            for key in [
                "status",
                "case",
                "fault_type",
                "severity",
                "seed",
                "learner",
                "method",
                "expected_operator",
                "actual_operator",
                "first_committed_operator",
                "fault_trigger_operator",
                "expected_operator_used",
                "operator_correct",
                "outcome_category",
                "refinement_possible",
                "evidence_reason",
                "baseline_reached_trajectories",
                "baseline_successful_trajectories",
                "baseline_failed_trajectories",
                "baseline_unsafe_trajectories",
                "baseline_unsafe_reached_trajectories",
                "baseline_affected_edge_probability",
                "initial_best_path_success",
                "final_best_path_success",
                "best_path_improvement",
                "error",
            ]:
                row[key] = record.get(key)

            row["fault_trigger_source"] = trigger[0] if len(trigger) > 0 else None
            row["fault_trigger_target"] = trigger[1] if len(trigger) > 1 else None

            for key in [
                "metric_type",
                "baseline_iou",
                "repaired_iou",
                "iou_improvement",
                "normalized_recovery_gain",
                "baseline_precision",
                "baseline_recall",
                "repaired_precision",
                "repaired_recall",
                "baseline_weighted_iou",
                "repaired_weighted_iou",
                "weighted_iou_improvement",
                "subset_fraction",
                "normalized_graph_edit_distance",
                "graph_similarity",
                "missing_edge_recovered",
                "topology_recovered",
                "recovered",
                "recovery_reason",
            ]:
                row[key] = recovery.get(key)

            for key in [
                "wall_clock_seconds",
                "train_edge_calls",
                "continue_edge_calls",
                "estimate_calls",
                "requested_training_steps",
                "evaluation_rollouts",
            ]:
                row[key] = cost.get(key)

            writer.writerow(row)


def _print_batch_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    ok = [record for record in records if record.get("status") == "ok"]
    diagnosable = [record for record in ok if bool(record.get("refinement_possible"))]

    print("\n" + "#" * 88)
    print("EXPERIMENT COMPLETE")
    print("#" * 88)
    print(f"Successful runs   : {len(ok)}/{len(records)}")
    print(f"Diagnosable runs  : {len(diagnosable)}/{len(ok) if ok else 0}")

    if ok:
        correct_all = sum(bool(record.get("operator_correct")) for record in ok)
        print(f"Operator accuracy : {correct_all}/{len(ok)} (all successful runs)")

    if diagnosable:
        correct_diag = sum(bool(record.get("operator_correct")) for record in diagnosable)
        recovered_diag = sum(
            bool(record.get("recovery", {}).get("recovered"))
            for record in diagnosable
        )
        print(
            "Operator accuracy | evidence available: "
            f"{correct_diag}/{len(diagnosable)}"
        )
        print(
            "Ground-truth recovery | evidence available: "
            f"{recovered_diag}/{len(diagnosable)}"
        )
    else:
        print("Operator accuracy | evidence available: n/a")
        print("Ground-truth recovery | evidence available: n/a")

    categories: dict[str, int] = {}
    for record in ok:
        category = str(record.get("outcome_category", "unknown"))
        categories[category] = categories.get(category, 0) + 1
    if categories:
        print("Outcome categories:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")

    print(f"Results written to: {output_dir.resolve()}")


def _print_optional(label: str, value: Any) -> None:
    if value is not None:
        print(f"{label}: {float(value):.3f}")

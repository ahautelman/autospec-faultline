"""Time one benchmark instance end to end (gold baselines, gates, one evaluation seed).

    python scripts/time_instance.py --cache-dir none            # code without the cache
    python scripts/time_instance.py --cache-dir /tmp/c          # cold cache, then rerun: warm
    python scripts/time_instance.py --cache-dir /tmp/c --score-all

Prints a JSON line with the wall-clock of each phase, the number of edge trainings
(new cache files) and the instance's AutoSpec outcome, so cached and uncached runs can be
compared for both time and result.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from faultline.benchmark.golds import rooms9_gold
from faultline.benchmark.instances import generate
from faultline.benchmark.runner import Settings, gold_baselines, run_instance
from faultline.faults.gates import Funnel
from faultline.faults.ground_truth import FaultKind
from faultline.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="mixed_star")
    parser.add_argument("--kind", default="goal_inflation")
    parser.add_argument("--level", default="mid")
    parser.add_argument("--seed", type=int, default=0, help="evaluation seed")
    parser.add_argument("--cache-dir", default="none")
    parser.add_argument("--score-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    protocol = load_protocol()
    cache = None if args.cache_dir == "none" else args.cache_dir
    settings = Settings(protocol, cache_dir=cache, score_all_operators=args.score_all)
    gold = rooms9_gold(args.gold, resolution=protocol["metrics"]["grid_resolution"], protocol_hash=protocol.hash)
    kind = FaultKind(args.kind)
    instances = generate([gold], families=(kind,), levels=(args.level,), seeds=(0,), protocol=protocol, funnel=Funnel())
    instance = next(i for i in instances if i.kind is kind)

    def files() -> int:
        return sum(1 for _ in Path(cache).rglob("*.npz")) if cache else 0

    timings, before = {}, files()
    t = time.perf_counter()
    base = gold_baselines(settings, gold)
    timings["gold_baselines"] = time.perf_counter() - t
    with tempfile.TemporaryDirectory() as rq2:
        t = time.perf_counter()
        record = run_instance(instance, settings, base, seeds=(args.seed,), skip_gates=True,
                              rq2_dir=Path(rq2) if args.score_all else None)
        timings["instance"] = time.perf_counter() - t
    timings["gates"] = record["gate"]["seconds"]
    timings["eval_seed"] = record["runs"][0]["seconds"]
    run = record["runs"][0]
    print(json.dumps({
        "instance": instance.id, "cache": cache is not None, "score_all": args.score_all,
        "seconds": {k: round(v, 1) for k, v in timings.items()},
        "total_seconds": round(timings["gold_baselines"] + timings["instance"], 1),
        "edge_trainings": files() - before if cache else None,
        "gate": record["gate"]["stage"],
        "autospec": {"best": run["autospec"]["best_path_success"], "steps": run["autospec"]["steps"]},
        "random_operator_best": run["random_operator"]["best_path_success"],
    }))


if __name__ == "__main__":
    main()

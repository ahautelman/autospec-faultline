"""Command line: generate instances and report the funnel, then optionally run them.

    python -m faultline.benchmark funnel --out reports/faultline/funnel.json
    python -m faultline.benchmark run --out reports/faultline --limit 3 --seeds 0

``funnel`` is the checkpoint before any evaluation run: it needs no learner and prints
generated / valid counts per family. ``run`` screens each instance with the learner and
writes one JSON record per instance (resumable: existing records are skipped).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from faultline.benchmark.golds import ROOMS9_LAYOUTS, rooms9_gold, world_gold
from faultline.benchmark.instances import generate
from faultline.benchmark.runner import Settings, gold_baselines, run_instance, write_record
from faultline.faults.gates import Funnel
from faultline.protocol import load_protocol


def build_golds(protocol, worlds: list[str]):
    """Build the rooms9 golds (with and without a trap) and any requested fixed worlds."""
    res = protocol["metrics"]["grid_resolution"]
    golds = []
    for name in ROOMS9_LAYOUTS:
        for trap in (False, True):
            gold = rooms9_gold(name, resolution=res, protocol_hash=protocol.hash, trap=trap)
            if gold is not None and gold.certificate.tier1_ok:
                golds.append(gold)
    for world in worlds:
        gold = world_gold(world, resolution=res, protocol_hash=protocol.hash)
        if gold.certificate.tier1_ok:
            golds.append(gold)
    return golds


def main() -> None:
    parser = argparse.ArgumentParser(prog="faultline.benchmark")
    parser.add_argument("command", choices=("funnel", "run"))
    parser.add_argument("--out", type=Path, default=Path("reports/faultline"))
    parser.add_argument("--worlds", nargs="*", default=[])
    parser.add_argument("--instance-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="evaluation seeds")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--learner", default="ppo")
    parser.add_argument("--only", default=None, help="substring of the instance id")
    args = parser.parse_args()

    protocol = load_protocol()
    funnel = Funnel()
    golds = build_golds(protocol, args.worlds)
    instances = generate(golds, seeds=args.instance_seeds, protocol=protocol, funnel=funnel)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "funnel.json").write_text(json.dumps(funnel.to_dict(), indent=2))
    print(json.dumps(funnel.to_dict()["counts"], indent=1))
    print(f"{len(golds)} golds, {len(instances)} valid instances")
    if args.command == "funnel":
        return

    settings = Settings(protocol, learner=args.learner)
    if args.only:
        instances = [i for i in instances if args.only in i.id]
    instances = instances[: args.limit]
    cache: dict[str, dict] = {}
    for inst in instances:
        target = args.out / (inst.id.replace("|", "__") + ".json")
        if target.exists():
            continue
        if inst.gold.id not in cache:
            cache[inst.gold.id] = gold_baselines(settings, inst.gold)
        record = run_instance(inst, settings, cache[inst.gold.id], seeds=tuple(args.seeds) if args.seeds else None, log=print)
        write_record(record, args.out)


if __name__ == "__main__":
    main()

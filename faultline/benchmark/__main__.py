"""Command line: certify golds, report the funnel, run instances.

    python -m faultline.benchmark certify --out reports/faultline --workers 6
    python -m faultline.benchmark funnel  --out reports/faultline
    python -m faultline.benchmark run     --out reports/faultline --workers 6 --seeds 0 1 2 3 4

``certify`` runs Tier 2 and Tier 3 with the protocol's seeds on every gold (one gold per
worker) and writes ``certificates.json``. ``funnel`` is the checkpoint before any
evaluation run: generation counts per family, no learner. ``run`` uses only golds that
certified (pass ``--uncertified`` to override, for pilots), screens each instance with the
learner and writes one JSON record per instance (resumable: existing records are skipped).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from faultline.benchmark.golds import ROOMS9_LAYOUTS, rooms9_gold, world_gold
from faultline.benchmark.instances import generate
from faultline.benchmark.runner import Settings, gold_baselines, run_instance, write_record
from faultline.faults.gates import Funnel
from faultline.golden.certify import certify_learning
from faultline.golden.env import make_env
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


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _certify(gold, settings):
    """Certify one gold with the protocol seeds (runs in a worker)."""
    p = settings.protocol
    return certify_learning(
        gold, lambda: make_env(gold), lambda: settings.edge_learner(0), settings.graph_learner,
        config=settings.config, cert_seeds=p["seeds"]["certification"],
        holdout_seeds=p["seeds"]["certification_holdout"], p_gold_min=p["certification"]["p_gold_min"],
        p_alt_min=p["certification"]["p_alt_min"], confidence=p["certification"]["confidence"],
        protocol_hash=p.hash,
    )


_GOLD_CACHE: dict[str, dict] = {}


def _run_one(instance, settings, seeds, out):
    """Run one instance in a worker, caching gold baselines per worker process."""
    if instance.gold.id not in _GOLD_CACHE:
        _GOLD_CACHE[instance.gold.id] = gold_baselines(settings, instance.gold)
    record = run_instance(instance, settings, _GOLD_CACHE[instance.gold.id], seeds=seeds, log=print)
    write_record(record, out)
    return instance.id


def _certified(golds, out: Path, require: bool):
    path = out / "certificates.json"
    if not require:
        return golds
    if not path.exists():
        raise SystemExit("No certificates.json: run the certify command first (or pass --uncertified)")
    ok = {k for k, v in json.loads(path.read_text()).items() if v["certified"]}
    return [g for g in golds if g.id in ok]


def main() -> None:
    parser = argparse.ArgumentParser(prog="faultline.benchmark")
    parser.add_argument("command", choices=("certify", "funnel", "run"))
    parser.add_argument("--out", type=Path, default=Path("reports/faultline"))
    parser.add_argument("--worlds", nargs="*", default=[])
    parser.add_argument("--instance-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="evaluation seeds")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--learner", default="ppo")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--uncertified", action="store_true")
    parser.add_argument("--only", default=None, help="substring of the instance id")
    args = parser.parse_args()

    protocol = load_protocol()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = Settings(protocol, learner=args.learner)
    golds = build_golds(protocol, args.worlds)
    context = multiprocessing.get_context("spawn")

    if args.command == "certify":
        results = {}
        with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker) as pool:
            futures = {pool.submit(_certify, g, settings): g.id for g in golds}
            for future in as_completed(futures):
                gold = future.result()
                c = gold.certificate
                results[gold.id] = {
                    "certified": c.certified, "tier1": c.tier1_ok, "tier2": c.tier2_ok, "tier3": c.tier3_ok,
                    "problems": list(c.problems), "notes": list(c.notes), "protocol_hash": c.protocol_hash,
                    "edge_cp_lower": {k: list(v) for k, v in c.edge_cp_lower.items()},
                }
                print(gold.id, "certified" if c.certified else f"FAILED {c.problems}", flush=True)
                (args.out / "certificates.json").write_text(json.dumps(results, indent=2))
        return

    golds = _certified(golds, args.out, require=args.command == "run" and not args.uncertified)
    funnel = Funnel()
    instances = generate(golds, seeds=args.instance_seeds, protocol=protocol, funnel=funnel)
    (args.out / "funnel.json").write_text(json.dumps(funnel.to_dict(), indent=2))
    print(json.dumps(funnel.to_dict()["counts"], indent=1))
    print(f"{len(golds)} golds, {len(instances)} valid instances")
    if args.command == "funnel":
        return

    if args.only:
        instances = [i for i in instances if args.only in i.id]
    instances = instances[: args.limit]
    pending = [i for i in instances if not (args.out / (i.id.replace("|", "__") + ".json")).exists()]
    seeds = tuple(args.seeds) if args.seeds else None
    if args.workers <= 1:
        for inst in pending:
            _run_one(inst, settings, seeds, args.out)
        return
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=_init_worker) as pool:
        for future in as_completed([pool.submit(_run_one, i, settings, seeds, args.out) for i in pending]):
            print("done", future.result(), flush=True)


if __name__ == "__main__":
    main()

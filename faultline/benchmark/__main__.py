"""Command line: certify golds, report the funnel, run instances.

    python -m faultline.benchmark certify --out reports/faultline --workers 6
    python -m faultline.benchmark funnel  --out reports/faultline
    python -m faultline.benchmark screen  --out reports/faultline --workers 6
    python -m faultline.benchmark run     --out reports/faultline --workers 6 --seeds 0 1 2 3 4
    python -m faultline.benchmark analyze --out reports/faultline

``screen`` runs only the effectiveness and observability gates on every instance and writes
``screen_funnel.json``: the checkpoint before any evaluation run. ``certify`` runs Tier 2 and Tier 3 with the protocol's seeds on every gold (one gold per
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

from faultline.benchmark import workers
from faultline.benchmark.golds import EXCLUDED_ROOMS9, ROOMS9_LAYOUTS, rooms9_gold, world_gold
from faultline.benchmark.instances import generate
from faultline.benchmark.runner import Settings
from faultline.faults.gates import Funnel
from faultline.protocol import load_protocol


def build_golds(protocol, worlds: list[str]):
    """Build the rooms9 golds (with and without a trap) and any requested fixed worlds."""
    res = protocol["metrics"]["grid_resolution"]
    golds = []
    for name in ROOMS9_LAYOUTS:
        if name in EXCLUDED_ROOMS9:
            continue
        for trap, shortcut in ((False, False), (True, False), (False, True)):
            gold = rooms9_gold(name, resolution=res, protocol_hash=protocol.hash, trap=trap, shortcut=shortcut)
            if gold is not None and gold.certificate.tier1_ok:
                golds.append(gold)
    for world in worlds:
        gold = world_gold(world, resolution=res, protocol_hash=protocol.hash)
        if gold.certificate.tier1_ok:
            golds.append(gold)
    return golds


def _certified(golds, out: Path, require: bool):
    path = out / "certificates.json"
    if not require:
        return golds
    if not path.exists():
        raise SystemExit("No certificates.json: run the certify command first (or pass --uncertified)")
    ok = {k for k, v in json.loads(path.read_text()).items() if v["certified"]}
    return [g for g in golds if g.id in ok]


def write_screen_funnel(out: Path) -> dict:
    """Aggregate the screen records into per-family stage counts and print them."""
    counts: dict[str, dict[str, int]] = {}
    for path in sorted((out / "screen").glob("*.json")):
        r = json.loads(path.read_text())
        stage = r.get("stratum") or r["gate"]["stage"]
        counts.setdefault(r["kind"], {}).setdefault(stage, 0)
        counts[r["kind"]][stage] += 1
    (out / "screen_funnel.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=1))
    return counts


def _warm_golds(pool, instances, settings, out: Path) -> None:
    """Learn every gold baseline once, in parallel, before the instances need them.

    Without this every worker relearned each gold's baselines; with the shared estimate
    cache the instances then read them from disk.
    """
    golds = {i.gold.id: i.gold for i in instances}
    for future in as_completed([pool.submit(workers.warm_gold, g, settings, out) for g in golds.values()]):
        print("gold baselines ready:", future.result(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="faultline.benchmark")
    parser.add_argument("command", choices=("certify", "funnel", "screen", "run", "analyze"))
    parser.add_argument("--out", type=Path, default=Path("reports/faultline"))
    parser.add_argument("--worlds", nargs="*", default=[])
    parser.add_argument("--instance-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="evaluation seeds")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--learner", default="ppo")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--uncertified", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="keep ids containing any of these substrings")
    parser.add_argument("--cache-dir", default="reports/cache",
                        help="exact edge-estimate cache shared by all runs; 'none' disables it")
    parser.add_argument("--no-score-all", action="store_true",
                        help="RQ2: do not score the operators the loop skipped after committing")
    args = parser.parse_args()

    protocol = load_protocol()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "analyze":
        from faultline.benchmark.analyze import analyze
        analyze(args.out, threshold=protocol["autospec"]["threshold"],
                resamples=protocol["statistics"]["bootstrap_resamples"])
        return
    settings = Settings(protocol, learner=args.learner,
                        cache_dir=None if args.cache_dir == "none" else args.cache_dir,
                        score_all_operators=not args.no_score_all)
    golds = build_golds(protocol, args.worlds)
    context = multiprocessing.get_context("spawn")

    if args.command == "certify":
        results = {}
        with ProcessPoolExecutor(args.workers, mp_context=context, initializer=workers.init_worker) as pool:
            chosen = [g for g in golds if not args.only or any(o in g.id for o in args.only)]
            futures = {pool.submit(workers.certify_gold, g, settings): g.id for g in chosen}
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

    golds = _certified(golds, args.out, require=not args.uncertified)
    funnel = Funnel()
    instances = generate(golds, seeds=args.instance_seeds, protocol=protocol, funnel=funnel)
    (args.out / "funnel.json").write_text(json.dumps(funnel.to_dict(), indent=2))
    print(json.dumps(funnel.to_dict()["counts"], indent=1))
    print(f"{len(golds)} golds, {len(instances)} valid instances")
    if args.command == "funnel":
        return

    if args.only:
        instances = [i for i in instances if any(o in i.id for o in args.only)]
    instances = instances[: args.limit]
    if args.command == "screen":
        done = {p.stem for p in (args.out / "screen").glob("*.json")}
        pending = [i for i in instances if i.id.replace("|", "__") not in done]
        with ProcessPoolExecutor(args.workers, mp_context=context, initializer=workers.init_worker) as pool:
            _warm_golds(pool, pending, settings, args.out)
            for future in as_completed([pool.submit(workers.screen_one, i, settings, args.out) for i in pending]):
                print("screened", future.result(), flush=True)
        write_screen_funnel(args.out)
        return
    pending = [i for i in instances if not (args.out / (i.id.replace("|", "__") + ".json")).exists()]
    seeds = tuple(args.seeds) if args.seeds else None
    if args.workers <= 1:
        for gold in {i.gold.id: i.gold for i in pending}.values():
            workers.warm_gold(gold, settings, args.out)
        for inst in pending:
            workers.run_one(inst, settings, seeds, args.out)
        return
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=workers.init_worker) as pool:
        _warm_golds(pool, pending, settings, args.out)
        for future in as_completed([pool.submit(workers.run_one, i, settings, seeds, args.out) for i in pending]):
            print("done", future.result(), flush=True)


if __name__ == "__main__":
    main()

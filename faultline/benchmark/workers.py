"""Functions executed in pool workers; they live here so spawned processes can import them."""

from __future__ import annotations

from faultline.benchmark.runner import gold_baselines, run_instance, write_record
from faultline.golden.certify import certify_learning
from faultline.golden.env import make_env

_GOLD_CACHE: dict[str, dict] = {}


def init_worker() -> None:
    """Limit torch to one thread per worker so a pool does not oversubscribe the cores."""
    import torch

    torch.set_num_threads(1)


def certify_gold(gold, settings):
    """Certify one gold with the protocol seeds."""
    p = settings.protocol
    return certify_learning(
        gold, lambda: make_env(gold), lambda: settings.edge_learner(0), settings.graph_learner,
        config=settings.config, cert_seeds=p["seeds"]["certification"],
        holdout_seeds=p["seeds"]["certification_holdout"], p_gold_min=p["certification"]["p_gold_min"],
        p_alt_min=p["certification"]["p_alt_min"], confidence=p["certification"]["confidence"],
        protocol_hash=p.hash,
    )


def run_one(instance, settings, seeds, out):
    """Run one instance, caching gold baselines per worker process."""
    if instance.gold.id not in _GOLD_CACHE:
        _GOLD_CACHE[instance.gold.id] = gold_baselines(settings, instance.gold)
    record = run_instance(instance, settings, _GOLD_CACHE[instance.gold.id], seeds=seeds, log=print)
    write_record(record, out)
    return instance.id


def screen_one(instance, settings, out):
    """Run only the gates of one instance and write the record under ``out/screen``."""
    if instance.gold.id not in _GOLD_CACHE:
        _GOLD_CACHE[instance.gold.id] = gold_baselines(settings, instance.gold)
    record = run_instance(instance, settings, _GOLD_CACHE[instance.gold.id], gates_only=True, log=print)
    write_record(record, out / "screen")
    return instance.id

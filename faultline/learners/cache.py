"""An exact, disk-backed memo of edge estimates shared by every run of the benchmark.

Training one PPO edge takes ~15 s and dominates the benchmark's runtime, while the same
edge is trained many times: the effectiveness gate, the AutoSpec run and the
random-operator baseline all start with the same unrefined phase for the same seed, the
random-operator baseline scores many of the candidates AutoSpec scored, and every pool
worker used to relearn the gold baselines. Training is deterministic given the
environment, the edge, the budget, the recipe and the seed (SB3 reseeds the environment,
and the DiRL start-distribution wrapper with it, at the start of ``learn``; every
evaluation episode is reseeded), so the estimate is a pure function of those inputs and
can be looked up instead of recomputed.

The key covers every input of that function (see :func:`estimate_key`). A cache hit is
therefore bit-identical to retraining: the rollouts are stored losslessly. Bump
:data:`CACHE_VERSION` whenever the environment dynamics, the reward, the learner or the
episode loop change, because those are code, not data, and cannot be fingerprinted.

This is unrelated to ``AutoSpecConfig.reuse_candidate_cache``, which changes which
evidence the loop uses (an ablation); this cache never changes a result.
"""

from __future__ import annotations

import functools
import hashlib
import io
import os
import tempfile
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faultline.graph.abstract_graph import Edge
from faultline.learners.base import EdgeEstimate, Rollout

# Bump when env dynamics, reward, PPO training or ``run_episode`` change.
CACHE_VERSION = 1


_CODE_TYPES = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, functools.partial, type)


def fingerprint(value: Any) -> bytes:
    """Return an exact, deterministic byte encoding of a plain value or object.

    How:
        Floats are encoded with ``float.hex`` and arrays by dtype, shape and raw bytes, so
        two values share an encoding only when they are equal bit for bit. Objects are
        encoded by class name and their sorted public attributes (predicates, layouts),
        dictionaries by sorted encoded items. Unknown types raise instead of falling back
        to ``repr``, so a key can never silently ignore an input.

    Raises:
        TypeError: For values with no exact encoding (callables, open handles, ...).
    """
    if value is None or isinstance(value, (bool, str)):
        return f"{type(value).__name__}:{value}".encode()
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}".encode()
    if isinstance(value, (float, np.floating)):
        return f"float:{float(value).hex()}".encode()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return b"nd:" + f"{array.dtype.str}{array.shape}".encode() + array.tobytes()
    if isinstance(value, (tuple, list)):
        return b"seq(" + b",".join(fingerprint(v) for v in value) + b")"
    if isinstance(value, dict):
        items = sorted(fingerprint(k) + b"=" + fingerprint(v) for k, v in value.items())
        return b"map(" + b",".join(items) + b")"
    if hasattr(value, "__dict__") and not isinstance(value, _CODE_TYPES):
        attrs = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return f"obj:{type(value).__qualname__}".encode() + fingerprint(attrs)
    raise TypeError(f"No exact fingerprint for {type(value).__qualname__}")


def env_fingerprint(env: Any) -> bytes:
    """Return the fingerprint of an environment, including any DiRL start distribution.

    How:
        Unwraps :class:`~faultline.learners.start_distribution.StartDistributionWrapper`
        layers, recording their start states, then encodes the base environment's class,
        layout, horizon and trap together with the module constants of its dynamics and
        reward. The subgoal (source, target, safe) is part of the edge, not of this.
    """
    parts = []
    while hasattr(env, "start_states") and hasattr(env, "env"):
        states = env.start_states
        parts.append(fingerprint(None if not states else np.stack([np.asarray(s, dtype=float) for s in states])))
        env = env.env
    module = __import__(type(env).__module__, fromlist=["_"])
    constants = {
        name: getattr(module, name)
        for name in ("MAX_SPEED", "MIN_SPEED", "MAX_HEADING", "SUCCESS_RADIUS", "SUCCESS_REWARD",
                     "VIOLATION_REWARD", "STEP_PENALTY", "PROGRESS_SCALE")
        if hasattr(module, name)
    }
    parts.append(fingerprint({
        "class": type(env).__qualname__, "layout": env.layout, "max_steps": env.max_steps,
        "unsafe": env.unsafe_region, "constants": constants,
    }))
    return b"env(" + b"|".join(parts) + b")"


@functools.cache
def _versions() -> bytes:
    """Fingerprint of the cache version and the installed learning stack."""
    import importlib.util
    from importlib.metadata import version

    versions = {"cache": CACHE_VERSION}
    for package in ("gymnasium", "torch", "stable-baselines3"):
        if importlib.util.find_spec(package.replace("-", "_")) is not None:
            versions[package] = version(package)
    return fingerprint(versions)


def estimate_key(learner_id: bytes, env: Any, edge: Edge, budget: int, train_seed: int | None,
                 n_rollouts: int, estimate_seed: int) -> str:
    """Return the cache key of one (train, estimate) pair.

    Args:
        learner_id: Fingerprint of the edge learner's configuration.
        env: The environment the edge is trained and estimated in (possibly wrapped).
        edge: The edge.
        budget: Training budget.
        train_seed: Effective training seed.
        n_rollouts: Rollouts in the estimate.
        estimate_seed: Seed of the estimate.

    Returns:
        A hex digest.
    """
    digest = hashlib.sha256()
    for part in (
        _versions(), learner_id, env_fingerprint(env),
        fingerprint((edge.source.region, edge.target.region, edge.safe)),
        fingerprint((int(budget), train_seed, int(n_rollouts), int(estimate_seed))),
    ):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def encode_estimate(estimate: EdgeEstimate) -> bytes:
    """Serialise an estimate losslessly to compressed ``.npz`` bytes.

    How:
        States are concatenated with per-rollout lengths. They are stored as float32 only
        when that is exact (the rooms observations are float32), otherwise as float64.
        ``final_safe`` is stored as -1 (unknown), 0 or 1.
    """
    rollouts = estimate.rollouts
    lengths = np.array([len(r.states) for r in rollouts], dtype=np.int64)
    states = (np.concatenate([np.stack([np.asarray(s, dtype=float) for s in r.states]) for r in rollouts])
              if rollouts else np.zeros((0, 0)))
    narrow = states.astype(np.float32)
    if np.array_equal(narrow.astype(float), states):
        states = narrow
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        success_probability=np.float64(estimate.success_probability),
        lengths=lengths,
        states=states,
        reached=np.array([r.reached for r in rollouts], dtype=bool),
        safe=np.array([r.safe for r in rollouts], dtype=bool),
        final_safe=np.array([-1 if r.final_safe is None else int(r.final_safe) for r in rollouts], dtype=np.int8),
    )
    return buffer.getvalue()


def decode_estimate(data: bytes | Path) -> EdgeEstimate:
    """Inverse of :func:`encode_estimate` (accepts bytes or a file path)."""
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with np.load(source) as npz:
        states = npz["states"].astype(float)
        bounds = np.concatenate([[0], np.cumsum(npz["lengths"])])
        rollouts = tuple(
            Rollout(
                tuple(states[i] for i in range(start, stop)),
                reached=bool(reached), safe=bool(safe),
                final_safe=None if final < 0 else bool(final),
            )
            for start, stop, reached, safe, final in zip(
                bounds[:-1], bounds[1:], npz["reached"], npz["safe"], npz["final_safe"]
            )
        )
        return EdgeEstimate(float(npz["success_probability"]), rollouts)


class EstimateStore:
    """A directory of encoded estimates, safe to share between processes.

    Args:
        root: Directory; created on first write.

    How:
        One ``<key[:2]>/<key>.npz`` file per key. Writes go to a temporary file in the
        same directory and are renamed into place, so a reader never sees a partial file
        and two workers writing the same key both write identical bytes.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def path(self, key: str) -> Path:
        """Return the file of a key."""
        return self.root / key[:2] / f"{key}.npz"

    def get(self, key: str) -> EdgeEstimate | None:
        """Return the stored estimate, or ``None``."""
        path = self.path(key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return decode_estimate(path)

    def put(self, key: str, estimate: EdgeEstimate) -> None:
        """Store an estimate atomically."""
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(handle, "wb") as out:
            out.write(encode_estimate(estimate))
        os.replace(tmp, path)


@dataclass
class LazyPolicy:
    """A policy handle that trains only when someone needs the policy itself.

    The cached learner hands this out from ``train_edge``; when the estimate is cached the
    policy is never trained. :attr:`model` trains it on demand (for example for a replay
    figure), with exactly the arguments ``train_edge`` received.
    """

    train: Callable[[], Any]
    train_seed: int | None
    budget: int
    _model: Any = None

    @property
    def model(self) -> Any:
        """Return the trained policy, training it on first access."""
        if self._model is None:
            self._model = self.train()
        return self._model


class CachedLearner:
    """Wrap an edge learner so that estimates are looked up before anything is trained.

    Args:
        inner: The edge learner (:class:`~faultline.learners.PPOEdgeLearner`, tabular, ...).
        store: Where estimates live.

    How:
        ``train_edge`` returns a :class:`LazyPolicy` without training. ``estimate_reach``
        builds the full key; on a hit it returns the stored estimate, on a miss it trains
        through the lazy handle, estimates with the inner learner and stores the result.
        The training environment and the estimate environment are the same object in
        every caller (the loop, DiRL and the gates), and the key fingerprints it at
        estimate time. ``continue_edge`` (LSTS) is not cacheable: it delegates to the inner
        learner after materialising the policy.
    """

    def __init__(self, inner: Any, store: EstimateStore) -> None:
        self.inner = inner
        self.store = store
        self.learner_id = fingerprint({
            "class": type(inner).__qualname__,
            "recipe": getattr(inner, "recipe", None),
            "device": getattr(inner, "device", None),
        })

    def train_edge(self, env: Any, edge: Edge, budget: int | None = None, callback: Any | None = None,
                   seed: int | None = None) -> LazyPolicy:
        """Return a lazy policy; training happens only on a cache miss."""
        if callback is not None:
            raise ValueError("CachedLearner cannot cache a training run with a callback")
        effective_budget = budget if budget is not None else getattr(self.inner, "budget", 0)
        effective_seed = seed if seed is not None else getattr(self.inner, "seed", None)
        return LazyPolicy(
            train=lambda: self.inner.train_edge(env, edge, effective_budget, seed=effective_seed),
            train_seed=effective_seed, budget=int(effective_budget),
        )

    def estimate_reach(self, env: Any, edge: Edge, policy: Any, n_rollouts: int = 1000, seed: int = 0) -> EdgeEstimate:
        """Return the cached estimate, or train, estimate and store it."""
        if not isinstance(policy, LazyPolicy):
            return self.inner.estimate_reach(env, edge, policy, n_rollouts, seed)
        key = estimate_key(self.learner_id, env, edge, policy.budget, policy.train_seed, n_rollouts, seed)
        estimate = self.store.get(key)
        if estimate is None:
            estimate = self.inner.estimate_reach(env, edge, policy.model, n_rollouts, seed)
            self.store.put(key, estimate)
        return estimate

    def continue_edge(self, env: Any, edge: Edge, policy: Any, budget: int, seed: int | None = None) -> Any:
        """Delegate to the inner learner (not cacheable)."""
        model = policy.model if isinstance(policy, LazyPolicy) else policy
        return self.inner.continue_edge(env, edge, model, budget, seed=seed)

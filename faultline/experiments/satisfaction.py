"""Specification-level satisfaction of a refined route, for the 9-rooms report.

The report's per-edge curves score what the learner trains on: reaching a goal
*point* sampled inside the target region (the authors' code does the same). That
is not SpectRL's ``achieve b``, which only asks to reach *some* state of ``b``.
It also lets a refinement look better by construction: ReachRefine stops sampling
unreachable goals, and PastRefine stops sampling failing starts.

This module measures the number those curves cannot: the probability that the
refined route, run end to end, satisfies the *original* edge. Each episode starts
from the original edge's source region, runs the route's edge policies one after
another (each continuing from where the previous one stopped), and the joined
trajectory is judged by the graph semantics of the original edge. A start that
the refined route does not cover (cut away by PastRefine) counts as a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from faultline.envs import RoomsEnv
from faultline.graph import models_trajectory
from faultline.graph.abstract_graph import Edge
from faultline.learners.ppo import run_episode
from faultline.scenarios.catalogue import single_edge

ActionFn = Callable[[np.ndarray], np.ndarray]
# One route edge paired with the action function of the policy trained for it.
RouteStep = tuple[Edge, ActionFn]


def route_satisfaction(
    env: RoomsEnv,
    edge: Edge,
    route: Sequence[RouteStep],
    n_rollouts: int,
    seed: int,
) -> float:
    """Estimate how often a route, run end to end, satisfies the original edge.

    Args:
        env: The environment to roll out in.
        edge: The original (unrefined) edge whose specification is judged.
        route: The edges standing in for ``edge``, in order, each paired with
            its policy's action function (observation to action, for example
            from :func:`~faultline.learners.ppo.greedy_action_fn`); a single
            ``(edge, action)`` step scores the unrefined policy itself.
        n_rollouts: Number of episodes; must be positive.
        seed: Base seed; episode ``i`` reseeds the environment with ``seed + i``.

    How:
        Builds a one-edge graph from ``edge`` as the reference specification,
        chains the route per episode (:func:`_chained_states`), and checks each
        joined trajectory with
        :func:`~faultline.graph.semantics.models_trajectory`, which applies the
        region semantics of ``achieve`` and ``ensuring``.

    Returns:
        The fraction of episodes whose trajectory satisfies ``edge``.

    Raises:
        ValueError: If ``n_rollouts`` is not positive or ``route`` is empty.
    """
    if n_rollouts < 1:
        raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
    if not route:
        raise ValueError("Route must have at least one edge")
    specification = single_edge(edge.source.region, edge.target.region, edge.safe)
    satisfied = sum(
        models_trajectory(specification, _chained_states(env, edge, route, seed + episode))
        for episode in range(n_rollouts)
    )
    return satisfied / n_rollouts


def _chained_states(
    env: RoomsEnv, edge: Edge, route: Sequence[RouteStep], seed: int
) -> list[np.ndarray]:
    """Run a route's policies back to back from one start of the original edge.

    Args:
        env: The environment to roll out in.
        edge: The original edge, whose source region supplies the start state.
        route: The route edges, in order, each with its action function.
        seed: Seed for this episode's reset.

    How:
        Resets with the original source region and the first route edge's target
        and safe regions; a start outside the first route edge's own source region
        is not covered by the route (PastRefine cut it), so nothing runs. Each
        later route edge resets from the previous episode's last position, which
        may lie just outside its source region because an edge succeeds within a
        radius of its goal point, so later hops are not re-checked. The chain
        stops after any edge that did not succeed. Consecutive episodes share
        their boundary state, which is kept once.

    Returns:
        The joined trajectory, empty when the route does not cover the start.
    """
    first, _action = route[0]
    env.set_abstract_states(edge.source.region, first.target.region, first.safe)
    observation, _info = env.reset(seed=seed)
    states: list[np.ndarray] = []
    if not first.source.region.evaluate(observation):
        return states
    for index, (hop, action) in enumerate(route):
        if index > 0:
            env.set_abstract_states(hop.source.region, hop.target.region, hop.safe)
            observation, _info = env.reset(options={"start": observation})
        rollout = run_episode(env, hop, action, observation)
        states.extend(rollout.states[1:] if states else rollout.states)
        observation = rollout.states[-1]
        if not rollout.success:
            break
    return states

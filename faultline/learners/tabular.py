"""A fast, deterministic planner learner for tests.

This learner is not reinforcement learning. It plans a shortest path on a
discretized copy of the environment, then follows that path. It exists to
produce the same output as a real learner cheaply and reproducibly, so the
refinement operators and the loop can be tested without training networks.

The environment samples one goal point inside the target region per episode. The
planner routes to that point. If the point is unreachable, the episode fails.
That is exactly the mechanism behind the paper's low success on a partly
unreachable target region, so the operators see realistic inputs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from faultline.graph.abstract_graph import Edge
from faultline.learners.base import EdgeEstimate, Rollout

CELL_SIZE = 1.0
WAYPOINT_TOLERANCE = 0.6
DEFAULT_BUDGET = 0
MAX_WAYPOINTS = 500

_NEIGHBOUR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class TabularPolicy:
    """The policy handle returned by :class:`TabularLearner`.

    Args:
        cell_size: Grid resolution used for planning.
        tolerance: Distance at which a waypoint counts as reached.

    How:
        The planner has no fitted parameters. It recomputes a route for each
        episode. The handle exists so callers pass an opaque policy object back,
        exactly as they do for the PPO learner.
    """

    cell_size: float = CELL_SIZE
    tolerance: float = WAYPOINT_TOLERANCE


class TabularLearner:
    """A deterministic planner-based learner.

    Args:
        cell_size: Grid resolution for the planner.
        tolerance: Distance at which a waypoint counts as reached.

    How:
        Implements the :class:`~faultline.learners.base.Learner` protocol with a
        breadth-first planner. The planner treats each grid cell centre as a
        node and connects two neighbours only when the straight move between
        them does not cross a solid wall.
    """

    def __init__(self, cell_size: float = CELL_SIZE, tolerance: float = WAYPOINT_TOLERANCE) -> None:
        self.cell_size = float(cell_size)
        self.tolerance = float(tolerance)

    def train_edge(
        self, env: Any, edge: Edge, budget: int = DEFAULT_BUDGET, seed: int | None = None
    ) -> TabularPolicy:
        """Return a planner handle for one edge.

        Args:
            env: The environment, used only for its dimensions.
            edge: The reach-avoid edge to learn.
            budget: Training budget. The planner ignores it because it does not
                fit parameters.
            seed: Accepted for interface compatibility; the planner is
                deterministic, so training has no randomness to seed and the
                value is unused.

        How:
            Records the planner settings on a handle. No learning happens here.

        Returns:
            A :class:`TabularPolicy` handle.
        """
        return TabularPolicy(self.cell_size, self.tolerance)

    def estimate_reach(
        self,
        env: Any,
        edge: Edge,
        policy: TabularPolicy,
        n_rollouts: int,
        seed: int = 0,
    ) -> EdgeEstimate:
        """Roll out the planner for an edge and estimate its success rate.

        Args:
            env: The environment, which implements ``set_abstract_states`` and
                the gymnasium ``reset`` and ``step`` methods.
            edge: The reach-avoid edge under test.
            policy: The handle from :meth:`train_edge`.
            n_rollouts: Number of episodes to sample; must be positive.
            seed: Base seed for the episode RNG.

        How:
            Sets the subgoal, then runs each episode from a freshly seeded
            reset. Evaluation seeds are allocated in disjoint 10,000-seed blocks
            per experiment seed, so different experiment seeds do not reuse almost
            the same rollout initialisations. The planner routes to the sampled goal
            and follows the route. The estimate is the fraction of successful episodes.

        Returns:
            An :class:`EdgeEstimate` with the success probability and rollouts.

        Raises:
            ValueError: If ``n_rollouts`` is not positive.
        """
        if n_rollouts < 1:
            raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
        rollouts = []
        for episode in range(n_rollouts):
            env.set_abstract_states(edge.source.region, edge.target.region, edge.safe)
            episode_seed = int(seed) * 10_000 + episode
            observation, _info = env.reset(seed=episode_seed)
            rollouts.append(self._rollout(env, edge, policy, observation))
        success = sum(1 for rollout in rollouts if rollout.success) / n_rollouts
        return EdgeEstimate(success, tuple(rollouts))

    def _rollout(
        self, env: Any, edge: Edge, policy: TabularPolicy, observation: np.ndarray
    ) -> Rollout:
        """Run one episode by planning and following a route.

        Args:
            env: The environment for the episode.
            edge: The reach-avoid edge under test.
            policy: The planner settings.
            observation: The first observation from ``reset``.

        How:
            Plans a route from the current position to the sampled goal. If no
            route exists, the episode fails with a single-state trajectory.
            Otherwise it follows the route with :meth:`_follow`.

        Returns:
            The rollout for this episode.
        """
        states = [np.asarray(observation, dtype=float)]
        waypoints = self._plan(env, policy, states[0][:2], env.current_goal)
        if waypoints is None:
            return Rollout(tuple(states), reached=False, safe=True)
        reached, safe = self._follow(env, edge, states, waypoints)
        return Rollout(tuple(states), reached=reached, safe=safe)

    def _follow(
        self, env: Any, edge: Edge, states: list[np.ndarray], waypoints: list[np.ndarray]
    ) -> tuple[bool, bool]:
        """Follow a waypoint route and record the visited states.

        Args:
            env: The environment to step.
            edge: The reach-avoid edge under test.
            states: Mutable state list; the trajectory is appended in place.
            waypoints: Route points to visit, in order.

        How:
            Steps toward each waypoint until it is within tolerance. A blocked
            step, the step limit, or a termination stops the route. The safe
            flag tracks whether every visited state stayed inside the safe
            region.

        Returns:
            A ``(reached, safe)`` pair.
        """
        safe = bool(edge.safe.evaluate(states[0][:2]))
        for waypoint in waypoints[:MAX_WAYPOINTS]:
            while True:
                offset = waypoint - states[-1][:2]
                distance = float(np.linalg.norm(offset))
                if distance <= self.tolerance:
                    break
                action = self._action(env, offset, distance)
                before = states[-1][:2].copy()
                observation, _reward, terminated, truncated, info = env.step(action)
                states.append(np.asarray(observation, dtype=float))
                # The edge's safe region is the specification's; the environment may
                # also carry an unmodelled trap (info["is_safe"]), which must count too.
                safe = (
                    safe
                    and bool(edge.safe.evaluate(states[-1][:2]))
                    and bool(info.get("is_safe", True))
                )
                if info["is_success"]:
                    return True, safe
                if terminated or truncated:
                    return False, safe
                if np.allclose(before, states[-1][:2]):
                    return False, safe
        return bool(edge.target.region.evaluate(states[-1][:2])), safe

    def _action(self, env: Any, offset: np.ndarray, distance: float) -> np.ndarray:
        """Build an action that moves toward a waypoint.

        Args:
            env: The environment, used for its maximum speed.
            offset: Vector from the current position to the waypoint.
            distance: Length of ``offset``.

        How:
            Sets the heading to the offset direction and the speed to the
            smaller of the maximum speed and the remaining distance, so the
            agent does not overshoot a near waypoint.

        Returns:
            The action ``(v, theta)``.
        """
        max_speed = float(env.action_space.high[0])
        heading = float(math.atan2(offset[1], offset[0]))
        return np.array([min(max_speed, distance), heading], dtype=np.float32)

    def _plan(
        self, env: Any, policy: TabularPolicy, start: np.ndarray, goal: np.ndarray
    ) -> list[np.ndarray] | None:
        """Find a route between two positions on the grid.

        Args:
            env: The environment, used for its layout and dimensions.
            policy: The planner settings.
            start: Start position ``(x, y)``.
            goal: Goal position ``(x, y)``.

        How:
            Runs a breadth-first search over grid cells. Two neighbouring cells
            connect only when the straight move between their centres does not
            cross a solid wall. The route is the cell centres from start to
            goal, with the exact goal point appended.

        Returns:
            The route points, or ``None`` when no route exists.
        """
        size = self._grid_size(env, policy)
        start_cell = self._nearest_cell(start, policy, size)
        goal_cell = self._nearest_cell(goal, policy, size)
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
        queue: deque[tuple[int, int]] = deque([start_cell])
        while queue:
            cell = queue.popleft()
            if cell == goal_cell:
                return self._route(came_from, cell, policy, goal)
            for neighbour in self._neighbours(cell, size):
                if neighbour in came_from:
                    continue
                if env.layout.segment_is_blocked(
                    self._cell_centre(cell, policy), self._cell_centre(neighbour, policy)
                ):
                    continue
                came_from[neighbour] = cell
                queue.append(neighbour)
        return None

    def _route(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        goal_cell: tuple[int, int],
        policy: TabularPolicy,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Rebuild a route from the breadth-first search tree.

        Args:
            came_from: Cell to parent mapping built by the search.
            goal_cell: The reached goal cell.
            policy: The planner settings.
            goal: The exact goal position to append.

        How:
            Walks parent links back to the start, reverses the result, converts
            each cell to its centre, and appends the exact goal point.

        Returns:
            The route points, in visit order.
        """
        cells = []
        current: tuple[int, int] | None = goal_cell
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        return [self._cell_centre(cell, policy) for cell in cells] + [
            np.asarray(goal, dtype=float)
        ]

    def _grid_size(self, env: Any, policy: TabularPolicy) -> int:
        """Return the number of cells along one axis.

        Args:
            env: The environment, used for its world size.
            policy: The planner settings.

        How:
            Divides the world size by the cell size and rounds to an integer.

        Returns:
            The grid size along one axis.
        """
        return max(1, int(round(env.layout.world_size / policy.cell_size)))

    def _nearest_cell(
        self, point: np.ndarray, policy: TabularPolicy, size: int
    ) -> tuple[int, int]:
        """Return the grid cell that contains a position.

        Args:
            point: A position ``(x, y)``.
            policy: The planner settings.
            size: The grid size along one axis.

        How:
            Divides by the cell size and floors, then clamps into range.

        Returns:
            The ``(col, row)`` cell.
        """
        col = min(size - 1, max(0, int(point[0] // policy.cell_size)))
        row = min(size - 1, max(0, int(point[1] // policy.cell_size)))
        return col, row

    def _cell_centre(self, cell: tuple[int, int], policy: TabularPolicy) -> np.ndarray:
        """Return the centre of a grid cell.

        Args:
            cell: The ``(col, row)`` cell.
            policy: The planner settings.

        How:
            Offsets the cell index by half a cell and scales by the cell size.

        Returns:
            The centre position ``(x, y)``.
        """
        return np.array(
            [(cell[0] + 0.5) * policy.cell_size, (cell[1] + 0.5) * policy.cell_size]
        )

    def _neighbours(self, cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
        """Return the in-bounds four-connected neighbours of a cell.

        Args:
            cell: The ``(col, row)`` cell.
            size: The grid size along one axis.

        How:
            Adds each axis-aligned offset and keeps the results inside the grid.

        Returns:
            The neighbouring cells.
        """
        col, row = cell
        return [
            (col + dcol, row + drow)
            for dcol, drow in _NEIGHBOUR_OFFSETS
            if 0 <= col + dcol < size and 0 <= row + drow < size
        ]

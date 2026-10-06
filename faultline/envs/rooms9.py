"""The 9-rooms continuous environment from paper section 8.

State is 4-D: ``(x, y, theta, d)`` = position, angle to the current goal sample,
and distance to it. Action is 2-D: ``(v, theta)`` = speed and heading. The
dynamics are point-mass: ``s' = s + (v * cos theta, v * sin theta)``. A move
that crosses a solid wall keeps the agent in place.

One episode targets one reach-avoid subgoal. Set the subgoal with
:meth:`Rooms9.set_abstract_states` before each episode. The environment samples
the goal point inside the target region at reset. Reaching that point while
staying inside the safe region counts as success. This sampling is what makes a
partly unreachable target region produce a low success rate.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from faultline.envs.layout import RoomsLayout
from faultline.geometry import region_bounds
from faultline.spectrl.predicates import PLANAR_DIMS, Empty, Predicate, Universe

MAX_SPEED = 2.0
MIN_SPEED = 0.0
MAX_HEADING = math.pi
DEFAULT_MAX_STEPS = 200

# Success is landing on the goal point sampled at reset, not merely entering the
# target region. The radius absorbs the small waypoint tolerance a controller uses
# to stop, so a controller that lands on the point registers success. It stays
# below one room so a goal behind a solid wall is not reached through the wall.
SUCCESS_RADIUS = 1.0

SUCCESS_REWARD = 10.0
VIOLATION_REWARD = -10.0
STEP_PENALTY = -0.01
PROGRESS_SCALE = 0.1

# Rejection sampling needs enough attempts to hit a small region inside the
# 24x24 world; the 6x6 checkpoint regions cover about 6% of the area.
SAMPLE_ATTEMPTS = 2000

STATE_DIM = 4
ACTION_DIM = 2


class Rooms9(gym.Env):
    """A 3x3 room grid with point-mass motion and reach-avoid subgoals.

    Args:
        layout: Wall and door model. Defaults to the single-corridor 9-rooms
            layout.
        max_steps: Episode step limit, after which the episode truncates.
        unsafe_region: Optional trap region. Entering it is an absorbing unsafe
            outcome: the episode ends as a violation, so the agent can never
            leave. Defaults to no trap.

    How:
        Holds the current subgoal (source, target, and safe regions), the optional
        trap, the agent position, and a seeded RNG. ``reset`` samples a start
        inside the source region and a goal inside the target region. ``step``
        applies the point-mass dynamics, blocks wall crossings, ends the episode
        as a violation when the agent enters the trap, and scores the reach-avoid
        condition.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        layout: RoomsLayout | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        unsafe_region: Predicate | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout if layout is not None else RoomsLayout()
        self.max_steps = int(max_steps)
        world = self.layout.world_size
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -MAX_HEADING, 0.0], dtype=np.float32),
            high=np.array([world, world, MAX_HEADING, world * 1.5], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([MIN_SPEED, -MAX_HEADING], dtype=np.float32),
            high=np.array([MAX_SPEED, MAX_HEADING], dtype=np.float32),
            dtype=np.float32,
        )
        self.source_region: Predicate = Universe()
        self.target_region: Predicate = Universe()
        self.safe_region: Predicate = Universe()
        self.unsafe_region: Predicate = unsafe_region if unsafe_region is not None else Empty()
        self._rng = np.random.default_rng(0)
        self._position = np.zeros(2, dtype=float)
        self._goal = np.zeros(2, dtype=float)
        self._steps = 0
        self._violated = False

    @property
    def current_goal(self) -> np.ndarray:
        """Return the goal point sampled for the current episode.

        How:
            Returns a copy of the stored goal so callers cannot mutate the
            environment state.

        Returns:
            The goal position ``(x, y)``.
        """
        return self._goal.copy()

    def set_abstract_states(
        self, source_region: Predicate, target_region: Predicate, safe_region: Predicate
    ) -> None:
        """Set the reach-avoid subgoal for the next episode.

        Args:
            source_region: Region that the start position is sampled from.
            target_region: Region that the goal point is sampled from.
            safe_region: Predicate that must hold throughout the episode.

        How:
            Stores the three predicates. The next :meth:`reset` samples start
            and goal from the two regions.

        Returns:
            None: the subgoal is stored on the environment.
        """
        self.source_region = source_region
        self.target_region = target_region
        self.safe_region = safe_region

    def _sample_point(self, region: Predicate) -> np.ndarray:
        """Sample a world point inside a region.

        Args:
            region: The predicate that the point must satisfy.

        How:
            Uses the region's axis-aligned bounds when they are known, which makes
            sampling a small refined region reliable: the sample is drawn inside
            the bounds and rejected when it falls outside the region. An unbounded
            region falls back to rejection sampling over the whole world. The
            seeded RNG makes both paths reproducible.

        Returns:
            A position ``(x, y)`` inside the region.

        Raises:
            RuntimeError: If no sample inside the region is found within
                ``SAMPLE_ATTEMPTS``.
        """
        world = self.layout.world_size
        bounds = region_bounds(region, PLANAR_DIMS)
        if bounds is not None:
            low = np.maximum(bounds[0], 0.0)
            high = np.minimum(bounds[1], world)
            if np.all(low <= high):
                for _ in range(SAMPLE_ATTEMPTS):
                    point = self._rng.uniform(low, high)
                    if region.evaluate(point):
                        return point
        for _ in range(SAMPLE_ATTEMPTS):
            point = self._rng.uniform(0.0, world, size=2)
            if region.evaluate(point):
                return point
        raise RuntimeError("Could not sample a point inside the region")

    def _observation(self) -> np.ndarray:
        """Build the 4-D observation for the current position.

        How:
            Computes the offset to the goal, then the angle and distance to it.

        Returns:
            ``(x, y, theta, d)`` as a float32 array.
        """
        offset = self._goal - self._position
        distance = float(np.linalg.norm(offset))
        heading = float(math.atan2(offset[1], offset[0]))
        return np.array(
            [self._position[0], self._position[1], heading, distance], dtype=np.float32
        )

    # def reset(self, *, seed: int | None = None, options: dict | None = None):
    #     """Start a new episode for the current subgoal.
    #
    #     Args:
    #         seed: Optional RNG seed; reseeds the environment when given.
    #         options: Unused, kept for the gymnasium interface.
    #
    #     How:
    #         Optionally reseeds, samples the goal from the target region and the
    #         start from the source region, then clears the step count and the
    #         violation flag.
    #
    #     Returns:
    #         The first observation and an info dict with ``is_success`` false.
    #     """
    #     if seed is not None:
    #         self._rng = np.random.default_rng(seed)
    #     self._goal = self._sample_point(self.target_region)
    #     self._position = self._sample_point(self.source_region)
    #
    #     self._steps = 0
    #     self._violated = False
    #     return self._observation(), {"is_success": False}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode for the current subgoal."""

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._goal = self._sample_point(self.target_region)

        if options is not None and "start_position" in options:
            start = np.asarray(options["start_position"], dtype=float)

            if start.shape != (2,):
                raise ValueError("start_position must have shape (2,)")

            if not self.source_region.evaluate(start):
                raise ValueError(
                    "start_position must lie inside the current source region"
                )

            self._position = start.copy()
        else:
            self._position = self._sample_point(self.source_region)

        self._steps = 0
        self._violated = False

        return self._observation(), {"is_success": False}


    def step(self, action: np.ndarray):
        """Apply one action and score the reach-avoid condition.

        Args:
            action: ``(v, theta)`` speed and heading.

        How:
            Moves the agent by the point-mass rule, clips to the world and
            blocks wall crossings, then checks the sampled goal point and the
            safe region. A violation or a success ends the episode. The step
            limit truncates the episode.

        Returns:
            The observation, reward, terminated flag, truncated flag, and an
            info dict with ``is_success``.
        """
        previous_distance = float(np.linalg.norm(self._goal - self._position))
        self._position = self._advance(action)
        self._steps += 1
        distance = float(np.linalg.norm(self._goal - self._position))
        spec_safe = bool(self.safe_region.evaluate(self._position))
        in_trap = bool(self.unsafe_region.evaluate(self._position))
        safe = spec_safe and not in_trap
        reached = bool(distance <= SUCCESS_RADIUS and safe)
        if not spec_safe or in_trap:
            self._violated = True
        success = reached and not self._violated
        reward = self._reward(success, safe, previous_distance, distance)
        terminated = bool(success or self._violated)
        truncated = bool(self._steps >= self.max_steps and not terminated)
        info = {"is_success": success, "is_safe": safe}
        return self._observation(), reward, terminated, truncated, info

    def _advance(self, action: np.ndarray) -> np.ndarray:
        """Return the next position for an action.

        Args:
            action: ``(v, theta)`` speed and heading.

        How:
            Builds the point-mass step, clips the result to the world, and
            returns the old position when the step crosses a solid wall.

        Returns:
            The next position ``(x, y)``.
        """
        speed = float(np.clip(action[0], MIN_SPEED, MAX_SPEED))
        heading = float(action[1])
        delta = np.array([speed * math.cos(heading), speed * math.sin(heading)])
        candidate = self.layout.clamp(self._position + delta)
        if self.layout.segment_is_blocked(self._position, candidate):
            return self._position
        return candidate

    def _reward(
        self, success: bool, safe: bool, previous_distance: float, distance: float
    ) -> float:
        """Score one step.

        Args:
            success: Whether the episode just succeeded.
            safe: Whether the position is inside the safe region.
            previous_distance: Distance to the goal before the step.
            distance: Distance to the goal after the step.

        How:
            Returns the success reward on success, the violation penalty on an
            unsafe position, and otherwise a small step penalty plus progress
            shaping toward the goal.

        Returns:
            The step reward.
        """
        if success:
            return SUCCESS_REWARD
        if not safe:
            return VIOLATION_REWARD
        return STEP_PENALTY + PROGRESS_SCALE * (previous_distance - distance)

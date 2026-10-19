"""Draw sampled rollouts as thin paths over a room."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from faultline.refinement.trajectories import TrajectorySample
from faultline.viz.style import ROLE_COLOURS, Role

PATH_WIDTH = 0.8
PATH_ALPHA = 0.6
PATH_ZORDER = 3.2


def draw_trajectories(ax: Any, samples: Sequence[TrajectorySample]) -> set[Role]:
    """Draw each sampled rollout as a path coloured by its outcome.

    Args:
        ax: Axes already holding the room.
        samples: The sampled rollouts, possibly empty.

    How:
        Draws failed rollouts first so successful ones stay visible on top. Each path
        is one thin, semi-transparent line, so dense bundles read as texture.

    Returns:
        The roles drawn: successful and failed rollout, as present.
    """
    roles: set[Role] = set()
    for success in (False, True):
        role = Role.TRAJECTORY_OK if success else Role.TRAJECTORY_FAIL
        for sample in (s for s in samples if s.success is success and len(s.xy) > 1):
            xs, ys = zip(*sample.xy, strict=True)
            ax.plot(xs, ys, color=ROLE_COLOURS[role], linewidth=PATH_WIDTH,
                    alpha=PATH_ALPHA, zorder=PATH_ZORDER, solid_capstyle="round")
            roles.add(role)
    return roles

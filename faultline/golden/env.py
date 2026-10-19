"""The environment of a gold spec: one definition shared by gold and every faulty variant."""

from __future__ import annotations

from faultline.envs.rooms.env import RoomsEnv
from faultline.golden.spec import GoldSpec
from faultline.spectrl.predicates import Empty, Or


def make_env(gold: GoldSpec) -> RoomsEnv:
    """Build the environment for a gold spec.

    Args:
        gold: The gold spec.

    How:
        Uses the gold layout and horizon and installs every gold trap as the
        environment's absorbing ``unsafe_region``. The trap is physical, so it is
        identical whichever specification a learner is given.

    Returns:
        A fresh environment.
    """
    traps = gold.traps
    unsafe = Empty() if not traps else traps[0] if len(traps) == 1 else Or(*traps)
    return RoomsEnv(layout=gold.layout, max_steps=gold.max_steps, unsafe_region=unsafe)

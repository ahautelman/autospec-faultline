"""Registry of the fixed n-rooms worlds: frozen generator outputs, Fig. 3, Fig. 6.

The two paper layouts live in :mod:`faultline.envs.rooms.fig3` and
:mod:`faultline.envs.rooms.fig6`; this module freezes them to JSON and builds
every fixed world by name.

The random worlds are stored as JSON data, not recomputed at import time, so
results survive a generator change. A test checks that
:func:`~faultline.envs.rooms.generator.generate_world` still reproduces each
stored file; if the generator is deliberately changed, the files are
regenerated in the same commit and the diff is visible.

``WORLD_BUILDERS`` maps a registry name to a zero-argument factory returning a
fresh world, which is what the AutoSpec run's worker lookup needs: a worker
process can rebuild a world from its name alone.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from faultline.envs.rooms.fig3 import (
    FIG3_DOORS,
    FIG3_EXPECTED_OPERATORS,
    FIG3_REGIONS,
    fig3_world,
)
from faultline.envs.rooms.fig6 import (
    FIG6_DOORS,
    FIG6_REGIONS,
    FIG6_WALLS,
    fig6_world,
)
from faultline.envs.rooms.generator import generate_world
from faultline.envs.rooms.world import RoomsWorld, world_from_dict, world_to_dict

DATA_DIR = Path(__file__).with_name("data")
FROZEN_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def frozen_name(seed: int) -> str:
    """Return the registry name of a frozen generator output.

    Args:
        seed: The generator seed.

    How:
        Formats the seed into the fixed ``random_seed<k>`` registry name.

    Returns:
        The registry name.
    """
    return f"random_seed{seed}"


def world_path(name: str, directory: Path = DATA_DIR) -> Path:
    """Return the JSON path for a named fixed world.

    Args:
        name: Registry name of the world.
        directory: Directory holding the JSON files.

    How:
        Joins the directory with ``<name>.json``.

    Returns:
        The JSON file path.
    """
    return directory / f"{name}.json"


def load_world(name: str, directory: Path = DATA_DIR) -> RoomsWorld:
    """Load a fixed world from its JSON file.

    Args:
        name: Registry name such as ``random_seed0``.
        directory: Directory holding the JSON files.

    How:
        Reads the file and rebuilds it with
        :func:`~faultline.envs.rooms.world.world_from_dict`.

    Returns:
        A fresh world instance.

    Raises:
        FileNotFoundError: If the named file does not exist.
    """
    payload = json.loads(world_path(name, directory).read_text())
    return world_from_dict(payload)


def freeze_world(
    world: RoomsWorld, directory: Path = DATA_DIR, filename: str | None = None
) -> Path:
    """Write a world to the catalogue as JSON and return its path.

    Args:
        world: The world to store.
        directory: Destination directory.
        filename: File stem, defaulting to the world's name. Pass it when the
            world name (for example ``rooms100_random_0``) differs from the
            catalogue registry name (``random_seed0``).

    How:
        Uses the same serialiser as reports, with stable two-space indentation
        and a trailing newline, so regeneration diffs are readable.

    Returns:
        The path of the written file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = world_path(filename or world.name, directory)
    path.write_text(json.dumps(world_to_dict(world), indent=2) + "\n")
    return path


def freeze_random_seeds(
    seeds: tuple[int, ...] = FROZEN_SEEDS, directory: Path = DATA_DIR
) -> tuple[Path, ...]:
    """Regenerate and store the frozen random worlds.

    Args:
        seeds: Seeds to regenerate.
        directory: Destination directory.

    How:
        Calls :func:`generate_world` for each seed and writes it through
        :func:`freeze_world`. Run this only when a generator change is intended;
        the resulting file diff is the record of the change.

    Returns:
        The written paths, one per seed.
    """
    return tuple(
        freeze_world(generate_world(seed), directory, filename=frozen_name(seed))
        for seed in seeds
    )


def _file_loader(name: str) -> Callable[[], RoomsWorld]:
    """Build a zero-argument loader for one fixed JSON file.

    Args:
        name: Registry name of the JSON file.

    How:
        Closes over the name so each call reads and decodes a fresh, independent
        world.

    Returns:
        A zero-argument loader.
    """

    def load() -> RoomsWorld:
        return load_world(name)

    return load


# Built lazily through the factories, so every caller gets an independent world.
WORLD_BUILDERS: dict[str, Callable[[], RoomsWorld]] = {
    frozen_name(seed): _file_loader(frozen_name(seed)) for seed in FROZEN_SEEDS
}
WORLD_BUILDERS["fig3"] = _file_loader("fig3")
WORLD_BUILDERS["fig6"] = _file_loader("fig6")


def world_names() -> tuple[str, ...]:
    """Return the fixed-world registry names in a stable order.

    How:
        Lists the registry keys in insertion order: frozen seeds, then Fig. 3,
        then Fig. 6.

    Returns:
        The fixed world names.
    """
    return tuple(WORLD_BUILDERS)


def build_world(name: str) -> RoomsWorld:
    """Build a fixed world by registry name.

    Args:
        name: A fixed world name from :func:`world_names`.

    How:
        Looks the name up in the registry and calls its loader.

    Returns:
        A fresh world.

    Raises:
        KeyError: If the name is unknown.
    """
    try:
        factory = WORLD_BUILDERS[name]
    except KeyError as missing:
        raise KeyError(f"Unknown fixed world {name!r}; known: {sorted(WORLD_BUILDERS)}") from missing
    return factory()


__all__ = [
    "DATA_DIR",
    "FIG3_DOORS",
    "FIG3_EXPECTED_OPERATORS",
    "FIG3_REGIONS",
    "FIG6_DOORS",
    "FIG6_REGIONS",
    "FIG6_WALLS",
    "FROZEN_SEEDS",
    "WORLD_BUILDERS",
    "build_world",
    "fig3_world",
    "fig6_world",
    "freeze_random_seeds",
    "freeze_world",
    "frozen_name",
    "load_world",
    "world_names",
    "world_path",
]

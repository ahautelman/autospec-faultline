"""Figure tests for the fixed n-rooms worlds.

Each fixed world is drawn, saved, and its wall segments are compared against the
segments implied by its door map. The comparison is structural (compute the
expected pieces, do not compare pixels), so a wrong door still fails even when
the image looks plausible.

Run with ``pytest --save-figures`` to keep a copy of every layout under
``reports/rooms100/layouts/`` for manual review.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.envs.rooms.catalogue import build_world, world_names  # noqa: E402
from faultline.envs.rooms.verify import verify_heuristic  # noqa: E402
from faultline.viz.room import draw_world  # noqa: E402
from faultline.viz.style import pyplot, save_figure  # noqa: E402

REPORT_DIR = Path("reports/rooms100/layouts")
MIN_IMAGE_SIDE = 800


def _wall_segments(layout):
    """Return the wall pieces a layout implies, computed independently.

    How:
        Walks the outer boundary and every internal grid line. A missing door is
        one solid piece; a door splits the run around its open interval. This is
        deliberately a second implementation of the drawing rule, not a call into
        the drawing code, so the test can catch a wrong wall.
    """
    size = layout.world_size
    segments = [
        ((0.0, 0.0), (size, 0.0)),
        ((size, 0.0), (size, size)),
        ((size, size), (0.0, size)),
        ((0.0, size), (0.0, 0.0)),
    ]
    for line in range(1, layout.rooms_per_side):
        position = line * layout.room_size
        for room in range(layout.rooms_per_side):
            low, high = room * layout.room_size, (room + 1) * layout.room_size
            vertical = layout.doors.get(("v", line - 1, room))
            horizontal = layout.doors.get(("h", room, line - 1))
            segments += [
                ((position, start), (position, end))
                for start, end in _solid_pieces(low, high, vertical)
            ]
            segments += [
                ((start, position), (end, position))
                for start, end in _solid_pieces(low, high, horizontal)
            ]
    return segments


def _solid_pieces(low, high, door):
    """Split one wall run around its door interval, if any."""
    if door is None:
        return [(low, high)]
    return [piece for piece in ((low, door[0]), (door[1], high)) if piece[1] > piece[0]]


def _normalised(segments):
    """Return segments as sorted, rounded, endpoint-ordered tuples."""
    normalised = []
    for first, second in segments:
        low = (round(float(first[0]), 6), round(float(first[1]), 6))
        high = (round(float(second[0]), 6), round(float(second[1]), 6))
        normalised.append(tuple(sorted((low, high))))
    return sorted(normalised)


@pytest.mark.parametrize("name", world_names())
def test_fixed_world_draws_its_walls_and_a_caption(name, tmp_path, request):
    """The drawn segments equal the door map's solid pieces."""
    world = build_world(name)
    report = verify_heuristic(world)
    figure, ax = pyplot().subplots(figsize=(4.5, 4.5))

    roles = draw_world(ax, world, report)

    drawn = _normalised(
        (tuple(segment[0]), tuple(segment[1]))
        for segment in ax.collections[0].get_segments()
    )
    assert drawn == _normalised(_wall_segments(world.layout))
    assert {"start", "goal"} <= {role.value for role in roles}
    assert ax.get_title().startswith(world.name)
    assert f"{len(world.layout.doors)} doors" in ax.get_title()
    assert ("PASS" if report.solvable else "FAIL") in ax.get_title()

    if request.config.getoption("--save-figures"):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        figure.savefig(REPORT_DIR / f"{name}.png")

    paths = save_figure(figure, tmp_path / name, ("png",))
    image = pyplot().imread(paths[0])
    assert paths[0].stat().st_size > 0
    assert min(image.shape[:2]) >= MIN_IMAGE_SIDE


def test_caption_reports_rejection_count_and_route_count(tmp_path):
    """A generated world's caption carries its attempts and feasible routes."""
    world = build_world("random_seed0")
    report = verify_heuristic(world)
    figure, ax = pyplot().subplots()

    draw_world(ax, world, report)

    assert f"{world.meta['attempts']} attempts" in ax.get_title()
    assert f"{len(report.routes)} routes" in ax.get_title()
    save_figure(figure, tmp_path / "caption-check", ("png",))


def test_compact_panel_drops_the_axes_and_uses_short_region_tags():
    """Compact mode keeps the room grid but removes ticks and long labels."""
    world = build_world("fig3")
    _figure, ax = pyplot().subplots()

    draw_world(ax, world, verify_heuristic(world), compact=True)

    labels = {text.get_text() for text in ax.texts}
    assert labels == {"S", "1", "2", "3", "4", "5", "G"}
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
    assert ax.get_title() == "fig3 · PASS"
    assert len(ax.collections[0].get_segments()) > 0

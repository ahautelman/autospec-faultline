"""Random-layout contact sheets and generator contract test.

Generates a batch of seeds, verifies every world, and draws the worlds as a few
readable contact sheets: six compact panels per sheet, each titled with its seed
and verdict. The per-world numbers (attempts, doors, routes, best route) go into
a markdown summary next to the sheets, not into the picture. A hand-built
door-less world is included so the failing branch of the display is exercised.

The sheets are written to pytest's temporary directory. Set ``--save-figures``
to keep a copy under ``reports/rooms100/layouts/``, and ``--figures-count N`` to
draw only the first ``N`` seeds (the contract is asserted on all of them).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.envs.rooms.catalogue import build_world  # noqa: E402
from faultline.envs.rooms.generator import generate_world  # noqa: E402
from faultline.envs.rooms.layout import RoomsLayout  # noqa: E402
from faultline.envs.rooms.verify import verify_heuristic  # noqa: E402
from faultline.envs.rooms.world import RoomsWorld  # noqa: E402
from faultline.viz.room import compact_caption, draw_world  # noqa: E402
from faultline.viz.style import pyplot, save_figure  # noqa: E402

CONTACT_SHEET_SEEDS = 20
PANELS_PER_SHEET = 6
SHEET_COLUMNS = 3
PANEL_INCHES = 4.5
TITLE_ROOM_INCHES = 0.9
MIN_IMAGE_SIDE = 800
REPORT_DIR = Path("reports/rooms100/layouts")
TABLE_HEADERS = ("world", "verdict", "attempts", "doors", "routes", "best route")
LEGEND = "S start · G goal · 1-5 intermediate regions (M1-M5)"


def _doorless_world() -> RoomsWorld:
    """Return a hand-built unsolvable world for the failing display branch.

    How:
        Keeps the Fig. 3 regions and topology but replaces the layout with one
        whose door map is empty, so every internal wall is solid. The generator
        would reject this world, which is exactly why it is built by hand.
    """
    base = build_world("fig3")
    return RoomsWorld(
        name="forced_fail",
        layout=RoomsLayout({}, base.layout.room_size, base.layout.rooms_per_side),
        regions=base.regions,
        topology=base.topology,
        meta={"attempts": 0},
    )


def _summary_row(world: RoomsWorld, report) -> list[str]:
    """Return one table row: the numbers a panel title leaves out."""
    best = " > ".join(report.best_route) if report.best_route else "-"
    return [
        world.name,
        "PASS" if report.solvable else "FAIL",
        str(world.meta.get("attempts", "-")),
        str(len(world.layout.doors)),
        str(len(report.routes)),
        best,
    ]


def _markdown_table(rows: list[list[str]]) -> str:
    """Render the summary rows as a markdown table."""
    lines = [
        "| " + " | ".join(TABLE_HEADERS) + " |",
        "| " + " | ".join("---" for _ in TABLE_HEADERS) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    return "\n".join(lines) + "\n"


def _draw_sheet(worlds, reports, stem: Path) -> Path:
    """Draw up to ``PANELS_PER_SHEET`` worlds as one compact contact sheet.

    Returns:
        The written PNG path.
    """
    plt = pyplot()
    rows = ceil(len(worlds) / SHEET_COLUMNS)
    figure, axes = plt.subplots(
        rows,
        SHEET_COLUMNS,
        figsize=(SHEET_COLUMNS * PANEL_INCHES, rows * PANEL_INCHES + TITLE_ROOM_INCHES),
        squeeze=False,
    )
    flat = list(axes.flat)
    for ax, world, report in zip(flat, worlds, reports, strict=False):
        draw_world(ax, world, report, compact=True)
    for ax in flat[len(worlds) :]:
        ax.axis("off")
    figure.suptitle(LEGEND, fontsize=12)
    figure.tight_layout()
    return save_figure(figure, stem, ("png",))[0]


def _sheets(worlds, reports, directory: Path) -> list[Path]:
    """Draw every world, ``PANELS_PER_SHEET`` per sheet, and return the paths."""
    return [
        _draw_sheet(
            worlds[start : start + PANELS_PER_SHEET],
            reports[start : start + PANELS_PER_SHEET],
            directory / f"random_worlds_{start // PANELS_PER_SHEET + 1}",
        )
        for start in range(0, len(worlds), PANELS_PER_SHEET)
    ]


def test_random_worlds_pass_and_the_sheets_record_verdicts(tmp_path, request):
    """Every generated world passes, and the forced-fail panel is marked FAIL."""
    worlds = [generate_world(seed) for seed in range(CONTACT_SHEET_SEEDS)]
    reports = [verify_heuristic(world) for world in worlds]

    assert all(report.solvable for report in reports)
    assert all(world.meta["attempts"] >= 1 for world in worlds)

    forced = _doorless_world()
    forced_report = verify_heuristic(forced)
    assert forced_report.solvable is False
    assert forced_report.routes == ()

    shown = request.config.getoption("--figures-count") or CONTACT_SHEET_SEEDS
    drawn_worlds = [*worlds[:shown], forced]
    drawn_reports = [*reports[:shown], forced_report]
    paths = _sheets(drawn_worlds, drawn_reports, tmp_path)

    assert len(paths) == ceil(len(drawn_worlds) / PANELS_PER_SHEET)
    for path in paths:
        image = pyplot().imread(path)
        assert path.stat().st_size > 0
        assert min(image.shape[:2]) >= MIN_IMAGE_SIDE

    rows = [_summary_row(world, report) for world, report in zip(worlds, reports, strict=True)]
    rows.append(_summary_row(forced, forced_report))
    assert [row[1] for row in rows[:-1]] == ["PASS"] * CONTACT_SHEET_SEEDS
    assert rows[-1][1] == "FAIL"
    assert all(row[2].isdigit() for row in rows[:-1])
    assert all(row[3].isdigit() and row[4].isdigit() for row in rows)
    table = _markdown_table(rows)
    (tmp_path / "random_worlds.md").write_text(table)
    assert table.count("\n") == len(rows) + 2

    if request.config.getoption("--save-figures"):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        for path in paths:
            (REPORT_DIR / path.name).write_bytes(path.read_bytes())
        (REPORT_DIR / "random_worlds.md").write_text(table)


def test_compact_titles_are_short_and_carry_seed_and_verdict():
    """A compact panel title is ``seed N · VERDICT`` and nothing else."""
    passing = generate_world(7)
    failing = _doorless_world()

    assert compact_caption(passing, verify_heuristic(passing)) == "seed 7 · PASS"
    assert compact_caption(failing, verify_heuristic(failing)) == "forced_fail · FAIL"
    assert compact_caption(passing) == "seed 7"

"""The geometric overlay figure for the 9-rooms report.

One figure per scenario shows the room grid, the bottleneck edge's source and
target regions, the sampled states the refinement operator consumed, and the
region or edge it produced. The geometry is rebuilt from the scenario's graph and
the stored overlay points, so no training is needed to redraw it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from faultline.experiments.drawing import (
    RegionStyle,
    configure_axis,
    draw_arrow,
    draw_half_plane,
    draw_layout,
    pyplot,
    region_centre,
    scatter_points,
    shade_region,
)
from faultline.experiments.overlays import refined_region
from faultline.experiments.pipeline import source_edge
from faultline.experiments.records import ScenarioReport
from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.learners import CannotRefineError
from faultline.refinement.operators import or_refine
from faultline.scenarios import Scenario

SOURCE_STYLE = RegionStyle("#4c72b0", 0.18, "source region")
TARGET_STYLE = RegionStyle("#dd8452", 0.18, "target region")
REFINED_STYLE = RegionStyle("#55a868", 0.45, "refined region")
SIBLING_STYLE = RegionStyle("#55a868", 0.35, "reused sibling region")
SAMPLE_STYLE = RegionStyle("#c44e52", 0.85, "operator samples")
FAILURE_STYLE = RegionStyle("#8172b3", 0.55, "failed starts")
ADDED_EDGE_LABEL = "added OrRefine edge"
GRID_COLUMNS = 3


def draw_overlay(ax: Any, report: ScenarioReport, scenario: Scenario) -> None:
    """Draw one scenario's geometric overlay on an axes.

    Args:
        ax: Target axes.
        report: The scenario's single-seed report.
        scenario: The scenario, supplying the layout and graph.

    How:
        Draws the grid, the source and target regions, then either the refined
        region with the operator's samples (Reach, Add, Past) or the structural
        edge OrRefine added. Labels the axes with the scenario and operator.

    Returns:
        None: the axes are drawn in place.
    """
    graph = scenario.build_graph()
    edge = source_edge(graph)
    draw_layout(ax, scenario.layout)
    shade_region(ax, edge.source.region, scenario.layout, SOURCE_STYLE)
    shade_region(ax, edge.target.region, scenario.layout, TARGET_STYLE)
    if report.overlay.kind == "edge":
        _draw_structural_edge(ax, graph, edge, scenario)
    else:
        _draw_refined_geometry(ax, report, edge, scenario)
    configure_axis(ax, scenario.layout, f"{scenario.name} ({report.operator})")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.9)


def write_overlay_figures(
    reports: Mapping[str, ScenarioReport],
    scenarios: Mapping[str, Scenario],
    output: Path,
) -> None:
    """Write one overlay figure per scenario plus a combined grid.

    Args:
        reports: Single-seed reports keyed by scenario name.
        scenarios: Scenarios keyed by scenario name.
        output: Directory the figures are written to.

    How:
        Writes ``overlay_<scenario>.png`` for each scenario and a combined
        ``overlays.png`` grid, then closes each figure to release memory.

    Returns:
        None: the figure files are written to ``output``.
    """
    plt = pyplot()
    output.mkdir(parents=True, exist_ok=True)
    names = list(reports)
    for name in names:
        figure, ax = plt.subplots(figsize=(4.8, 4.8))
        draw_overlay(ax, reports[name], scenarios[name])
        figure.tight_layout()
        figure.savefig(output / f"overlay_{name}.png", dpi=150)
        plt.close(figure)
    _write_grid(reports, scenarios, output / "overlays.png")


def _draw_refined_geometry(ax: Any, report: ScenarioReport, edge: Edge, scenario: Scenario) -> None:
    """Shade the refined region and scatter the operator's samples.

    Args:
        ax: Target axes.
        report: The scenario's single-seed report.
        edge: The bottleneck edge that was refined.
        scenario: The scenario, supplying the layout.

    How:
        Shades the region rebuilt from the overlay inputs, scatters the operator's
        sampled states, and for a half-plane split adds the boundary and the
        failed starts.

    Returns:
        None: the axes are drawn in place.
    """
    region = refined_region(report.operator, edge, report.overlay)
    if region is not None:
        shade_region(ax, region, scenario.layout, REFINED_STYLE)
    scatter_points(ax, report.overlay.points, SAMPLE_STYLE)
    if report.overlay.kind == "half_plane":
        draw_half_plane(ax, report.overlay, scenario.layout)
        scatter_points(ax, report.overlay.counter_points, FAILURE_STYLE)


def _draw_structural_edge(ax: Any, graph: AbstractGraph, edge: Edge, scenario: Scenario) -> None:
    """Draw the vertex regions and the edge OrRefine added.

    Args:
        ax: Target axes.
        graph: The graph the operator runs on.
        edge: The blocked edge being routed around.
        scenario: The scenario, supplying the layout.

    How:
        Runs OrRefine, finds the edge it added, shades the reused sibling region,
        and draws an arrow from the source centre to the sibling centre. A refusal
        leaves the graph undrawn beyond the base regions.

    Returns:
        None: the axes are drawn in place.
    """
    try:
        refined = or_refine(graph, edge)
    except CannotRefineError:
        return
    added = next(
        (
            candidate
            for candidate in refined.edges
            if candidate.source.name == edge.source.name and candidate.target.name != edge.target.name
        ),
        None,
    )
    if added is None:
        return
    shade_region(ax, added.target.region, scenario.layout, SIBLING_STYLE)
    start = region_centre(added.source.region, scenario.layout)
    stop = region_centre(added.target.region, scenario.layout)
    draw_arrow(ax, start, stop, ADDED_EDGE_LABEL)


def _write_grid(
    reports: Mapping[str, ScenarioReport],
    scenarios: Mapping[str, Scenario],
    path: Path,
) -> None:
    """Write the combined overlay grid.

    Args:
        reports: Single-seed reports keyed by scenario name.
        scenarios: Scenarios keyed by scenario name.
        path: Destination image path.

    How:
        Lays the overlays out in a fixed-column grid, blanks any unused panels,
        and saves the figure.

    Returns:
        None: the figure is written to ``path``.
    """
    plt = pyplot()
    names = list(reports)
    rows = math.ceil(len(names) / GRID_COLUMNS)
    figure, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(4.8 * GRID_COLUMNS, 4.8 * rows))
    panels = list(np.ravel(axes))
    for ax, name in zip(panels, names, strict=False):
        draw_overlay(ax, reports[name], scenarios[name])
    for ax in panels[len(names) :]:
        ax.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

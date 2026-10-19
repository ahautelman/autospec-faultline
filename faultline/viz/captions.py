"""Titles and caption text generated from what a run did.

Every sentence is built from the recorded steps (the operator, the edge, the edit's
shape, the success before and after), so a caption always matches the figure it sits
under and any run, including a new operator, is described without hand-written text.
"""

from __future__ import annotations

from collections.abc import Sequence

from faultline.geometry import predicate_from_dict
from faultline.refinement.loop import RefinementStep
from faultline.refinement.trace import StepTrace
from faultline.viz.polygons import polygon_area, region_polygons

REFERENCE_KSTEPS = 80

OPERATOR_NAMES = {
    "seq": "SeqRefine",
    "add": "AddRefine",
    "past": "PastRefine",
    "or": "OrRefine",
}
NO_REFINEMENT = "no refinement"
ARROW = "→"


def operator_name(operator: str) -> str:
    """Return an operator's display name.

    Args:
        operator: The short operator key, such as ``"seq"``.

    How:
        Looks the key up in the display table and falls back to the key itself, so
        an operator added later still gets a caption.

    Returns:
        The display name.
    """
    return OPERATOR_NAMES.get(operator, operator)


def run_title(scenario: str, steps: Sequence[RefinementStep]) -> str:
    """Build a figure title naming the scenario and the operators applied.

    Args:
        scenario: The scenario name, with underscores.
        steps: The run's refinement steps.

    How:
        Joins the committed operators' display names in scan order, or says no
        refinement was applied.

    Returns:
        A title such as ``"blocked two path: OrRefine"``.
    """
    applied = [operator_name(step.operator) for step in steps if step.operator]
    return f"{scenario.replace('_', ' ')}: {' + '.join(applied) if applied else NO_REFINEMENT}"


def step_caption(index: int, step: RefinementStep, world: float) -> str:
    """Describe one refinement step in a sentence or two.

    Args:
        index: Zero-based position of the step in the run.
        step: The step to describe.
        world: Side length of the square world, used to measure region areas.

    How:
        A committed step lists what its trace changed, then the edge's success before
        and after; with no recorded trace it gives only the operator and the success.
        A step where no operator cleared the threshold says so.

    Returns:
        The caption text, starting with ``Step <n>``.
    """
    edge = f"{step.source} {ARROW} {step.target}"
    if step.operator is not None and step.trace is None:
        return (
            f"Step {index + 1}: {operator_name(step.operator)} on {edge}. "
            f"Success {step.before:.2f} {ARROW} {step.after:.2f}."
        )
    if step.operator is None:
        return (
            f"Step {index + 1}: no operator cleared the threshold on {edge} "
            f"(success {step.before:.2f}); the edge was left unchanged."
        )
    edits = "; ".join(describe_edits(step.trace, world)) or "no visible change"
    return (
        f"Step {index + 1}: {operator_name(step.operator)} on {edge}: {edits}. "
        f"Success {step.before:.2f} {ARROW} {step.after:.2f}."
    )


def run_caption(steps: Sequence[RefinementStep], world: float) -> str:
    """Describe a whole run, one sentence group per step.

    Args:
        steps: The run's refinement steps.
        world: Side length of the square world.

    How:
        Joins the step captions, or states that nothing needed refining.

    Returns:
        The caption text.
    """
    if not steps:
        return "No edge fell below the success threshold, so no refinement was applied."
    return " ".join(step_caption(index, step, world) for index, step in enumerate(steps))


def describe_edits(trace: StepTrace, world: float) -> list[str]:
    """List, in words, what a step's trace changed.

    Args:
        trace: The step's trace.
        world: Side length of the square world.

    How:
        One phrase per changed region, safe region, new region, new edge and removed
        edge. Areas are reported only for regions that are a single convex polygon.

    Returns:
        The phrases, possibly empty.
    """
    phrases = [
        f"region {change.name} shrunk{_area_change(change.before, change.after, world)}"
        for change in trace.vertex_changes
    ]
    phrases += [
        f"unsafe area removed from the safe region of {c.source} {ARROW} {c.target}"
        for c in trace.safe_changes
    ]
    phrases += [f"new region {vertex.name}" for vertex in trace.added_vertices]
    phrases += [f"edge {e.source} {ARROW} {e.target} added" for e in trace.added_edges]
    phrases += [f"edge {source} {ARROW} {target} replaced" for source, target in trace.removed_edges]
    return phrases


def _area_change(before: dict, after: dict, world: float) -> str:
    """Return `` (area A -> B)`` when both regions are convex, else an empty string."""
    areas = [_convex_area(region, world) for region in (before, after)]
    if None in areas:
        return ""
    return f" (area {areas[0]:.0f} {ARROW} {areas[1]:.0f})"


def _convex_area(region: dict, world: float) -> float | None:
    """Return a region's area if it is a single convex polygon, else ``None``."""
    polygons = region_polygons(predicate_from_dict(region), world)
    if polygons is None or len(polygons) != 1:
        return None
    return polygon_area(polygons[0])


def trajectory_sentence(has_after_panel: bool) -> str:
    """Explain the sampled rollout lines drawn in an overlay.

    Args:
        has_after_panel: Whether the figure has separate before and after panels.

    How:
        Names the two outcome colours and which panel shows which graph's best path.

    Returns:
        One sentence for the end of a caption.
    """
    where = "before (left) and after (right) refinement" if has_after_panel else "on the best path"
    return f"Thin lines are sampled rollouts of the edges on the best path, {where}."


def group_title(scenario: str, runs_steps: Sequence[Sequence[RefinementStep]]) -> str:
    """Build a title for several seeds of one scenario.

    Args:
        scenario: The scenario name, with underscores.
        runs_steps: Each seed's refinement steps.

    How:
        If every seed applied the same operators the title is the usual run title;
        if they differ it says so rather than naming one seed's choice.

    Returns:
        The title.
    """
    titles = {run_title(scenario, steps) for steps in runs_steps}
    if len(titles) == 1:
        return titles.pop()
    return f"{scenario.replace('_', ' ')}: operators vary by seed"


def curves_caption(seed_count: int, has_threshold: bool, has_reference: bool) -> str:
    """Explain a learning-curve figure.

    Args:
        seed_count: Number of seeds plotted.
        has_threshold: Whether a success threshold line is drawn.
        has_reference: Whether the paper's reference markers are drawn.

    How:
        States what the axes show, then adds a clause for each optional element that
        is actually on the figure.

    Returns:
        The caption.
    """
    parts = ["Success probability of the best start-to-goal path against training steps per edge."]
    if seed_count > 1:
        parts.append(
            f"Thick lines are the mean over {seed_count} seeds, faint lines individual seeds, "
            "and shading the minimum to maximum."
        )
    if has_threshold:
        parts.append("The dotted line is the success threshold the loop refines below.")
    if has_reference:
        parts.append(f"Diamonds mark the paper's reported value at {REFERENCE_KSTEPS}k steps.")
    return " ".join(parts)


def seeds_caption(seed_count: int, checkpoints: str) -> str:
    """Explain a per-seed spread figure.

    Args:
        seed_count: Number of seeds plotted.
        checkpoints: Plain-language list of the checkpoints shown, for example
            ``"80k and 200k steps"``.

    How:
        Says each dot is one seed, that grey lines pair a seed's before and after, and
        what the bars show.

    Returns:
        The caption.
    """
    return (
        f"Success probability before and after refinement at {checkpoints}. Each dot is one "
        f"of {seed_count} seeds and grey lines join a seed's two values; the bar marks the "
        "mean and the whisker one standard deviation, limited to the range 0 to 1."
    )


def reference_caption(seed_count: int) -> str:
    """Explain the paper-comparison bar chart.

    Args:
        seed_count: Number of seeds averaged per scenario.

    How:
        Says hatched bars are the paper's numbers and solid bars are this run's mean at
        the paper's checkpoint.

    Returns:
        The caption.
    """
    mean = f"mean over {seed_count} seeds" if seed_count > 1 else "single seed"
    whisker = ", whiskers one standard deviation limited to the range 0 to 1" if seed_count > 1 else ""
    return (
        f"Hatched bars are the paper's reported success probabilities; solid bars are this "
        f"run ({mean}) at {REFERENCE_KSTEPS}k steps{whisker}."
    )

"""Tests for the generated titles and captions.

Captions are built only from recorded steps, so each test builds a step by hand and
checks the sentence a reader would see.
"""

from __future__ import annotations

from faultline.geometry import convex_hull, predicate_to_dict
from faultline.graph.snapshot import EdgeSnapshot, VertexSnapshot
from faultline.refinement.loop import RefinementStep
from faultline.refinement.trace import SafeChange, StepTrace, VertexChange
from faultline.refinement.trajectories import StepSamples
from faultline.spectrl.predicates import Box, Or
from faultline.viz.captions import describe_edits, run_caption, run_title, step_caption

WORLD = 24.0
EMPTY_SAMPLES = StepSamples((), (), ())


def _trace(**edits) -> StepTrace:
    fields = dict(vertex_changes=(), safe_changes=(), added_vertices=(), added_edges=(),
                  removed_edges=(), samples=EMPTY_SAMPLES)
    return StepTrace(**{**fields, **edits})


def _shrink(before, after) -> VertexChange:
    return VertexChange("mid", predicate_to_dict(before), predicate_to_dict(after))


def _step(operator, trace, before=0.31, after=0.87) -> RefinementStep:
    return RefinementStep("start", "mid", (operator,), operator, before, after, trace)


def test_title_names_the_scenario_and_the_operators_in_order():
    steps = [_step("seq", None), _step("or", None)]

    assert run_title("blocked_two_path", steps) == "blocked two path: SeqRefine + OrRefine"


def test_title_says_so_when_nothing_was_refined():
    skipped = RefinementStep("a", "b", ("seq",), None, 0.4, 0.4)

    assert run_title("narrow_door", [skipped]) == "narrow door: no refinement"
    assert run_title("narrow_door", []) == "narrow door: no refinement"


def test_committed_step_caption_gives_operator_edge_edit_and_success():
    trace = _trace(vertex_changes=(_shrink(Box((0.0, 0.0), (6.0, 6.0)), Box((1.0, 1.0), (4.0, 4.0))),))

    caption = step_caption(0, _step("seq", trace), WORLD)

    assert caption == (
        "Step 1: SeqRefine on start → mid: region mid shrunk (area 36 → 9). "
        "Success 0.31 → 0.87."
    )


def test_uncommitted_step_caption_says_the_edge_was_left_unchanged():
    skipped = RefinementStep("start", "mid", ("seq", "add"), None, 0.4, 0.4)

    caption = step_caption(1, skipped, WORLD)

    assert caption.startswith("Step 2: no operator cleared the threshold on start → mid")
    assert "left unchanged" in caption


def test_unknown_operator_is_named_by_its_key():
    caption = step_caption(0, _step("splice", _trace()), WORLD)

    assert "splice on start" in caption and "no visible change" in caption


def test_run_caption_with_no_steps_says_nothing_needed_refining():
    assert "no refinement was applied" in run_caption([], WORLD)


def test_run_caption_numbers_each_step():
    caption = run_caption([_step("seq", _trace()), _step("or", _trace())], WORLD)

    assert "Step 1:" in caption and "Step 2:" in caption


def test_structural_edits_are_described_in_words():
    trace = _trace(
        added_vertices=(VertexSnapshot("add1", predicate_to_dict(Box((0.0, 0.0), (1.0, 1.0)))),),
        added_edges=(EdgeSnapshot("start", "add1", predicate_to_dict(Box((0.0, 0.0), (9.0, 9.0)))),),
        removed_edges=(("start", "mid"),),
        safe_changes=(SafeChange("start", "mid", predicate_to_dict(Box((0.0, 0.0), (9.0, 9.0))),
                                 predicate_to_dict(Box((0.0, 0.0), (5.0, 5.0)))),),
    )

    phrases = describe_edits(trace, WORLD)

    assert "new region add1" in phrases
    assert "edge start → add1 added" in phrases
    assert "edge start → mid replaced" in phrases
    assert any(p.startswith("unsafe area removed from the safe region of start") for p in phrases)


def test_area_is_omitted_for_a_region_that_is_not_one_convex_polygon():
    two_pieces = Or(Box((0.0, 0.0), (1.0, 1.0)), Box((5.0, 5.0), (6.0, 6.0)))
    hull = convex_hull([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])

    (phrase,) = describe_edits(_trace(vertex_changes=(_shrink(two_pieces, hull),)), WORLD)

    assert phrase == "region mid shrunk"

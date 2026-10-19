"""Tests for the candidate routes the Algorithm 1 commit gate scores.

OrRefine adds ``source -> sibling`` links only for siblings that do not close a
cycle, and may add a link parallel to an existing edge. These tests pin that the
OrRefine candidate scores exactly the links the operator added.
"""

from __future__ import annotations

from faultline.refinement import AVOID_TAIL_LENGTH
from faultline.refinement.candidates import or_candidate
from tests.refinement_support import (
    SAFE,
    cyclic_sibling_graph,
    estimate,
    existing_link_graph,
    rollout,
)

# OrRefine is purely structural, so its estimate is never read.
UNUSED_ESTIMATE = estimate((rollout([[1.5, 1.5]], reached=False, safe=True),))


def test_or_candidate_routes_only_through_siblings_that_avoid_a_cycle():
    graph, blocked = cyclic_sibling_graph()

    candidate = or_candidate(graph, blocked, UNUSED_ESTIMATE, AVOID_TAIL_LENGTH)

    assert [[(e.source.name, e.target.name) for e in route] for route in candidate.routes] == [
        [("mid", "alt"), ("alt", "goal")]
    ]


def test_or_candidate_scores_the_link_or_refine_added_not_an_older_parallel_one():
    graph, blocked = existing_link_graph()

    candidate = or_candidate(graph, blocked, UNUSED_ESTIMATE, AVOID_TAIL_LENGTH)

    (link, _sibling), = candidate.routes
    assert link.safe == SAFE

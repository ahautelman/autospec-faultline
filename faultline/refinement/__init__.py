"""Refinement operators and the Algorithm 1 loop (paper sections 3-4).

Re-exports the four graph operators, the two region sub-procedures they build on,
the candidate wrappers the commit gate scores, and the loop itself. The loop
calls the operators in the paper's fixed order: SeqRefine, then AddRefine, then
PastRefine, then OrRefine.
"""

from faultline.refinement.candidates import (
    Candidate,
    add_candidate,
    or_candidate,
    past_candidate,
    seq_candidate,
)
from faultline.refinement.loop import (
    AutoSpecConfig,
    AutoSpecRun,
    EdgeSuccess,
    RefinementStep,
    run_autospec,
)
from faultline.refinement.operators import (
    AVOID_TAIL_LENGTH,
    add_refine,
    avoid_refine,
    or_refine,
    past_refine,
    reach_refine,
    seq_refine,
)

__all__ = [
    "AVOID_TAIL_LENGTH",
    "AutoSpecConfig",
    "AutoSpecRun",
    "Candidate",
    "EdgeSuccess",
    "RefinementStep",
    "add_candidate",
    "add_refine",
    "avoid_refine",
    "or_candidate",
    "or_refine",
    "past_candidate",
    "past_refine",
    "reach_refine",
    "run_autospec",
    "seq_candidate",
    "seq_refine",
]

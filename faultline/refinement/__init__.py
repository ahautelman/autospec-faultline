"""Refinement operators for the abstract graph (paper section 4).

Re-exports the four graph operators and the two region sub-procedures they build
on. The Algorithm 1 loop calls the operators in the paper's fixed order:
SeqRefine, then AddRefine, then PastRefine, then OrRefine.
"""

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
    "add_refine",
    "avoid_refine",
    "or_refine",
    "past_refine",
    "reach_refine",
    "seq_refine",
]

"""FaultLine fault injection: parametric mutations of a gold spec, the validity gate and the screening gates.

``mutations`` turns a certified gold into a faulty spec with a known ground-truth repair
(``ground_truth``); ``validity`` checks the fault only coarsens gold; ``gates`` screens out
faults the learner cannot notice (effectiveness) or the operators cannot act on
(observability).
"""

from faultline.faults.ground_truth import (
    EXPECTED_OPERATORS,
    FaultKind,
    GroundTruthRepair,
    Locus,
    MutationNotApplicable,
    MutationParams,
)
from faultline.faults.mutations import mutate
from faultline.faults.validity import check_validity

__all__ = [
    "EXPECTED_OPERATORS",
    "FaultKind",
    "GroundTruthRepair",
    "Locus",
    "MutationNotApplicable",
    "MutationParams",
    "check_validity",
    "mutate",
]

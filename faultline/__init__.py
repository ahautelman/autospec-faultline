"""FaultLine: a fault-injection recovery benchmark for AutoSpec.

This package is a from-scratch, tested implementation of the AutoSpec
specification-refinement pipeline. It compiles SpectRL specifications into
abstract graphs, runs the Algorithm 1 refinement loop, and applies the four
sound refinement operators. The original authors' code under
``autospec-original-code/`` is kept as a read-only behavioural reference and is
never imported.

Module to paper-section map (see ``AutoSpec_research_paper.md``):

    spectrl/    -> section 2.2  (SpectRL grammar, predicates, AST)
    graph/      -> section 2.3  (abstract graph and the AST-to-graph compiler)
    geometry/   -> section 4    (convex hulls and half-plane partitioning)
    refinement/ -> sections 3-4 (Algorithm 1 loop and the four operators)
    envs/       -> section 8    (the 9-rooms environment)
    learners/   -> section 6    (pluggable spec-guided RL backends)
    metrics/    -> README       (FaultLine IoU / wIoU / GED for RQ1)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

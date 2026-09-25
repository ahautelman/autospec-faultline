"""Reference semantics for SpectRL specifications.

Implements the ``zeta models phi`` relation from paper section 2.2 over finite
trajectories. This is the independent oracle the abstract-graph compiler is
checked against: ``models_spec(phi, zeta)`` must agree with
``models_trajectory(compile_spec(phi), zeta)`` for every trajectory.

The relation is computed by :func:`_completion_indices`, which returns the set
of trajectory indices at which a spec can finish when it begins at a given
index. Consecutive steps of a sequence share their boundary index, mirroring
the bridge edges the compiler inserts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from faultline.spectrl.ast import Achieve, Choice, Ensuring, Seq, Spec

State = np.ndarray


def models_spec(spec: Spec, states: Sequence[State]) -> bool:
    """Return whether a finite trajectory satisfies a specification.

    Args:
        spec: The specification to evaluate.
        states: The trajectory as a sequence of state vectors, in time order.

    How:
        Asks for the indices at which ``spec`` can complete starting from the
        trajectory's first state; the trajectory satisfies the spec when at
        least one such index exists.

    Returns:
        ``True`` if the trajectory satisfies the specification.
    """
    if not states:
        return False
    return bool(_completion_indices(spec, states, 0))


def _completion_indices(spec: Spec, states: Sequence[State], start: int) -> set[int]:
    """Return indices at which ``spec`` can complete starting at ``start``.

    Args:
        spec: The specification node being evaluated.
        states: The trajectory as a sequence of state vectors.
        start: The index at which the spec begins.

    How:
        ``Achieve`` accepts any later index whose state meets the predicate;
        ``Ensuring`` filters the body's completions to those whose whole span
        keeps the safety predicate true; ``Choice`` unions the branches; and
        ``Seq`` folds the steps, letting each step begin where the previous one
        completed.

    Returns:
        The set of completion indices in ``[start, len(states))``.

    Raises:
        TypeError: If ``spec`` is not a known AST node.
    """
    if isinstance(spec, Achieve):
        return {t for t in range(start, len(states)) if spec.predicate.evaluate(states[t])}
    if isinstance(spec, Ensuring):
        ends = _completion_indices(spec.body, states, start)
        return {
            end
            for end in ends
            if all(spec.predicate.evaluate(states[t]) for t in range(start, end + 1))
        }
    if isinstance(spec, Choice):
        ends: set[int] = set()
        for branch in spec.branches:
            ends |= _completion_indices(branch, states, start)
        return ends
    if isinstance(spec, Seq):
        current = {start}
        for step in spec.steps:
            following: set[int] = set()
            for begin in sorted(current):
                following |= _completion_indices(step, states, begin)
            current = following
        return current
    raise TypeError(f"Unsupported specification node: {type(spec).__name__}")

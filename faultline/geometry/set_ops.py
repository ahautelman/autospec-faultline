"""Region algebra for the refinement operators (paper section 4).

ReachRefine and AddRefine intersect a region with a hull; AvoidRefine subtracts a
hull from a safe region; OrRefine intersects two safe regions. Each operation is
a thin, simplifying wrapper over the predicate combinators so the operators read
as region algebra instead of constructor plumbing.
"""

from __future__ import annotations

from faultline.spectrl.predicates import And, Empty, Not, Predicate, Universe


def intersect(left: Predicate, right: Predicate) -> Predicate:
    """Return the region of states that satisfy both operands.

    Args:
        left: First region predicate.
        right: Second region predicate.

    How:
        Collapses the cases with an exact answer: an empty side stays empty and a
        universe side is the other operand. Anything else becomes an ``And`` so
        evaluation short-circuits on the first false side.

    Returns:
        The predicate satisfied by states in both regions.
    """
    if isinstance(left, Empty) or isinstance(right, Empty):
        return Empty()
    if isinstance(left, Universe):
        return right
    if isinstance(right, Universe):
        return left
    return And(left, right)


def difference(left: Predicate, right: Predicate) -> Predicate:
    """Return the region of states in ``left`` but not in ``right``.

    Args:
        left: Region to subtract from.
        right: Region to remove.

    How:
        Collapses the cases with an exact answer: subtracting the universe gives
        :class:`Empty`, subtracting nothing keeps ``left``, and subtracting from
        :class:`Empty` stays empty. Anything else becomes ``left and not right``.

    Returns:
        The predicate satisfied by states in ``left`` and outside ``right``.
    """
    if isinstance(right, Universe) or isinstance(left, Empty):
        return Empty()
    if isinstance(right, Empty):
        return left
    return And(left, Not(right))

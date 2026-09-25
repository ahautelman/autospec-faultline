"""SpectRL specification AST.

Implements the formula layer of the SpectRL grammar from paper section 2.2::

    phi := achieve b | phi ensuring b | phi1 ; phi2 | phi1 or phi2

Nodes wrap the state predicates from :mod:`faultline.spectrl.predicates`.
``Achieve`` is the only leaf; ``Ensuring`` attaches a safety condition to a
sub-spec; ``Seq`` composes steps in order; ``Choice`` is the spec-level
disjunction (the grammar's ``or``, distinct from the predicate-level ``Or``).
Nodes are immutable and hashable so parsed specs can be compared and used as
dictionary keys in tests and fault-injection bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from faultline.spectrl.predicates import Predicate

# Rendering precedence: Achieve binds tightest, then Ensuring, then sequential
# composition, then choice. Drives where the printer inserts parentheses.
_ACHIEVE_PRECEDENCE = 3
_ENSURE_PRECEDENCE = 2
_SEQ_PRECEDENCE = 1
_CHOICE_PRECEDENCE = 0


def _wrap_spec(child: Spec, parent_precedence: int) -> str:
    """Render a child spec, parenthesising it when it binds more loosely.

    Args:
        child: The nested specification to render.
        parent_precedence: Precedence of the enclosing operator.

    How:
        Wrap in parentheses when the child's precedence is lower than the
        parent's, which is exactly when omitting them would change the parse.

    Returns:
        The child rendering, optionally surrounded by parentheses.
    """
    rendered = child.to_string()
    return f"({rendered})" if child.precedence < parent_precedence else rendered


class Spec:
    """Base class for SpectRL specification formulas.

    Subclasses expose :meth:`to_string` for canonical rendering; ``str`` is a
    shorthand for it.
    """

    precedence: ClassVar[int] = _ACHIEVE_PRECEDENCE

    def to_string(self) -> str:
        """Return the canonical SpectRL-syntax rendering of this spec.

        How:
            Abstract hook: each subclass builds its operator-specific text, so the
            base raises until it is overridden.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_string()


@dataclass(frozen=True)
class Achieve(Spec):
    """``achieve b``: reach a state satisfying the predicate ``b``.

    Args:
        predicate: The target-region predicate that must eventually hold.

    How:
        A leaf node; the compiler turns it into a single vertex labelled with
        ``predicate`` and an edge whose safety region is the whole state space.
    """

    predicate: Predicate

    precedence: ClassVar[int] = _ACHIEVE_PRECEDENCE

    def to_string(self) -> str:
        return f"achieve {self.predicate.to_string()}"


@dataclass(frozen=True)
class Ensuring(Spec):
    """``phi ensuring b``: satisfy ``phi`` while keeping ``b`` true throughout.

    Args:
        body: The sub-specification being constrained.
        predicate: The safety predicate that must hold at every step.

    How:
        The compiler compiles ``body`` and intersects ``predicate`` into the
        safety region of every edge that ``body`` introduced.
    """

    body: Spec
    predicate: Predicate

    precedence: ClassVar[int] = _ENSURE_PRECEDENCE

    def to_string(self) -> str:
        body = _wrap_spec(self.body, self.precedence)
        return f"{body} ensuring {self.predicate.to_string()}"


class Seq(Spec):
    """``phi1 ; phi2 ; ...``: satisfy the sub-specs sequentially.

    Args:
        *steps: Two or more sub-specifications, satisfied in the given order.

    How:
        Stores the steps as a tuple and requires at least two entries; the
        compiler chains each step's graph in order. Variadic to match the
        predicate combinators (``And``/``Or``), so ``Seq(a, b)`` reads the same
        way as ``And(a, b)``.
    """

    precedence: ClassVar[int] = _SEQ_PRECEDENCE

    def __init__(self, *steps: Spec) -> None:
        if len(steps) <= 1:
            raise ValueError(f"Seq needs at least 2 steps, got {len(steps)}")
        self.steps = tuple(steps)

    def to_string(self) -> str:
        return " ; ".join(_wrap_spec(step, self.precedence) for step in self.steps)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Seq) and self.steps == other.steps

    def __hash__(self) -> int:
        return hash(("Seq", self.steps))


class Choice(Spec):
    """``phi1 or phi2``: the specification-level disjunction.

    Args:
        *branches: Two or more alternative sub-specifications; satisfying any
            one satisfies the choice.

    How:
        Stores the branches as a tuple and requires at least two entries; the
        compiler builds these as parallel paths through fresh branch vertices.
    """

    precedence: ClassVar[int] = _CHOICE_PRECEDENCE

    def __init__(self, *branches: Spec) -> None:
        if len(branches) <= 1:
            raise ValueError(f"Choice needs at least 2 branches, got {len(branches)}")
        self.branches = tuple(branches)

    def to_string(self) -> str:
        return " or ".join(_wrap_spec(branch, self.precedence) for branch in self.branches)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Choice) and self.branches == other.branches

    def __hash__(self) -> int:
        return hash(("Choice", self.branches))


# The grammar spells this operator ``or``; the alias lets callers match that
# keyword directly while the class name stays distinct from predicates.Or.
Or = Choice

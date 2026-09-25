"""State predicates for SpectRL specifications.

Implements the boolean layer of the SpectRL grammar from paper section 2.2:
``b := a | b1 and b2 | b1 or b2``. A predicate is a pure boolean function over
a state vector. Atomic predicates are geometric (axis-aligned box, disc,
half-plane) and read a chosen subset of the state coordinates; composite
predicates nest atoms with And/Or/Not.

Every predicate is numpy-only and side-effect free so the parser, the compiler,
the 9-rooms environment, and the refinement geometry helpers can share them.
"""

from __future__ import annotations

import numpy as np

State = np.ndarray
Dims = tuple[int, ...]

# The 9-rooms state is (x, y, theta, d): geometric atoms act on (x, y) by
# default, so planar predicates need no explicit dims argument in the common
# case and higher-dimensional states stay supported by passing dims explicitly.
PLANAR_DIMS: Dims = (0, 1)

# Rendering precedence: atoms bind tightest, then Not, And, Or. Used only to
# decide where the printer must insert parentheses for unambiguous output.
_ATOM_PRECEDENCE = 3
_NOT_PRECEDENCE = 2
_AND_PRECEDENCE = 1
_OR_PRECEDENCE = 0


def _select(state: State, dims: Dims) -> State:
    """Project a state vector onto the coordinates a predicate reads.

    Args:
        state: Full state vector of shape (state_dim,).
        dims: Coordinate indices the predicate depends on, in order.

    How:
        Coerce the input to a flat float array and fancy-index the requested
        coordinates, always returning a fresh 1-D array.

    Returns:
        Projected vector of shape (len(dims),).
    """
    array = np.asarray(state, dtype=float).reshape(-1)
    return array[list(dims)]


def _format_number(value: float) -> str:
    """Render a float compactly, dropping the trailing ``.0`` on integers.

    Args:
        value: The number to render.

    How:
        Emit an integer form when the value is integral so ``6.0`` prints as
        ``6``, otherwise fall back to ``repr`` for a round-trippable float.

    Returns:
        A short canonical string for the value.
    """
    return str(int(value)) if float(value).is_integer() else repr(float(value))


class Predicate:
    """Base class for a boolean function over a state vector.

    Subclasses implement :meth:`evaluate` and :meth:`to_string`; calling an
    instance is shorthand for ``evaluate``, and ``str`` renders the canonical
    SpectRL predicate syntax.
    """

    precedence = _ATOM_PRECEDENCE

    def evaluate(self, state: State) -> bool:
        """Return whether ``state`` satisfies this predicate.

        Args:
            state: Full state vector of shape (state_dim,).

        How:
            Abstract hook: each subclass decides membership, so the base raises
            until it is overridden.
        """
        raise NotImplementedError

    def to_string(self) -> str:
        """Return a canonical SpectRL-syntax rendering of this predicate.

        How:
            Abstract hook: each subclass builds its operator-specific text, so the
            base raises until it is overridden.
        """
        raise NotImplementedError

    def __call__(self, state: State) -> bool:
        return self.evaluate(state)

    def __str__(self) -> str:
        return self.to_string()


class Universe(Predicate):
    """The predicate satisfied by every state (the whole state space)."""

    def evaluate(self, state: State) -> bool:
        return True

    def to_string(self) -> str:
        return "true"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Universe)

    def __hash__(self) -> int:
        return hash("Universe")


class Empty(Predicate):
    """The predicate satisfied by no state (the empty set)."""

    def evaluate(self, state: State) -> bool:
        return False

    def to_string(self) -> str:
        return "false"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Empty)

    def __hash__(self) -> int:
        return hash("Empty")


class Box(Predicate):
    """Axis-aligned box: ``lower <= state[dims] <= upper`` componentwise.

    Args:
        lower: Per-coordinate lower bounds, shape (len(dims),).
        upper: Per-coordinate upper bounds, shape (len(dims),).
        dims: State coordinates the box constrains; defaults to (x, y).

    How:
        Validates at construction that both bounds are one-dimensional, match
        the number of dims, and are ordered (``lower <= upper``), so malformed
        regions fail fast. Evaluation projects the state and checks both
        inequalities elementwise.
    """

    def __init__(self, lower, upper, dims: Dims = PLANAR_DIMS) -> None:
        self.lower = np.asarray(lower, dtype=float).reshape(-1)
        self.upper = np.asarray(upper, dtype=float).reshape(-1)
        self.dims = tuple(dims)
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"Box lower/upper lengths differ: {self.lower.shape} vs {self.upper.shape}"
            )
        if self.lower.shape[0] != len(self.dims):
            raise ValueError(
                f"Box has {self.lower.shape[0]} bounds but {len(self.dims)} dims"
            )
        if np.any(self.lower > self.upper):
            raise ValueError(f"Box lower {self.lower} exceeds upper {self.upper}")

    def evaluate(self, state: State) -> bool:
        point = _select(state, self.dims)
        return bool(np.all(point >= self.lower) and np.all(point <= self.upper))

    def to_string(self) -> str:
        bounds = [*self.lower.tolist(), *self.upper.tolist()]
        return "Rect(" + ", ".join(_format_number(v) for v in bounds) + ")"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Box)
            and self.dims == other.dims
            and np.array_equal(self.lower, other.lower)
            and np.array_equal(self.upper, other.upper)
        )

    def __hash__(self) -> int:
        return hash(("Box", self.dims, tuple(self.lower), tuple(self.upper)))


class Disc(Predicate):
    """Closed disc: ``||state[dims] - center|| <= radius``.

    Args:
        center: Centre coordinates, shape (len(dims),).
        radius: Non-negative radius.
        dims: State coordinates the disc constrains; defaults to (x, y).

    How:
        Validates the centre length and non-negative radius at construction;
        evaluation projects the state and compares the Euclidean distance to
        the radius.
    """

    def __init__(self, center, radius: float, dims: Dims = PLANAR_DIMS) -> None:
        self.center = np.asarray(center, dtype=float).reshape(-1)
        self.radius = float(radius)
        self.dims = tuple(dims)
        if self.center.shape[0] != len(self.dims):
            raise ValueError(
                f"Disc centre has {self.center.shape[0]} coords but {len(self.dims)} dims"
            )
        if self.radius < 0:
            raise ValueError(f"Disc radius must be non-negative, got {self.radius}")

    def evaluate(self, state: State) -> bool:
        point = _select(state, self.dims)
        return bool(np.linalg.norm(point - self.center) <= self.radius)

    def to_string(self) -> str:
        coords = ", ".join(_format_number(v) for v in self.center)
        return f"Disc({coords}, {_format_number(self.radius)})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Disc)
            and self.dims == other.dims
            and self.radius == other.radius
            and np.array_equal(self.center, other.center)
        )

    def __hash__(self) -> int:
        return hash(("Disc", self.dims, tuple(self.center), self.radius))


class HalfPlane(Predicate):
    """Closed half-plane: ``normal . state[dims] >= offset``.

    Args:
        normal: Non-zero normal vector, shape (len(dims),).
        offset: Scalar offset along the normal.
        dims: State coordinates the half-plane reads; defaults to (x, y).

    How:
        Rejects a zero normal (which would describe all or no states
        ambiguously) at construction; evaluation projects the state and tests
        the signed projection against the offset.
    """

    def __init__(self, normal, offset: float, dims: Dims = PLANAR_DIMS) -> None:
        self.normal = np.asarray(normal, dtype=float).reshape(-1)
        self.offset = float(offset)
        self.dims = tuple(dims)
        if self.normal.shape[0] != len(self.dims):
            raise ValueError(
                f"HalfPlane normal has {self.normal.shape[0]} coords but {len(self.dims)} dims"
            )
        if np.linalg.norm(self.normal) == 0.0:
            raise ValueError("HalfPlane normal must be non-zero")

    def evaluate(self, state: State) -> bool:
        point = _select(state, self.dims)
        return bool(float(self.normal @ point) >= self.offset)

    def to_string(self) -> str:
        coords = ", ".join(_format_number(v) for v in self.normal)
        return f"HalfPlane({coords}, {_format_number(self.offset)})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HalfPlane)
            and self.dims == other.dims
            and self.offset == other.offset
            and np.array_equal(self.normal, other.normal)
        )

    def __hash__(self) -> int:
        return hash(("HalfPlane", self.dims, tuple(self.normal), self.offset))


def _wrap_child(child: Predicate, parent_precedence: int) -> str:
    """Render a child predicate, parenthesising it when it binds more loosely.

    Args:
        child: The nested predicate to render.
        parent_precedence: Precedence of the enclosing combinator.

    How:
        Wrap in parentheses when the child's precedence is lower than the
        parent's, which is exactly when omitting them would change the parse.

    Returns:
        The child rendering, optionally surrounded by parentheses.
    """
    rendered = child.to_string()
    return f"({rendered})" if child.precedence < parent_precedence else rendered


class And(Predicate):
    """Conjunction of two or more predicates; true only if all are true.

    Args:
        *predicates: At least two child predicates.

    How:
        Stores the children in order (order is preserved for canonical
        rendering) and short-circuits evaluation on the first false child.
    """

    precedence = _AND_PRECEDENCE

    def __init__(self, *predicates: Predicate) -> None:
        if len(predicates) <= 1:
            raise ValueError(f"And needs at least 2 predicates, got {len(predicates)}")
        self.predicates = tuple(predicates)

    def evaluate(self, state: State) -> bool:
        return all(child.evaluate(state) for child in self.predicates)

    def to_string(self) -> str:
        return " ∧ ".join(_wrap_child(c, self.precedence) for c in self.predicates)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, And) and self.predicates == other.predicates

    def __hash__(self) -> int:
        return hash(("And", self.predicates))


class Or(Predicate):
    """Disjunction of two or more predicates; true if any child is true.

    Args:
        *predicates: At least two child predicates.

    How:
        Stores the children in order and short-circuits evaluation on the
        first true child.
    """

    precedence = _OR_PRECEDENCE

    def __init__(self, *predicates: Predicate) -> None:
        if len(predicates) <= 1:
            raise ValueError(f"Or needs at least 2 predicates, got {len(predicates)}")
        self.predicates = tuple(predicates)

    def evaluate(self, state: State) -> bool:
        return any(child.evaluate(state) for child in self.predicates)

    def to_string(self) -> str:
        return " ∨ ".join(_wrap_child(c, self.precedence) for c in self.predicates)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Or) and self.predicates == other.predicates

    def __hash__(self) -> int:
        return hash(("Or", self.predicates))


class Not(Predicate):
    """Negation of a predicate.

    Args:
        predicate: The child predicate to negate.

    How:
        Stores the child and inverts its evaluation result.
    """

    precedence = _NOT_PRECEDENCE

    def __init__(self, predicate: Predicate) -> None:
        self.predicate = predicate

    def evaluate(self, state: State) -> bool:
        return not self.predicate.evaluate(state)

    def to_string(self) -> str:
        return "¬" + _wrap_child(self.predicate, self.precedence)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Not) and self.predicate == other.predicate

    def __hash__(self) -> int:
        return hash(("Not", self.predicate))

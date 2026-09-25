"""SpectRL front end: predicates, AST, parser, and canonical printer.

Re-exports the public surface so callers and tests can import from one place,
for example ``from faultline.spectrl import parse, to_spectrl, Achieve, Seq``.

Naming note: the predicate-level disjunction is :class:`faultline.spectrl.predicates.Or`
and keeps the name ``Or``; the specification-level disjunction is
:class:`faultline.spectrl.ast.Choice` (the grammar's ``or``).
"""

from faultline.spectrl.ast import Achieve, Choice, Ensuring, Seq, Spec
from faultline.spectrl.parser import ParseError, parse, tokenize
from faultline.spectrl.predicates import (
    And,
    Box,
    Disc,
    Empty,
    HalfPlane,
    Not,
    Or,
    Predicate,
    Universe,
)
from faultline.spectrl.printer import to_spectrl
from faultline.spectrl.semantics import models_spec

__all__ = [
    "Achieve",
    "And",
    "Box",
    "Choice",
    "Disc",
    "Empty",
    "Ensuring",
    "HalfPlane",
    "Not",
    "Or",
    "ParseError",
    "Predicate",
    "Seq",
    "Spec",
    "Universe",
    "models_spec",
    "parse",
    "to_spectrl",
    "tokenize",
]

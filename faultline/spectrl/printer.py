"""Canonical serialisation of SpectRL specifications.

Pairs with :mod:`faultline.spectrl.parser`: :func:`~faultline.spectrl.parser.parse`
maps text to an AST and :func:`to_spectrl` maps an AST back to text. The
rendering logic itself lives on the AST and predicate nodes (their
``to_string`` methods), because deciding where parentheses are needed is
knowledge about the grammar's precedence; this module exposes the single public
entry point so callers and tests go through one name.

Output is canonical: ``parse(to_spectrl(spec))`` yields an equal AST, and the
rendering of a given AST is stable. Fault-injection labels and reported
ground-truth repairs rely on that stability.
"""

from __future__ import annotations

from faultline.spectrl.ast import Spec


def to_spectrl(spec: Spec) -> str:
    """Render a specification AST as canonical SpectRL text.

    Args:
        spec: The specification AST to render.

    How:
        Delegates to the AST's ``to_string``, which walks the tree and applies
        the grammar's precedence to insert parentheses only where required.

    Returns:
        A single-line SpectRL rendering that parses back to an equal AST.
    """
    return spec.to_string()

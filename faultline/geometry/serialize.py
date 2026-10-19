"""JSON round-tripping for region predicates.

A refinement trace has to be redrawn from disk, so every region the operators can
produce is turned into plain tagged dictionaries and back. It lives in
``geometry`` rather than ``spectrl`` because :class:`ConvexHull` is defined here
and ``geometry`` already depends on ``spectrl``, not the other way round.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from faultline.geometry.convex_hull import ConvexHull
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

Payload = dict[str, Any]


def predicate_to_dict(predicate: Predicate) -> Payload:
    """Convert a region predicate to a JSON-serialisable mapping.

    Args:
        predicate: Any predicate the refinement operators can build.

    How:
        Looks the concrete class up in the encoder table and recurses into the
        children of ``And``, ``Or`` and ``Not``. Coordinates become plain lists.

    Returns:
        A mapping with a ``type`` tag plus the predicate's parameters.

    Raises:
        TypeError: If the predicate class has no encoder.
    """
    encoder = _ENCODERS.get(type(predicate))
    if encoder is None:
        raise TypeError(f"Cannot serialise predicate of type {type(predicate).__name__}")
    return encoder(predicate)


def predicate_from_dict(payload: Mapping[str, Any]) -> Predicate:
    """Rebuild a predicate from :func:`predicate_to_dict`'s output.

    Args:
        payload: The mapping read back from JSON.

    How:
        Dispatches on the ``type`` tag, recursing into composite children, so the
        result compares equal to the original predicate.

    Returns:
        The reconstructed predicate.

    Raises:
        ValueError: If the tag is missing or unknown.
    """
    decoder = _DECODERS.get(payload.get("type"))
    if decoder is None:
        raise ValueError(f"Unknown predicate payload: {dict(payload)!r}")
    return decoder(payload)


def _box_to_dict(box: Box) -> Payload:
    """Encode a box as its bounds and dims."""
    return {
        "type": "box",
        "lower": box.lower.tolist(),
        "upper": box.upper.tolist(),
        "dims": list(box.dims),
    }


def _disc_to_dict(disc: Disc) -> Payload:
    """Encode a disc as its centre, radius and dims."""
    return {
        "type": "disc",
        "center": disc.center.tolist(),
        "radius": disc.radius,
        "dims": list(disc.dims),
    }


def _half_plane_to_dict(half_plane: HalfPlane) -> Payload:
    """Encode a half-plane as its normal, offset and dims."""
    return {
        "type": "half_plane",
        "normal": half_plane.normal.tolist(),
        "offset": half_plane.offset,
        "dims": list(half_plane.dims),
    }


def _hull_to_dict(hull: ConvexHull) -> Payload:
    """Encode a convex hull as its vertex list and dims."""
    return {"type": "hull", "vertices": hull.vertices.tolist(), "dims": list(hull.dims)}


def _composite_to_dict(tag: str) -> Callable[[And | Or], Payload]:
    """Build the encoder for an n-ary combinator with the given tag."""

    def encode(predicate: And | Or) -> Payload:
        return {"type": tag, "predicates": [predicate_to_dict(c) for c in predicate.predicates]}

    return encode


def _not_to_dict(negation: Not) -> Payload:
    """Encode a negation as its child."""
    return {"type": "not", "predicate": predicate_to_dict(negation.predicate)}


_ENCODERS: dict[type, Callable[[Any], Payload]] = {
    Box: _box_to_dict,
    Disc: _disc_to_dict,
    HalfPlane: _half_plane_to_dict,
    ConvexHull: _hull_to_dict,
    And: _composite_to_dict("and"),
    Or: _composite_to_dict("or"),
    Not: _not_to_dict,
    Universe: lambda _: {"type": "universe"},
    Empty: lambda _: {"type": "empty"},
}

_DECODERS: dict[str | None, Callable[[Mapping[str, Any]], Predicate]] = {
    "box": lambda p: Box(p["lower"], p["upper"], tuple(p["dims"])),
    "disc": lambda p: Disc(p["center"], p["radius"], tuple(p["dims"])),
    "half_plane": lambda p: HalfPlane(p["normal"], p["offset"], tuple(p["dims"])),
    "hull": lambda p: ConvexHull(p["vertices"], tuple(p["dims"])),
    "and": lambda p: And(*(predicate_from_dict(c) for c in p["predicates"])),
    "or": lambda p: Or(*(predicate_from_dict(c) for c in p["predicates"])),
    "not": lambda p: Not(predicate_from_dict(p["predicate"])),
    "universe": lambda _: Universe(),
    "empty": lambda _: Empty(),
}

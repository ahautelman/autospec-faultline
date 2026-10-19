"""Round-trip tests for region predicate serialisation.

Each predicate the refinement operators can build must survive a trip through
JSON text unchanged, since a trace is redrawn from disk after training.
"""

from __future__ import annotations

import json

import pytest

from faultline.geometry import convex_hull, difference, intersect, predicate_from_dict, predicate_to_dict
from faultline.spectrl.predicates import And, Box, Disc, Empty, HalfPlane, Not, Or, Universe

SQUARE = Box((0.0, 0.0), (4.0, 4.0))
HULL = convex_hull([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (1.0, 1.0)])

REGIONS = {
    "box": SQUARE,
    "disc": Disc((2.0, 2.0), 1.5),
    "half_plane": HalfPlane((1.0, -1.0), 0.5),
    "hull": HULL,
    "universe": Universe(),
    "empty": Empty(),
    "not": Not(SQUARE),
    "and": And(SQUARE, HULL),
    "or": Or(SQUARE, Disc((9.0, 9.0), 1.0)),
    "refinement_shape": difference(intersect(SQUARE, HULL), Box((1.0, 1.0), (2.0, 2.0))),
}


@pytest.mark.parametrize("region", REGIONS.values(), ids=REGIONS.keys())
def test_region_survives_a_json_text_round_trip(region):
    """Encoding to JSON text and decoding yields an equal predicate."""
    restored = predicate_from_dict(json.loads(json.dumps(predicate_to_dict(region))))
    assert restored == region


def test_round_tripped_region_keeps_its_membership():
    """The decoded refinement shape accepts and rejects the same states."""
    region = REGIONS["refinement_shape"]
    restored = predicate_from_dict(predicate_to_dict(region))
    assert restored.evaluate([0.5, 0.5]) is True
    assert restored.evaluate([1.5, 1.5]) is False


def test_non_planar_dims_are_preserved():
    """A box over other state coordinates keeps its dims."""
    region = Box((0.0,), (1.0,), dims=(2,))
    assert predicate_from_dict(predicate_to_dict(region)).dims == (2,)


def test_unknown_payload_tag_is_rejected_with_context():
    """A payload with an unrecognised tag names the offending mapping."""
    with pytest.raises(ValueError, match="Unknown predicate payload"):
        predicate_from_dict({"type": "blob"})


def test_unserialisable_predicate_is_rejected():
    """A predicate class without an encoder raises a TypeError."""

    class Strange(Box):
        pass

    with pytest.raises(TypeError, match="Strange"):
        predicate_to_dict(Strange((0.0, 0.0), (1.0, 1.0)))

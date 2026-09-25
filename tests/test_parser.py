"""Behavioural tests for the SpectRL parser and canonical printer.

Covers precedence and associativity of both spec and predicate operators,
region-name resolution, literal constructors, malformed-input errors, and the
parse/print/parse round trip that fault-injection labelling depends on.
"""

from __future__ import annotations

import pytest

from faultline.spectrl import (
    Achieve,
    Choice,
    Ensuring,
    ParseError,
    Seq,
    parse,
    to_spectrl,
)
from faultline.spectrl.predicates import And, Box, Not, Or, Universe

REGIONS = {
    "a": Box((0.0, 0.0), (1.0, 1.0)),
    "b": Box((2.0, 0.0), (3.0, 1.0)),
    "c": Box((0.0, 2.0), (1.0, 3.0)),
    "s": Box((0.0, 0.0), (9.0, 9.0)),
}


def test_parses_achieve_with_a_named_region():
    """A bare identifier resolves to the predicate in the region mapping."""
    assert parse("achieve a", REGIONS) == Achieve(REGIONS["a"])


def test_parses_sequence_of_achieves():
    """Two achieves joined by a semicolon form a sequential spec."""
    expected = Seq(Achieve(REGIONS["a"]), Achieve(REGIONS["b"]))
    assert parse("achieve a ; achieve b", REGIONS) == expected


def test_semicolon_binds_tighter_than_or():
    """The spec-level or is loosest, so the sequence groups before the choice."""
    expected = Choice(Seq(Achieve(REGIONS["a"]), Achieve(REGIONS["b"])), Achieve(REGIONS["c"]))
    assert parse("achieve a ; achieve b or achieve c", REGIONS) == expected


def test_ensuring_binds_tighter_than_semicolon():
    """An ensuring clause attaches to the preceding spec, not the whole sequence."""
    expected = Seq(Ensuring(Achieve(REGIONS["a"]), REGIONS["s"]), Achieve(REGIONS["b"]))
    assert parse("achieve a ensuring s ; achieve b", REGIONS) == expected


def test_parentheses_group_choice_inside_sequence():
    """Parentheses let a choice appear as one step of a sequence."""
    expected = Seq(Achieve(REGIONS["a"]), Choice(Achieve(REGIONS["b"]), Achieve(REGIONS["c"])))
    assert parse("achieve a ; (achieve b or achieve c)", REGIONS) == expected


def test_repeated_ensuring_nests_leftwards():
    """Two ensuring clauses nest, the later one wrapping the earlier result."""
    expected = Ensuring(Ensuring(Achieve(REGIONS["a"]), REGIONS["s"]), REGIONS["s"])
    assert parse("achieve a ensuring s ensuring s", REGIONS) == expected


def test_predicate_and_binds_tighter_than_predicate_or():
    """Within a predicate, conjunction groups before disjunction."""
    expected = Achieve(Or(And(REGIONS["a"], REGIONS["b"]), REGIONS["c"]))
    assert parse("achieve a ∧ b ∨ c", REGIONS) == expected


def test_predicate_not_binds_tightest():
    """Negation applies to a single atom, not the whole conjunction."""
    expected = Achieve(And(Not(REGIONS["a"]), REGIONS["b"]))
    assert parse("achieve ¬ a ∧ b", REGIONS) == expected


def test_ascii_operator_aliases_match_unicode():
    """The ASCII aliases & and | parse the same as ∧ and ∨."""
    assert parse("achieve a & b | c", REGIONS) == parse("achieve a ∧ b ∨ c", REGIONS)


def test_parses_rect_literal_without_region_table():
    """A Rect literal builds a box without needing any region mapping."""
    assert parse("achieve Rect(0, 0, 6, 6)") == Achieve(Box((0.0, 0.0), (6.0, 6.0)))


def test_parses_truth_literals():
    """The keywords true and false map to Universe and Empty."""
    assert parse("achieve true") == Achieve(Universe())


def test_unknown_region_raises_with_known_names():
    """An unmapped identifier fails and the error names the known regions."""
    with pytest.raises(ParseError, match="Unknown region 'zzz'"):
        parse("achieve zzz", REGIONS)


def test_invalid_rect_literal_raises_parse_error():
    """A Rect with inverted corners surfaces as a ParseError, not a raw ValueError."""
    with pytest.raises(ParseError, match="Invalid Rect"):
        parse("achieve Rect(5, 5, 1, 1)")


@pytest.mark.parametrize(
    "text",
    [
        "achieve a",
        "achieve a ; achieve b",
        "achieve a ensuring s ; achieve b ensuring s",
        "achieve a ; achieve b or achieve c",
        "(achieve a ; achieve b) or achieve c",
        "achieve ¬ a ∧ b ∨ c",
        "achieve Rect(0, 0, 6, 6)",
        "achieve true",
    ],
)
def test_parse_print_parse_round_trips(text):
    """Re-parsing the canonical rendering yields an equal AST."""
    first_parse = parse(text, REGIONS)
    assert parse(to_spectrl(first_parse), REGIONS) == first_parse


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "achieve",
        "achieve a ;",
        "achieve a achieve b",
        "(achieve a",
        "achieve a ensuring",
        "achieve a ∧",
        "achieve a @ b",
        "achieve a or b",
    ],
)
def test_malformed_input_raises_parse_error(text):
    """Invalid specifications fail with a ParseError rather than partial output."""
    with pytest.raises(ParseError):
        parse(text, REGIONS)

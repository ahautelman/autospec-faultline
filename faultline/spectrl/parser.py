"""Recursive-descent parser for SpectRL specifications.

Translates the textual SpectRL grammar (paper section 2.2) into the AST from
:mod:`faultline.spectrl.ast`. Operator precedence, loosest first, is
``or`` < ``;`` < ``ensuring`` < ``achieve``; predicate connectives bind as
``∨`` < ``∧`` < ``¬`` and must be written with symbols (``∨``/``|``, ``∧``/``&``,
``¬``/``!``) because the word ``or`` is reserved for the specification-level
choice.

Bare identifiers name regions and are resolved through a caller-supplied
mapping, so a spec can refer to rooms such as ``start`` or ``goal``. Atomic
predicates can also be written literally as ``Rect(x0, y0, x1, y1)``,
``Disc(cx, cy, r)``, or ``HalfPlane(nx, ny, offset)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from faultline.spectrl.ast import Achieve, Choice, Ensuring, Seq, Spec
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

_WHITESPACE = " \t\r\n"
_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_TOKEN_PATTERN = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<symbol>[();,&|!∧∨¬])"
)

_AND_SYMBOLS = {"∧", "&"}
_OR_SYMBOLS = {"∨", "|"}
_NOT_SYMBOLS = {"¬", "!"}
_LPAREN = "("
_RPAREN = ")"
_SEMICOLON = ";"
_COMMA = ","

# Literal predicate constructors and the number of arguments each requires.
_CONSTRUCTORS = {"Rect": 4, "Disc": 3, "HalfPlane": 3}

_ENSURING = "ensuring"
_ACHIEVE = "achieve"
_OR_KEYWORD = "or"
_TRUE = "true"
_FALSE = "false"


class ParseError(ValueError):
    """Raised when a specification text is not syntactically valid."""


@dataclass(frozen=True)
class Token:
    """A lexical token together with its source offset.

    Args:
        kind: One of ``"number"``, ``"ident"``, or ``"symbol"``.
        text: The matched source text.
        position: Zero-based character offset of the token's first character.
    """

    kind: str
    text: str
    position: int


def tokenize(text: str) -> list[Token]:
    """Split specification text into an ordered token list.

    Args:
        text: The specification source.

    How:
        Scans left to right with one regex, skipping whitespace; a character
        that matches no token raises :class:`ParseError` naming its offset.

    Returns:
        The tokens in source order.

    Raises:
        ParseError: If a character cannot start any valid token.
    """
    tokens: list[Token] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] in _WHITESPACE:
            cursor += 1
            continue
        match = _TOKEN_PATTERN.match(text, cursor)
        if match is None:
            raise ParseError(f"Unexpected character {text[cursor]!r} at offset {cursor}")
        tokens.append(Token(match.lastgroup, match.group(), match.start()))
        cursor = match.end()
    return tokens


def _build_constructor(name: str, args: Sequence[float]) -> Predicate:
    """Build a literal atomic predicate from parsed numeric arguments.

    Args:
        name: A key of ``_CONSTRUCTORS``.
        args: Numeric arguments in source order.

    How:
        ``Rect`` reads the first and second coordinate pairs as the lower and
        upper corners of a planar box; ``Disc`` and ``HalfPlane`` pass their
        centre/radius and normal/offset through. A domain error from the
        predicate constructor is re-raised as a :class:`ParseError`.

    Returns:
        The constructed atomic predicate.

    Raises:
        ParseError: If the arguments do not describe a valid region.
    """
    try:
        if name == "Rect":
            return Box((args[0], args[1]), (args[2], args[3]))
        if name == "Disc":
            return Disc((args[0], args[1]), args[2])
        return HalfPlane((args[0], args[1]), args[2])
    except ValueError as error:
        raise ParseError(f"Invalid {name} predicate: {error}") from error


class _Parser:
    """Mutable recursive-descent state over a token list.

    Args:
        tokens: Tokens produced by :func:`tokenize`.
        regions: Mapping from region names to predicates for identifier lookup.
    """

    def __init__(self, tokens: Sequence[Token], regions: Mapping[str, Predicate]) -> None:
        self._tokens = list(tokens)
        self._regions = dict(regions)
        self._index = 0

    def _peek(self) -> Token | None:
        """Return the next token without consuming it.

        How:
            Reads the token at the current cursor, or ``None`` at end of input.

        Returns:
            The upcoming token, or ``None`` if the input is exhausted.
        """
        if self._index < len(self._tokens):
            return self._tokens[self._index]
        return None

    def _at_end(self) -> bool:
        """Return whether every token has been consumed.

        How:
            Compares the cursor against the token count.

        Returns:
            ``True`` when no tokens remain.
        """
        return self._index >= len(self._tokens)

    def _advance(self) -> Token:
        """Consume and return the next token.

        How:
            Reads the token at the cursor and moves the cursor past it.

        Returns:
            The consumed token.
        """
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _describe_next(self) -> str:
        """Render the upcoming token for an error message.

        How:
            Returns ``"end of input"`` when the cursor is exhausted, otherwise
            the quoted token text.

        Returns:
            A short human-readable description of what comes next.
        """
        token = self._peek()
        return "end of input" if token is None else repr(token.text)

    def _match_symbol(self, symbols: set[str]) -> Token | None:
        """Consume the next token if it is one of ``symbols``.

        Args:
            symbols: Acceptable symbol texts.

        How:
            Peeks at the next token and consumes it only when it is a symbol in
            the set, so callers can test for optional punctuation.

        Returns:
            The consumed token, or ``None`` when there was no match.
        """
        token = self._peek()
        if token is not None and token.kind == "symbol" and token.text in symbols:
            return self._advance()
        return None

    def _match_keyword(self, keyword: str) -> Token | None:
        """Consume the next token if it equals ``keyword``.

        Args:
            keyword: The reserved word to look for.

        How:
            Peeks at the next token and consumes it only when it is an
            identifier equal to the keyword.

        Returns:
            The consumed token, or ``None`` when there was no match.
        """
        token = self._peek()
        if token is not None and token.kind == "ident" and token.text == keyword:
            return self._advance()
        return None

    def _expect_symbol(self, symbol: str) -> Token:
        """Consume a required symbol or fail.

        Args:
            symbol: The required symbol text.

        How:
            Delegates to :meth:`_match_symbol`; on failure raises a
            :class:`ParseError` that names what was expected and found.

        Returns:
            The consumed token.

        Raises:
            ParseError: If the next token is not the required symbol.
        """
        token = self._match_symbol({symbol})
        if token is None:
            raise ParseError(f"Expected {symbol!r} but found {self._describe_next()}")
        return token

    def _expect_keyword(self, keyword: str) -> Token:
        """Consume a required keyword or fail.

        Args:
            keyword: The required reserved word.

        How:
            Delegates to :meth:`_match_keyword`; on failure raises a
            :class:`ParseError` that names what was expected and found.

        Returns:
            The consumed token.

        Raises:
            ParseError: If the next token is not the required keyword.
        """
        token = self._match_keyword(keyword)
        if token is None:
            raise ParseError(f"Expected keyword {keyword!r} but found {self._describe_next()}")
        return token

    def parse_spec(self) -> Spec:
        """Parse a complete specification and require end of input.

        How:
            Parses a choice, then rejects any trailing token so partial input
            such as ``"achieve a ;"`` cannot be silently accepted.

        Returns:
            The parsed specification AST.

        Raises:
            ParseError: If tokens remain after the specification.
        """
        spec = self._parse_choice()
        if not self._at_end():
            token = self._peek()
            raise ParseError(f"Unexpected {token.text!r} at offset {token.position}")
        return spec

    def _parse_choice(self) -> Spec:
        """Parse ``spec ("or" spec)*`` into a :class:`Choice`.

        How:
            Parses sequential specs separated by the ``or`` keyword, returning
            the bare operand when there is only one.

        Returns:
            A :class:`Choice` node, or the single parsed spec.
        """
        branches = [self._parse_seq()]
        while self._match_keyword(_OR_KEYWORD) is not None:
            branches.append(self._parse_seq())
        return branches[0] if len(branches) == 1 else Choice(*branches)

    def _parse_seq(self) -> Spec:
        """Parse ``ensure (";" ensure)*`` into a :class:`Seq`.

        How:
            Parses ensuring-specs joined by ``;``, returning the bare operand
            when there is only one.

        Returns:
            A :class:`Seq` node, or the single parsed spec.
        """
        steps = [self._parse_ensure()]
        while self._match_symbol({_SEMICOLON}) is not None:
            steps.append(self._parse_ensure())
        return steps[0] if len(steps) == 1 else Seq(*steps)

    def _parse_ensure(self) -> Spec:
        """Parse ``primary ("ensuring" predicate)*`` into nested safeguards.

        How:
            Parses a primary spec, then wraps it once per trailing ``ensuring``
            clause so later clauses apply to the whole preceding spec.

        Returns:
            The (possibly wrapped) specification.
        """
        spec = self._parse_primary_spec()
        while self._match_keyword(_ENSURING) is not None:
            spec = Ensuring(spec, self._parse_predicate())
        return spec

    def _parse_primary_spec(self) -> Spec:
        """Parse an ``achieve`` spec or a parenthesised specification.

        How:
            A leading ``(`` starts a nested choice closed by ``)``; otherwise an
            ``achieve`` keyword is required, followed by a predicate.

        Returns:
            The parsed specification.

        Raises:
            ParseError: If neither form is present.
        """
        if self._match_symbol({_LPAREN}) is not None:
            spec = self._parse_choice()
            self._expect_symbol(_RPAREN)
            return spec
        self._expect_keyword(_ACHIEVE)
        return Achieve(self._parse_predicate())

    def _parse_predicate(self) -> Predicate:
        """Parse a predicate, starting at the loosest connective.

        How:
            Delegates to the disjunction level so ``∨`` binds loosest.

        Returns:
            The parsed predicate.
        """
        return self._parse_predicate_or()

    def _parse_predicate_or(self) -> Predicate:
        """Parse ``and-expr ("∨" and-expr)*`` into an :class:`Or`.

        How:
            Collects conjunction operands split by ``∨``/``|`` and returns the
            bare operand when there is only one.

        Returns:
            An :class:`Or` predicate, or the single parsed operand.
        """
        parts = [self._parse_predicate_and()]
        while self._match_symbol(_OR_SYMBOLS) is not None:
            parts.append(self._parse_predicate_and())
        return parts[0] if len(parts) == 1 else Or(*parts)

    def _parse_predicate_and(self) -> Predicate:
        """Parse ``not-expr ("∧" not-expr)*`` into an :class:`And`.

        How:
            Collects negation operands split by ``∧``/``&`` and returns the bare
            operand when there is only one.

        Returns:
            An :class:`And` predicate, or the single parsed operand.
        """
        parts = [self._parse_predicate_not()]
        while self._match_symbol(_AND_SYMBOLS) is not None:
            parts.append(self._parse_predicate_not())
        return parts[0] if len(parts) == 1 else And(*parts)

    def _parse_predicate_not(self) -> Predicate:
        """Parse ``("¬" not-expr) | atom`` into a :class:`Not` or atom.

        How:
            Consumes any run of negation symbols, recursing so ``¬¬b`` nests,
            then delegates to atom parsing.

        Returns:
            The parsed predicate.
        """
        if self._match_symbol(_NOT_SYMBOLS) is not None:
            return Not(self._parse_predicate_not())
        return self._parse_predicate_atom()

    def _parse_predicate_atom(self) -> Predicate:
        """Parse a parenthesised predicate, truth literal, or named region.

        How:
            A leading ``(`` parses a nested predicate; otherwise an identifier
            is checked against the truth literals and predicate constructors
            before being resolved as a region name.

        Returns:
            The parsed atomic predicate.

        Raises:
            ParseError: If no identifier is present or a region is unknown.
        """
        if self._match_symbol({_LPAREN}) is not None:
            predicate = self._parse_predicate()
            self._expect_symbol(_RPAREN)
            return predicate
        token = self._peek()
        if token is None or token.kind != "ident":
            raise ParseError(f"Expected a predicate but found {self._describe_next()}")
        name = self._advance().text
        if name == _TRUE:
            return Universe()
        if name == _FALSE:
            return Empty()
        if name in _CONSTRUCTORS:
            return self._parse_constructor(name)
        return self._resolve_region(name, token.position)

    def _resolve_region(self, name: str, position: int) -> Predicate:
        """Look up a region name in the caller-supplied mapping.

        Args:
            name: The identifier naming the region.
            position: Source offset of the identifier, used in error messages.

        How:
            Returns the mapped predicate, or raises a :class:`ParseError` that
            lists the regions that were made available.

        Returns:
            The predicate bound to ``name``.

        Raises:
            ParseError: If the name is not a known region.
        """
        if name not in self._regions:
            known = ", ".join(sorted(self._regions)) or "none"
            raise ParseError(
                f"Unknown region {name!r} at offset {position}; known regions: {known}"
            )
        return self._regions[name]

    def _parse_constructor(self, name: str) -> Predicate:
        """Parse a literal predicate constructor call.

        Args:
            name: The constructor name, already known to be in ``_CONSTRUCTORS``.

        How:
            Reads the fixed number of comma-separated numeric arguments for the
            constructor and builds the predicate via :func:`_build_constructor`.

        Returns:
            The constructed predicate.

        Raises:
            ParseError: If the argument list is malformed.
        """
        self._expect_symbol(_LPAREN)
        arg_count = _CONSTRUCTORS[name]
        args = [self._parse_number()]
        for _ in range(arg_count - 1):
            self._expect_symbol(_COMMA)
            args.append(self._parse_number())
        self._expect_symbol(_RPAREN)
        return _build_constructor(name, args)

    def _parse_number(self) -> float:
        """Parse a numeric literal.

        How:
            Requires the next token to be a number and converts it to float.

        Returns:
            The parsed value.

        Raises:
            ParseError: If the next token is not a number.
        """
        token = self._peek()
        if token is None or token.kind != "number":
            raise ParseError(f"Expected a number but found {self._describe_next()}")
        return float(self._advance().text)


def parse(text: str, regions: Mapping[str, Predicate] | None = None) -> Spec:
    """Parse SpectRL specification text into an AST.

    Args:
        text: Specification source, e.g. ``"achieve start ; achieve goal"``.
        regions: Optional region-name to predicate mapping used to resolve bare
            identifiers in the text.

    How:
        Tokenises the input and runs a recursive-descent parse that mirrors the
        grammar's precedence, then requires end of input so trailing tokens are
        rejected.

    Returns:
        The parsed :class:`~faultline.spectrl.ast.Spec`.

    Raises:
        ParseError: If the text is empty or not a valid specification.
    """
    if not text.strip():
        raise ParseError("Empty specification text")
    return _Parser(tokenize(text), regions or {}).parse_spec()
